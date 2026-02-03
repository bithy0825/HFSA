import torch
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial
import torchvision.utils as vutils
import os

from .tools import make_coord
from .common import Averager


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == "benchmark":
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == "div2k":
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)


def calc_ssim(sr, hr, dataset=None, scale=1, rgb_range=1):
    if dataset is not None:
        if dataset == "benchmark":
            shave = scale
            if sr.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = sr.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                sr = sr.mul(convert).sum(dim=1, keepdim=True)
                hr = hr.mul(convert).sum(dim=1, keepdim=True)
        elif dataset == "div2k":
            shave = scale + 6
        else:
            raise NotImplementedError
        sr = sr[..., shave:-shave, shave:-shave]
        hr = hr[..., shave:-shave, shave:-shave]

    C1 = (0.01 * rgb_range) ** 2
    C2 = (0.03 * rgb_range) ** 2

    def gaussian(window_size, sigma):
        gauss = torch.Tensor(
            [
                torch.exp(
                    torch.tensor(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                )
                for x in range(window_size)
            ]
        )
        return gauss / gauss.sum()

    window_size = 11
    sigma = 1.5
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(sr.size(1), 1, window_size, window_size).to(sr.device)

    mu1 = F.conv2d(sr, window, padding=window_size // 2, groups=sr.size(1))
    mu2 = F.conv2d(hr, window, padding=window_size // 2, groups=hr.size(1))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(sr * sr, window, padding=window_size // 2, groups=sr.size(1)) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(hr * hr, window, padding=window_size // 2, groups=hr.size(1)) - mu2_sq
    )
    sigma12 = (
        F.conv2d(sr * hr, window, padding=window_size // 2, groups=sr.size(1)) - mu1_mu2
    )

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        B, H, W, _ = coord.shape
        preds = []

        for start_idx in range(0, H, bsize):
            end_idx = min(start_idx + bsize, H)
            coord_ = coord[:, start_idx:end_idx, :, :].view(B, -1, 2)
            cell_ = cell[:, start_idx:end_idx, :, :].view(B, -1, 2)
            pred = model.query_rgb(coord_, cell_).view(B, -1, W, inp.shape[1])
            preds.append(pred)

        pred = torch.cat(preds, dim=1)
        del preds
    return pred.permute(0, 3, 1, 2)  # B, C, H, W


def batched_predict_fast(model, inp, coord, cell, bsize, confidence_fixed=None, offset_scale=1.0):
    with torch.no_grad():
        model.gen_feat(inp)
        B, H, W, _ = coord.shape
        preds = []

        for start_idx in range(0, H, bsize):
            end_idx = min(start_idx + bsize, H)
            coord_ = coord[:, start_idx:end_idx, :, :].view(B, -1, 2)
            cell_ = cell[:, start_idx:end_idx, :, :].view(B, -1, 2)
            pred = model.query_rgb(coord_, cell_, confidence_fixed, offset_scale).view(B, -1, W, inp.shape[1])
            preds.append(pred)

        pred = torch.cat(preds, dim=1)
        del preds
    return pred.permute(0, 3, 1, 2)  # B, C, H, W


def eval_psnr(
    loader,
    model,
    data_norm=None,
    eval_type=None,
    eval_bsize=None,
    window_size=None,
    limit_cell=None,
    verbose=False,
):
    model.eval()
    device = next(model.parameters()).device

    if data_norm is None:
        data_norm = {"inp": {"sub": [0], "div": [1]}, "gt": {"sub": [0], "div": [1]}}
    inp_sub = torch.FloatTensor(data_norm["inp"]["sub"]).view(1, -1, 1, 1).to(device)
    inp_div = torch.FloatTensor(data_norm["inp"]["div"]).view(1, -1, 1, 1).to(device)
    gt_sub = torch.FloatTensor(data_norm["gt"]["sub"]).view(1, 1, -1).to(device)
    gt_div = torch.FloatTensor(data_norm["gt"]["div"]).view(1, 1, -1).to(device)

    if eval_type is None:
        metric_fn = calc_psnr
    elif eval_type.startswith("div2k"):
        scale = int(eval_type.split("-")[1])
        metric_fn = partial(calc_psnr, dataset="div2k", scale=scale)
    elif eval_type.startswith("benchmark"):
        scale = int(eval_type.split("-")[1])
        metric_fn = partial(calc_psnr, dataset="benchmark", scale=scale)
    else:
        raise NotImplementedError

    val_res = Averager()

    for batch in tqdm(loader, leave=False, desc="VAL", disable=True):
        for k, v in batch.items():
            batch[k] = v.to(device)

        inp = (batch["inp"] - inp_sub) / inp_div
        # SwinIR Evaluation - reflection padding
        if window_size is not None:
            _, _, h_old, w_old = inp.size()
            h_pad = (window_size - h_old % window_size) % window_size
            w_pad = (window_size - w_old % window_size) % window_size
            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, : h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[:, :, :, : w_old + w_pad]

            coord = (
                make_coord(
                    (scale * (h_old + h_pad), scale * (w_old + w_pad)), flatten=False
                )
                .unsqueeze(0)
                .to(device)
            )
            cell = torch.ones_like(coord)
            cell[..., 0] *= 2 / inp.shape[-2] / scale
            cell[..., 1] *= 2 / inp.shape[-1] / scale
        else:
            h_pad = 0
            w_pad = 0

            coord = batch["coord"]
            cell = batch["cell"]

        if limit_cell is not None:
            cell = cell * max(scale / limit_cell, 1)

        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, coord, cell)
        else:
            with torch.no_grad():
                # pred = batched_predict_fast(model, inp, coord, cell, eval_bsize)
                pred = batched_predict(model, inp, coord, cell, eval_bsize)

        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if window_size is not None:
            shape = [3, round(scale * (h_old + h_pad)), round(scale * (w_old + w_pad))]
        else:
            shape = [3, batch["inp"].shape[-2] * scale, batch["inp"].shape[-1] * scale]

        pred = pred.view(shape).contiguous()
        pred = pred[..., : batch["inp"].shape[-2] * scale, : batch["inp"].shape[-1] * scale]

        res = metric_fn(pred, batch["gt"])
        val_res.add(res.item(), batch["inp"].shape[0])
        if verbose:
            tqdm.write("val {:.6f}".format(val_res.item()))

    return val_res.item()


def eval_ssim(
    loader,
    model,
    data_norm=None,
    eval_type=None,
    eval_bsize=None,
    window_size=None,
    limit_cell=None,
    verbose=False,
):
    model.eval()
    device = next(model.parameters()).device

    if data_norm is None:
        data_norm = {"inp": {"sub": [0], "div": [1]}, "gt": {"sub": [0], "div": [1]}}
    inp_sub = torch.FloatTensor(data_norm["inp"]["sub"]).view(1, -1, 1, 1).to(device)
    inp_div = torch.FloatTensor(data_norm["inp"]["div"]).view(1, -1, 1, 1).to(device)
    gt_sub = torch.FloatTensor(data_norm["gt"]["sub"]).view(1, 1, -1).to(device)
    gt_div = torch.FloatTensor(data_norm["gt"]["div"]).view(1, 1, -1).to(device)

    if eval_type is None:
        metric_fn = calc_ssim
    elif eval_type.startswith("div2k"):
        scale = int(eval_type.split("-")[1])
        metric_fn = partial(calc_ssim, dataset="div2k", scale=scale)
    elif eval_type.startswith("benchmark"):
        scale = int(eval_type.split("-")[1])
        metric_fn = partial(calc_ssim, dataset="benchmark", scale=scale)
    else:
        raise NotImplementedError

    val_res = Averager()

    for batch in tqdm(loader, leave=False, desc="VAL", disable=True):
        for k, v in batch.items():
            batch[k] = v.to(device)

        inp = (batch["inp"] - inp_sub) / inp_div
        # SwinIR Evaluation - reflection padding
        if window_size is not None:
            _, _, h_old, w_old = inp.size()
            h_pad = (window_size - h_old % window_size) % window_size
            w_pad = (window_size - w_old % window_size) % window_size
            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, : h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[:, :, :, : w_old + w_pad]

            coord = (
                make_coord(
                    (scale * (h_old + h_pad), scale * (w_old + w_pad)), flatten=False
                )
                .unsqueeze(0)
                .to(device)
            )
            cell = torch.ones_like(coord)
            cell[..., 0] *= 2 / inp.shape[-2] / scale
            cell[..., 1] *= 2 / inp.shape[-1] / scale
        else:
            h_pad = 0
            w_pad = 0

            coord = batch["coord"]
            cell = batch["cell"]

        if limit_cell is not None:
            cell = cell * max(scale / limit_cell, 1)

        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, coord, cell)
        else:
            with torch.no_grad():
                # pred = batched_predict_fast(model, inp, coord, cell, eval_bsize)
                pred = batched_predict(model, inp, coord, cell, eval_bsize)

        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if window_size is not None:
            shape = [1,3, round(scale * (h_old + h_pad)), round(scale * (w_old + w_pad))]
        else:
            shape = [1,3, batch["inp"].shape[-2] * scale, batch["inp"].shape[-1] * scale]

        pred = pred.view(shape).contiguous()
        pred = pred[..., : batch["inp"].shape[-2] * scale, : batch["inp"].shape[-1] * scale]

        res = metric_fn(pred, batch["gt"])
        val_res.add(res.item(), batch["inp"].shape[0])
        if verbose:
            tqdm.write("val {:.6f}".format(val_res.item()))

    return val_res.item()