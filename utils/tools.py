import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def make_coord(shape, ranges=None, flatten=True):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]

        r = (v1 - v0) * 0.5 / n
        seq = v0 + r + 2 * r * torch.arange(n, dtype=torch.float32)
        coord_seqs.append(seq)

    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing="ij"), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])  # [B, H*W, 2]

    return ret


def to_pixel_samples(img):
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb


def pts_to_image(inp, gt, coord, pred, h_pad=0, w_pad=0):
    '''ih, iw = inp.shape[-2:]

    s = math.sqrt(gt.shape[1] / (ih * iw))
    gt_shape = [inp.shape[0], round(ih * s), round(iw * s), 3]
    gt = gt.view(*gt_shape).permute(0, 3, 1, 2).contiguous()

    ih_padded = ih + h_pad
    iw_padded = iw + w_pad
    s = math.sqrt(coord.shape[1] / (ih_padded * iw_padded))
    pred_shape = [inp.shape[0], round(ih_padded * s), round(iw_padded * s), 3]
    pred = pred.view(*pred_shape).permute(0, 3, 1, 2).contiguous()

    pred = pred[..., : gt.shape[-2], : gt.shape[-1]]'''
    B, C, H, W = gt.shape
    pred = pred[..., :H, :W]


    return pred, gt


@torch.jit.script
def grid_sample(
    map: torch.Tensor, coord: torch.Tensor, mode: str = "nearest"
) -> torch.Tensor:
    return F.grid_sample(
        map, coord.flip(-1).unsqueeze(1), mode=mode, align_corners=False
    )[:, :, 0, :].permute(0, 2, 1)
