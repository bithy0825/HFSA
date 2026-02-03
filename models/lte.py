import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import register, make_model
from utils import make_coord


@register("lte")
class LTE(nn.Module):
    def __init__(self, encoder_spec, imnet_spec=None, hidden_dim=256):
        super().__init__()
        self.encoder = make_model(encoder_spec)
        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.phase = nn.Linear(2, hidden_dim // 2, bias=False)

        self.imnet = make_model(imnet_spec, args={"in_dim": hidden_dim})

    def gen_feat(self, inp):
        self.inp = inp
        self.feat_coord = (
            make_coord(inp.shape[-2:], flatten=False)
            .to(inp.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(inp.shape[0], 2, *inp.shape[-2:])
        )

        self.feat = self.encoder(inp)
        self.coeff = self.coef(self.feat)
        self.freqq = self.freq(self.feat)
        return self.feat

    def _grid_sample(self, map, coord):
        return F.grid_sample(
            map, coord.flip(-1).unsqueeze(1), mode="nearest", align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)

    def query_rgb(self, coord, cell):
        feat = self.feat
        coef = self.coeff
        freq = self.freqq
        feat_coord = self.feat_coord

        B, Q = coord.shape[:2]
        _, _, H, W = feat.shape
        rx, ry = 1 / W, 1 / H
        eps_shift = 1e-6

        rel_cell = cell.clone()
        rel_cell[:, :, 0].mul_(H)
        rel_cell[:, :, 1].mul_(W)
        preds, areas = [], []

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()
                coord_[:, :, 0].add_(vy * ry + eps_shift)
                coord_[:, :, 1].add_(vx * rx + eps_shift)
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                q_coef = self._grid_sample(coef, coord_)
                q_freq = self._grid_sample(freq, coord_)
                q_coord = self._grid_sample(feat_coord, coord_)

                rel_coord = coord - q_coord
                rel_coord[:, :, 0].mul_(H)
                rel_coord[:, :, 1].mul_(W)

                q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
                q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))
                q_freq = torch.sum(q_freq, dim=-2)
                q_freq += self.phase(rel_cell.view((B * Q, -1))).view(B, Q, -1)
                q_freq = torch.cat(
                    (torch.cos(np.pi * q_freq), torch.sin(np.pi * q_freq)), dim=-1
                )

                inp = torch.mul(q_coef, q_freq)

                pred = self.imnet(inp.contiguous().view(B * Q, -1)).view(B, Q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        areas[0], areas[3] = areas[3], areas[0]
        areas[1], areas[2] = areas[2], areas[1]

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        ret += F.grid_sample(
            self.inp,
            coord.flip(-1).unsqueeze(1),
            padding_mode="border",
            mode="bilinear",
            align_corners=False,
        )[:, :, 0, :].permute(0, 2, 1)

        return ret

    def forward(self, inp, coord, cell) -> torch.Tensor:
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
