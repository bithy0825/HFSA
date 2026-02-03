import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import LinearAtten, fourier_encoding, MLP_with_shortcut
from models import register, make_model
from utils import make_coord, grid_sample


class FlowPre(nn.Module):
    def __init__(self, in_dim, hidden_dim, act=nn.GELU):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, 2),
            nn.Tanh(),
        )
        self.c = nn.Sequential(
            nn.Linear(in_dim + 4, in_dim),
            act(),
            nn.Linear(in_dim, in_dim),
            act(),
            nn.Linear(in_dim, 2),
            nn.Sigmoid(),
        )

    def forward(self, feat, ffeat, coord, feat_coord, rel_cell, H, W):
        C_f, C_ff = feat.shape[1], ffeat.shape[1]
        q_ = grid_sample(torch.cat([feat, ffeat], dim=1), coord, mode="bilinear")
        feat_, ffeat_ = torch.split(q_, [C_f, C_ff], dim=-1)

        deltas = torch.tensor([[-1 / H, -1 / W], [1 / H, 1 / W]], device=feat.device)
        deltas = deltas.unsqueeze(0).unsqueeze(0)
        coord_ = (coord.unsqueeze(-2) + deltas + 1e-6).clamp(-1 + 1e-6, 1 - 1e-6)
        q_coord = F.grid_sample(
            feat_coord, coord_.flip(-1), mode="nearest", align_corners=False
        )
        rel_coord = coord.unsqueeze(-1).permute(0, 2, 1, 3) - q_coord
        rel_coord[:, 0, ...] *= H
        rel_coord[:, 1, ...] *= W
        rel_coord = rel_coord.sum(dim=-1).transpose(-2, -1) / 2  # B, Q, 2

        flow = self.f(torch.cat([ffeat_, rel_cell, rel_coord], dim=-1).contiguous())
        c = self.c(torch.cat([feat_, rel_cell, rel_coord], dim=-1).contiguous())

        flow = torch.stack([flow[..., 0] / H, flow[..., 1] / W], dim=-1)
        return flow, c


@register("hfsa")
class Hfsa(nn.Module):
    def __init__(self, encoder_spec, n_heads=16, hid_dim=256, n_layers=4):
        super().__init__()
        self.encoder = make_model(encoder_spec)
        in_dim = self.encoder.out_dim
        self.expan = nn.Conv2d(in_dim, hid_dim, 3, padding=1)

        self.flow = FlowPre(in_dim, hid_dim)

        self.n_layers = n_layers

        self.fusion = MLP_with_shortcut(hid_dim * 4 + 2 + 4, hid_dim, hid_dim)

        self.atten1 = LinearAtten(hid_dim, n_heads)
        self.atten2 = LinearAtten(hid_dim, n_heads)

        self.imnet = nn.ModuleList(
            [
                MLP_with_shortcut(
                    hid_dim + 4,
                    3 if i == self.n_layers - 1 else hid_dim,
                    hid_dim,
                )
                for i in range(n_layers)
            ]
        )

    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        self.ffeat = self.expan(self.feat)
        self.feat_coord = (
            make_coord(inp.shape[-2:], flatten=False)
            .to(inp.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(inp.shape[0], 2, *inp.shape[-2:])
        )  # 16, 2, 48, 48

    def query_rgb(self, coord, cell, confidence_fixed=None, offset_scale=1.0):
        feat, ffeat, feat_coord = self.feat, self.ffeat, self.feat_coord
        B, _, H, W = feat.shape
        Q = coord.shape[1]
        rx, ry = 1 / W, 1 / H
        rel_cell = torch.stack([cell[..., 0] * H, cell[..., 1] * W], dim=-1)

        constraint = F.grid_sample(
            self.inp,
            coord.flip(-1).unsqueeze(1),
            padding_mode="border",
            mode="bilinear",
            align_corners=False,
        )[:, :, 0, :].permute(0, 2, 1)

        flow, c = self.flow(feat, ffeat, coord, feat_coord, rel_cell, H, W)
        if confidence_fixed is not None:
            c = torch.full_like(c, confidence_fixed)
        coord = coord + flow * c * offset_scale

        deltas = (
            torch.tensor(
                [[-ry, -rx], [ry, -rx], [ry, rx], [-ry, rx]], device=feat.device
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )  # 1, 1, 4, 2 -> B, Q, 4, 2
        coord_ = (coord.unsqueeze(-2) + deltas + 1e-6).clamp(-1 + 1e-6, 1 - 1e-6)

        C_ffeat, C_coord = ffeat.shape[1], feat_coord.shape[1]
        q_combined = F.grid_sample(
            torch.cat([ffeat, feat_coord], dim=1),
            coord_.flip(-1),
            mode="nearest",
            align_corners=False,
        )  # B, C_ffeat + C_coord, Q, 4
        q_ffeat, q_coord = torch.split(q_combined, [C_ffeat, C_coord], dim=1)
        rel_coord = coord.unsqueeze(-1).permute(0, 2, 1, 3) - q_coord
        rel_coord[:, 0, ...] *= H
        rel_coord[:, 1, ...] *= W  # B, 2, Q, 4

        areas = torch.abs(rel_coord[:, 0, ...] * rel_coord[:, 1, ...]) + 1e-9
        areas = areas[..., [3, 2, 1, 0]]  # B, Q, 4
        tot_area = areas.sum(dim=-1, keepdim=True)  # B, Q, 1
        areas = (areas / (tot_area + 1e-9)).unsqueeze(1)  # B, 1, Q, 4
        q_ffeat = (q_ffeat * areas).transpose(-3, -2).reshape(B, Q, -1)  # B, Q, 1024

        ratio_coord = rel_coord.sum(dim=-1).transpose(-2, -1) / 4  # B, Q, 2
        code = fourier_encoding(ratio_coord, 0)  # B, Q, 4
        grid = torch.cat([q_ffeat, rel_cell], dim=-1)  # B, Q, 1024 + 2

        grid = self.atten2(self.atten1(self.fusion(grid, code)))  # B, Q, 256

        for i in range(self.n_layers):
            code = fourier_encoding(ratio_coord, i + 1)
            grid = self.imnet[i](grid, code)

        return grid + constraint

    def forward(self, inp, coord, cell, confidence_fixed=None, offset_scale=1.0):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell, confidence_fixed, offset_scale)
