import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register, make_model
from utils import make_coord


@register("liif")
class LIIF(nn.Module):
    def __init__(
        self,
        encoder_spec,
        imnet_spec=None,
        local_ensemble=True,
        feat_unfold=True,
        cell_decode=True,
    ):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = make_model(encoder_spec)

        if imnet_spec is not None:
            in_dim = self.encoder.out_dim
            if self.feat_unfold:
                in_dim *= 9
            in_dim += 2
            if self.cell_decode:
                in_dim += 2
            self.imnet = make_model(imnet_spec, args={"in_dim": in_dim})
        else:
            self.imnet = None

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def _grid_sample(self, map, coord):
        return F.grid_sample(
            map, coord.flip(-1).unsqueeze(1), mode="nearest", align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.imnet is None:
            ret = self._grid_sample(feat, coord)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3]
            )

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        B, Q = coord.shape[:2]
        _, _, H, W = feat.shape
        rx, ry = 1 / W, 1 / H

        feat_coord = (
            make_coord(
                feat.shape[-2:],
                flatten=False,
            )
            .to(feat.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(feat.shape[0], 2, *feat.shape[-2:])
        )

        rel_cell = cell.clone()
        rel_cell[:, :, 0].mul_(H)
        rel_cell[:, :, 1].mul_(W)
        preds, areas = [], []

        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0].add_(vy * ry + eps_shift)
                coord_[:, :, 1].add_(vx * rx + eps_shift)
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                q_feat = self._grid_sample(feat, coord_)
                q_coord = self._grid_sample(feat_coord, coord_)

                rel_coord = coord - q_coord
                rel_coord[:, :, 0].mul_(H)
                rel_coord[:, :, 1].mul_(W)

                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    inp = torch.cat([inp, rel_cell], dim=-1)

                pred = self.imnet(inp.contiguous().view(B * Q, -1)).view(B, Q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            areas[0], areas[3] = areas[3], areas[0]
            areas[1], areas[2] = areas[2], areas[1]

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        return ret

    def forward(self, inp, coord, cell) -> torch.Tensor:
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
