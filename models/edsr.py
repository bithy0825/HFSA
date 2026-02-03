import torch
import torch.nn as nn

from models import register


class ResBlock(nn.Module):
    def __init__(
        self,
        n_feats: int,
        kernel_size: int = 3,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale: float = 1.0,
    ):
        super().__init__()
        body = []
        for i in range(2):
            body.append(
                nn.Conv2d(
                    n_feats, n_feats, kernel_size, padding=kernel_size // 2, bias=bias
                )
            )
            if bn:
                body.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                body.append(act)

        self.body = nn.Sequential(*body)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


url = {
    "r16f64x2": "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt",
    "r16f64x3": "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt",
    "r16f64x4": "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt",
    "r32f256x2": "https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt",
    "r32f256x3": "https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt",
    "r32f256x4": "https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt",
}


@register("edsr-baseline")
class EDSR(nn.Module):
    def __init__(
        self, in_dim=3, n_feats=64, n_blocks=16, res_scale=1.0, kernel_size=3, scale=2
    ):
        super().__init__()
        act = nn.ReLU(True)
        url_name = "r{}f{}x{}".format(n_blocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None

        self.head = nn.Conv2d(in_dim, n_feats, kernel_size, padding=kernel_size // 2)
        body = []
        body.extend(
            ResBlock(
                n_feats=n_feats,
                kernel_size=kernel_size,
                bias=True,
                bn=False,
                act=act,
                res_scale=res_scale,
            )
            for _ in range(n_blocks)
        )
        body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2))

        self.body = nn.Sequential(*body)

        self.out_dim = n_feats

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x
        return res
