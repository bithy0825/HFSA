import torch
import torch.nn as nn

from models import register


class RDBConv(nn.Module):
    def __init__(self, in_dim, grow_rate, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            *[
                nn.Conv2d(
                    in_dim,
                    grow_rate,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                    stride=1,
                ),
                nn.ReLU(inplace=True),
            ]
        )

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, grow_rate0, grow_rate, n_layers, kernel_size=3):
        super().__init__()
        G0 = grow_rate0
        G = grow_rate
        C = n_layers

        convs = []
        for c in range(C):
            convs.append(RDBConv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        self.lff = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.lff(self.convs(x)) + x


@register("rdn")
class RDN(nn.Module):
    def __init__(self, in_dim=3, G0=64, kernel_size=3, cfg="B"):
        super().__init__()
        cfg_dict = {"A": (20, 6, 32), "B": (16, 8, 64)}
        self.D, C, G = cfg_dict[cfg]

        self.sfe1 = nn.Conv2d(
            in_dim, G0, kernel_size, padding=(kernel_size - 1) // 2, stride=1
        )
        self.sfe2 = nn.Conv2d(
            G0, G0, kernel_size, padding=(kernel_size - 1) // 2, stride=1
        )

        self.rdbs = nn.ModuleList([RDB(G0, G, C, kernel_size) for _ in range(self.D)])

        self.gff = nn.Sequential(
            *[
                nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
                nn.Conv2d(
                    G0, G0, kernel_size, padding=(kernel_size - 1) // 2, stride=1
                ),
            ]
        )

        self.out_dim = G0

    def forward(self, x):
        sfe1 = self.sfe1(x)
        x = self.sfe2(sfe1)

        rdb_outputs = [None] * self.D
        for i, rdb in enumerate(self.rdbs):
            x = rdb(x)
            rdb_outputs[i] = x

        x = self.gff(torch.cat(rdb_outputs, 1))
        x += sfe1
        return x
