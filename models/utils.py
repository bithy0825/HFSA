import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier_encoding(ratio, L):
    # ratio: B, Q, 2
    # L: int
    # return: B, Q, 4
    freq = 2**L
    ratio = ratio * freq * torch.pi
    return torch.cat([torch.sin(ratio), torch.cos(ratio)], dim=-1)  # B, Q, 4


class LinearAtten(nn.Module):
    def __init__(self, in_dim, n_heads, qkv_bias=True, act=nn.GELU):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = in_dim // n_heads

        self.temperature = nn.Parameter(torch.ones(n_heads, 1, 1))
        self.bias = nn.Parameter(torch.zeros(n_heads, self.head_dim, self.head_dim))

        self.qkv = nn.Linear(in_dim, 3 * in_dim, bias=qkv_bias)
        self.knorm = nn.LayerNorm(self.head_dim)
        self.vnorm = nn.LayerNorm(self.head_dim)

        self.proj1 = nn.Linear(in_dim, in_dim)
        self.proj2 = nn.Linear(in_dim, in_dim)

        self.act = act()

    def forward(self, x):
        B, Q, C = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(B, Q, 3, self.n_heads, self.head_dim)
            .permute(0, 2, 3, 1, 4)
            .unbind(1)
        )  # (B, H, Q, C)
        k, v = self.knorm(k), self.vnorm(v)

        atten = (k.transpose(-2, -1) @ v) / Q * self.temperature + self.bias
        atten = (q @ atten).permute(0, 2, 1, 3).reshape(B, Q, C)
        atten = atten + x

        return self.proj2(self.act(self.proj1(atten))) + x


class MLP_with_shortcut(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, code_dim=4, act=nn.GELU, drop=0.0):
        super().__init__()
        self.code_dim = code_dim
        self.norm = nn.LayerNorm(in_dim - code_dim)
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.act = act()
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, code):
        identity = x
        x = self.norm(x)
        x = torch.cat([x, code], dim=-1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        if x.shape[-1] == identity.shape[-1]:
            x = x + identity
        return x
