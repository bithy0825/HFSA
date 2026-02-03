import torch
import torch.nn as nn
import torch.nn.functional as F


class MSSSIML1(nn.Module):
    def __init__(
        self,
        data_range=1.0,
        gaussian_sigmas=[0.5, 1.0, 1.5, 2.0, 2.5],
        K=(0.01, 0.03),
        alpha=0.05,
        compensation=100.0,
    ):
        super().__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation = compensation
        fliter_size = int(4 * gaussian_sigmas[-1] + 1)

        g_masks = torch.zeros(3 * len(gaussian_sigmas), 1, fliter_size, fliter_size)
        for idx, sigma in enumerate(gaussian_sigmas):
            g_masks[3 * idx + 0, 0, :, :] = self._fspecial_gauss_2d(fliter_size, sigma)
            g_masks[3 * idx + 1, 0, :, :] = self._fspecial_gauss_2d(fliter_size, sigma)
            g_masks[3 * idx + 2, 0, :, :] = self._fspecial_gauss_2d(fliter_size, sigma)
        self.register_buffer("g_masks", g_masks)

    def _fspecial_gauss_1d(self, size, sigma):
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(vec, vec)

    def forward(self, x, y):
        B, C, H, W = x.shape
        mux = F.conv2d(x, self.g_masks, groups=3, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=3, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=3, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=3, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=3, padding=self.pad) - muxy

        l = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
        PIcs = cs.prod(dim=1)

        loss_msssim = 1 - lM * PIcs  # B, H, W
        loss_l1 = F.l1_loss(x, y, reduction="none")  # B, 3, H, W

        gaussian_l1 = F.conv2d(
            loss_l1,
            self.g_masks.narrow(dim=0, start=-3, length=3),
            groups=3,
            padding=self.pad,
        ).mean(
            1
        )  # B, H, W

        loss_mix = self.alpha * loss_msssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation * loss_mix
        return loss_mix.mean(), loss_msssim.mean(), (gaussian_l1 / self.DR).mean()
