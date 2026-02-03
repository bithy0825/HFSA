import random
import math
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from datasets import register
from utils import to_pixel_samples, make_coord


@register("sr-implicit-paired")
class SRImplicitPaired(Dataset):
    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def _apply_augment(self, crop_lr, crop_hr):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        dflip = random.random() < 0.5

        def augment(x):
            if hflip:
                x = x.flip(-2)
            if vflip:
                x = x.flip(-1)
            if dflip:
                x = x.transpose(-2, -1)
            return x

        return augment(crop_lr), augment(crop_hr)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        scale = img_hr.shape[-2] // img_lr.shape[-2]
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, : h_lr * scale, : w_lr * scale]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0 : x0 + w_lr, y0 : y0 + w_lr]
            w_hr = w_lr * scale
            x1 = x0 * scale
            y1 = y0 * scale
            crop_hr = img_hr[:, x1 : x1 + w_hr, y1 : y1 + w_hr]

        if self.augment:
            crop_lr, crop_hr = self._apply_augment(crop_lr, crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0].mul_(2 / crop_hr.shape[-2])
        cell[:, 1].mul_(2 / crop_hr.shape[-1])

        return {"inp": crop_lr, "coord": hr_coord, "cell": cell, "gt": hr_rgb}


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(transforms.ToPILImage()(img))
    )


@register("sr-implicit-downsampled")
class SRImplicitDownsampled(Dataset):
    def __init__(
        self,
        dataset,
        inp_size=None,
        scale_min=1,
        scale_max=None,
        augment=False,
        sample_q=None,
    ):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            self.scale_max = scale_min
        else:
            self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def _apply_augment(self, crop_lr, crop_hr):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        dflip = random.random() < 0.5

        def augment(x):
            if hflip:
                x = x.flip(-2)
            if vflip:
                x = x.flip(-1)
            if dflip:
                x = x.transpose(-2, -1)
            return x

        return augment(crop_lr), augment(crop_hr)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        scale = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / scale + 1e-9)
            w_lr = math.floor(img.shape[-1] / scale + 1e-9)
            img = img[:, : round(h_lr * scale), : round(w_lr * scale)]
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * scale)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0 : x0 + w_hr, y0 : y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            crop_lr, crop_hr = self._apply_augment(crop_lr, crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0].mul_(2 / crop_hr.shape[-2])
        cell[:, 1].mul_(2 / crop_hr.shape[-1])

        return {"inp": crop_lr, "coord": hr_coord, "cell": cell, "gt": hr_rgb}


@register("sr-implicit-paired-fast")
class SRImplicitPairedFast(Dataset):
    # only for fast inference
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        scale = img_hr.shape[-2] // img_lr.shape[-2]

        h_lr, w_lr = img_lr.shape[-2:]
        h_hr = round(h_lr * scale)
        w_hr = round(w_lr * scale)
        img_lr = img_lr[:, :h_lr, :w_lr]
        crop_lr, crop_hr = img_lr, img_hr

        hr_coord = make_coord([h_hr, w_hr], flatten=False)  # H, W, 2
        hr_rgb = crop_hr  # C, H, W

        cell = torch.ones_like(hr_coord)
        cell[..., 0].mul_(2 / crop_hr.shape[-2])
        cell[..., 1].mul_(2 / crop_hr.shape[-1])

        return {"inp": crop_lr, "coord": hr_coord, "cell": cell, "gt": hr_rgb}


@register("sr-implicit-downsampled-fast")
class SRImplicitDownsampledFast(Dataset):
    # only for fast inference
    def __init__(
        self,
        dataset,
        scale_min=1,
        scale_max=None,
    ):
        self.dataset = dataset
        self.scale_min = scale_min
        if scale_max is None:
            self.scale_max = scale_min
        else:
            self.scale_max = scale_max

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        scale = random.uniform(self.scale_min, self.scale_max)

        h_lr = math.floor(img.shape[-2] / scale + 1e-9)
        w_lr = math.floor(img.shape[-1] / scale + 1e-9)
        h_hr = round(h_lr * scale)
        w_hr = round(w_lr * scale)
        img = img[:, :h_hr, :w_hr]
        img_down = resize_fn(img, (h_lr, w_lr))
        crop_lr, crop_hr = img_down, img

        hr_coord = make_coord([h_hr, w_hr], flatten=False)  # H, W, 2
        hr_rgb = crop_hr  # H, W, C

        cell = torch.ones_like(hr_coord)  # H, W, 2
        cell[..., 0].mul_(2 / crop_hr.shape[-2])
        cell[..., 1].mul_(2 / crop_hr.shape[-1])

        return {"inp": crop_lr, "coord": hr_coord, "cell": cell, "gt": hr_rgb}
