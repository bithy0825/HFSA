import re
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register


@register("image-folder")
class ImageFolder(Dataset):
    def __init__(
        self,
        root_path,
        split_rule=r"\d+\.(png)",
        first_k=None,
        repeat=1,
        cache=None,
    ):
        self.repeat = repeat
        self.cache = cache

        if cache not in [None, "in_memory"]:
            raise ValueError(
                f"Invalid cache option: {cache}. Must be None or 'in_memory'"
            )

        if split_rule is not None:
            self.split_rule = re.compile(split_rule)
            all_files = sorted(os.listdir(root_path))
            filenames = [f for f in all_files if self.split_rule.search(f)]
        else:
            filenames = sorted(os.listdir(root_path))

        if len(filenames) == 0:
            raise ValueError(f"No matched files found in {root_path}")

        if first_k is not None and len(filenames) > first_k:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == "in_memory":
                self.files.append(
                    transforms.ToTensor()(Image.open(file).convert("RGB"))
                )
            else:
                self.files.append(file)

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        file_idx = idx % len(self.files)
        x = self.files[file_idx]

        if self.cache is None:
            return transforms.ToTensor()(Image.open(x).convert("RGB"))

        elif self.cache == "in_memory":
            return x


@register("paired-image-folders")
class PairedImageFolders(Dataset):
    def __init__(self, root_path1, root_path2, **kwargs):
        self.dataset1 = ImageFolder(root_path1, **kwargs)
        self.dataset2 = ImageFolder(root_path2, **kwargs)

        if len(self.dataset1) != len(self.dataset2):
            raise ValueError(
                f"Dataset 1 and Dataset 2 have different lengths: {len(self.dataset1)} vs {len(self.dataset2)}"
            )

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        return self.dataset1[idx], self.dataset2[idx]
