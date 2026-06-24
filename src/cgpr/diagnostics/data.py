from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cgpr.diagnostics.config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD


ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class TargetData:
    dataset: datasets.ImageFolder
    loader: DataLoader


class VisDATargetDataModule:
    def __init__(self, visda_root: Path, batch_size: int, num_workers: int):
        self.visda_root = visda_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def target_dir(self) -> Path:
        return self.visda_root / "validation"

    def build_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def build(self) -> TargetData:
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Target directory not found: {self.target_dir}")

        dataset = datasets.ImageFolder(
            str(self.target_dir),
            transform=self.build_transform(),
        )

        if dataset.classes != CLASS_NAMES:
            print("Warning: ImageFolder class order differs from expected class order.")
            print("ImageFolder classes:", dataset.classes)
            print("Expected classes:", CLASS_NAMES)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

        return TargetData(dataset=dataset, loader=loader)
