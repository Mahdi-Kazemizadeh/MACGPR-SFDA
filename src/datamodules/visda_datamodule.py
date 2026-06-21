import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


class VisDADataModule:
    """Data module for VisDA-C Synthetic-to-Real experiments."""

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.dataset_config = config["dataset"]

        self.root = self._resolve_root()
        self.source_split = self.dataset_config.get("source_split", "train")
        self.target_split = self.dataset_config.get(
            "target_split", "validation")

    def _resolve_root(self) -> Path:
        root_env = self.dataset_config.get("root_env", "VISDA_ROOT")
        default_root = self.dataset_config.get(
            "default_root", "./data/visda-c")
        return Path(os.environ.get(root_env, default_root))

    def build_source_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.IMAGENET_MEAN,
                    std=self.IMAGENET_STD,
                ),
            ]
        )

    def build_target_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.IMAGENET_MEAN,
                    std=self.IMAGENET_STD,
                ),
            ]
        )

    def get_source_dataset(self) -> Dataset:
        source_dir = self.root / self.source_split

        if not source_dir.exists():
            raise FileNotFoundError(f"Source split not found: {source_dir}")

        return datasets.ImageFolder(
            root=str(source_dir),
            transform=self.build_source_transform(),
        )

    def get_target_dataset(self) -> Dataset:
        target_dir = self.root / self.target_split

        if not target_dir.exists():
            raise FileNotFoundError(f"Target split not found: {target_dir}")

        return datasets.ImageFolder(
            root=str(target_dir),
            transform=self.build_target_transform(),
        )

    def build_source_loaders(
        self,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        val_ratio: float,
        seed: int,
    ) -> tuple[DataLoader, DataLoader]:
        dataset = self.get_source_dataset()

        val_size = int(len(dataset) * val_ratio)
        train_size = len(dataset) - val_size

        generator = torch.Generator().manual_seed(seed)

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=generator,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader

    def build_target_loader(
        self,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        shuffle: bool = False,
    ) -> DataLoader:
        dataset = self.get_target_dataset()

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def get_class_names(self) -> list[str]:
        return self.dataset_config["class_names"]

    def get_num_classes(self) -> int:
        return int(self.dataset_config["num_classes"])

    def get_root(self) -> Path:
        return self.root
