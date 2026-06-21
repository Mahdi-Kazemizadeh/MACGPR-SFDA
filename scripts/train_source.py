import argparse
from pathlib import Path

import torch

from src.datamodules.visda_datamodule import VisDADataModule
from src.models.resnet_classifier import ResNetClassifier
from src.training.source_trainer import SourceTrainer
from src.utils.config import ConfigLoader
from src.utils.seed import SeedManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train source model on VisDA-C Synthetic domain."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/visda_source.yaml",
        help="Path to the YAML configuration file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ConfigLoader.load(args.config)
    seed = int(config["seed"])
    SeedManager.set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Source Training | VisDA-C Synthetic")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Device: {device}")

    data_module = VisDADataModule(config)

    train_loader, val_loader = data_module.build_source_loaders(
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
        val_ratio=config["training"]["val_ratio"],
        seed=seed,
    )

    model = ResNetClassifier(
        num_classes=config["model"]["num_classes"],
        backbone=config["model"]["backbone"],
        pretrained=config["model"]["pretrained"],
    )

    trainer = SourceTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    result = trainer.train()

    print("=" * 70)
    print("Source Training Finished")
    print("=" * 70)
    print(f"Best validation accuracy: {result['best_val_accuracy']:.4f}")
    print(f"Checkpoint: {result['checkpoint_path']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
