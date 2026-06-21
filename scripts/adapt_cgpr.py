import argparse
from pathlib import Path

import torch

from src.adaptation.cgpr_adapter import CGPRAdapter
from src.datamodules.visda_datamodule import VisDADataModule
from src.models.resnet_classifier import ResNetClassifier
from src.utils.config import ConfigLoader
from src.utils.seed import SeedManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CGPR adaptation on VisDA-C target domain."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/visda_cgpr.yaml",
        help="Path to the YAML configuration file.",
    )

    return parser.parse_args()


def load_source_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state" not in checkpoint:
        raise KeyError("Checkpoint must contain 'model_state'.")

    model.load_state_dict(checkpoint["model_state"], strict=True)

    return checkpoint


def main() -> None:
    args = parse_args()

    config = ConfigLoader.load(args.config)
    seed = int(config["seed"])
    SeedManager.set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("CGPR Adaptation | VisDA-C Synthetic to Real")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Device: {device}")

    data_module = VisDADataModule(config)
    target_dataset = data_module.get_target_dataset()

    model = ResNetClassifier(
        num_classes=config["model"]["num_classes"],
        backbone=config["model"]["backbone"],
        pretrained=False,
    )

    checkpoint = load_source_checkpoint(
        model=model,
        checkpoint_path=config["model"]["checkpoint_path"],
        device=device,
    )

    print(f"Loaded checkpoint: {config['model']['checkpoint_path']}")

    if "best_val_acc" in checkpoint:
        print(f"Source validation accuracy: {checkpoint['best_val_acc']:.4f}")

    adapter = CGPRAdapter(
        model=model,
        target_dataset=target_dataset,
        config=config,
        device=device,
    )

    metrics = adapter.adapt()

    print("=" * 70)
    print("CGPR Final Results")
    print("=" * 70)
    print(
        f"Initial source-only accuracy: {metrics['initial_source_only_accuracy']:.4f}")
    print(
        f"Best debug accuracy         : {metrics['best_debug_accuracy_during_adaptation']:.4f}")
    print(
        f"Best unsupervised score     : {metrics['best_unsupervised_selection_score']:.4f}")
    print(f"Final accuracy              : {metrics['accuracy']:.4f}")
    print(
        f"Mean class accuracy         : {metrics['mean_class_accuracy']:.4f}")
    print(f"Macro-F1                    : {metrics['macro_f1']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
