import argparse
from pathlib import Path

import torch

from src.datamodules.visda_datamodule import VisDADataModule
from src.evaluation.source_evaluator import SourceEvaluator
from src.models.resnet_classifier import ResNetClassifier
from src.utils.config import ConfigLoader
from src.utils.seed import SeedManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate source model on VisDA-C target domain."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/visda_source_eval.yaml",
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
    print("Source-only Evaluation | VisDA-C Real Target")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Device: {device}")

    data_module = VisDADataModule(config)

    target_loader = data_module.build_target_loader(
        batch_size=config["loader"]["batch_size"],
        num_workers=config["loader"]["num_workers"],
        pin_memory=config["loader"]["pin_memory"],
        shuffle=False,
    )

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

    evaluator = SourceEvaluator(
        model=model,
        target_loader=target_loader,
        config=config,
        device=device,
    )

    metrics = evaluator.evaluate()
    metrics["checkpoint_path"] = config["model"]["checkpoint_path"]
    metrics["source_best_val_acc"] = checkpoint.get("best_val_acc", None)
    metrics["class_names"] = config["dataset"]["class_names"]

    metrics_path = evaluator.save_metrics(metrics)

    print("-" * 70)
    print("Source-only target results")
    print("-" * 70)
    print(f"Accuracy           : {metrics['accuracy']:.4f}")
    print(f"Mean Class Accuracy: {metrics['mean_class_accuracy']:.4f}")
    print(f"Macro-F1           : {metrics['macro_f1']:.4f}")
    print(f"Saved metrics to   : {metrics_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
