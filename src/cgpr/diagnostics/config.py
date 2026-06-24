from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


CLASS_NAMES = [
    "aeroplane", "bicycle", "bus", "car", "horse", "knife",
    "motorcycle", "person", "plant", "skateboard", "train", "truck",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class CapacityAuditConfig:
    visda_root: Path
    checkpoint: Path
    output_dir: Path

    batch_size: int
    num_workers: int
    seed: int

    capacity_per_class: int
    reserve_per_class: int
    prototype_seed_per_class: int

    audit_min_confidence: float
    audit_min_margin: float
    audit_max_proto_dist: float
    audit_pseudo_conflict_confidence: float

    replace_min_confidence: float
    replace_min_margin: float
    replace_max_proto_dist: float
    replace_pseudo_conflict_confidence: float
    replacement_tiers: int

    debug_labels: bool

    @property
    def num_classes(self) -> int:
        return len(CLASS_NAMES)

    @staticmethod
    def build_arg_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Capacity-aware audit diagnostic for VisDA-C SFDA."
        )

        parser.add_argument(
            "--visda_root",
            type=str,
            default=os.environ.get("VISDA_ROOT", "./data/visda-c"),
        )
        parser.add_argument(
            "--checkpoint",
            type=str,
            default="checkpoints/source_model.pth",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="outputs/capacity_audit",
        )

        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--seed", type=int, default=42)

        parser.add_argument("--capacity_per_class", type=int, default=300)
        parser.add_argument("--reserve_per_class", type=int, default=900)
        parser.add_argument("--prototype_seed_per_class", type=int, default=500)

        parser.add_argument("--audit_min_confidence", type=float, default=0.70)
        parser.add_argument("--audit_min_margin", type=float, default=0.015)
        parser.add_argument("--audit_max_proto_dist", type=float, default=0.55)
        parser.add_argument(
            "--audit_pseudo_conflict_confidence",
            type=float,
            default=0.90,
        )

        parser.add_argument("--replace_min_confidence", type=float, default=0.60)
        parser.add_argument("--replace_min_margin", type=float, default=0.010)
        parser.add_argument("--replace_max_proto_dist", type=float, default=0.60)
        parser.add_argument(
            "--replace_pseudo_conflict_confidence",
            type=float,
            default=0.92,
        )
        parser.add_argument("--replacement_tiers", type=int, default=3)

        parser.add_argument("--debug_labels", action="store_true")

        return parser

    @classmethod
    def from_args(cls) -> "CapacityAuditConfig":
        parser = cls.build_arg_parser()
        args = parser.parse_args()

        return cls(
            visda_root=Path(args.visda_root),
            checkpoint=Path(args.checkpoint),
            output_dir=Path(args.output_dir),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            capacity_per_class=args.capacity_per_class,
            reserve_per_class=args.reserve_per_class,
            prototype_seed_per_class=args.prototype_seed_per_class,
            audit_min_confidence=args.audit_min_confidence,
            audit_min_margin=args.audit_min_margin,
            audit_max_proto_dist=args.audit_max_proto_dist,
            audit_pseudo_conflict_confidence=args.audit_pseudo_conflict_confidence,
            replace_min_confidence=args.replace_min_confidence,
            replace_min_margin=args.replace_min_margin,
            replace_max_proto_dist=args.replace_max_proto_dist,
            replace_pseudo_conflict_confidence=args.replace_pseudo_conflict_confidence,
            replacement_tiers=args.replacement_tiers,
            debug_labels=args.debug_labels,
        )
