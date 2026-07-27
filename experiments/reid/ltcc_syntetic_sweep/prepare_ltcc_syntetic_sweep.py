#!/usr/bin/env python3
"""Prepare LTCC + synthetic percentage-sweep datasets and TAO configs."""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import yaml


PROJECT_ROOT = Path("/mnt/2tb_ssd/TwinProject")
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments/reid/ltcc_syntetic_sweep"
LTCC_ROOT = PROJECT_ROOT / "experiments/reid/ltcc/data"
SYNTHETIC_ROOT = PROJECT_ROOT / "datasets/final_syntetic_market1501"
BASE_CONFIG = PROJECT_ROOT / "experiments/reid/ltcc/ltcc_swin_plain.yaml"

PERCENTAGES = (10, 25, 50, 75, 100)
PID_RE = re.compile(r"^(?P<pid>\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild generated data folders.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--val-batch-size", type=int, default=32, help="Validation batch size.")
    parser.add_argument("--num-workers", type=int, default=8, help="Dataset worker count.")
    parser.add_argument(
        "--synthetic-offset",
        type=int,
        default=None,
        help="PID offset for synthetic images. Default is max LTCC PID + 1000.",
    )
    return parser.parse_args()


def image_paths(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})


def pid_from_name(path: Path) -> int:
    match = PID_RE.match(path.name)
    if not match:
        raise ValueError(f"Cannot parse Market-1501 PID from filename: {path}")
    return int(match.group("pid"))


def grouped_by_pid(paths: Iterable[Path]) -> dict[int, list[Path]]:
    grouped: dict[int, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[pid_from_name(path)].append(path)
    return dict(sorted(grouped.items()))


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def relabeled_synthetic_name(src: Path, offset: int) -> str:
    old_pid = pid_from_name(src)
    new_pid = old_pid + offset
    return PID_RE.sub(f"{new_pid:04d}_", src.name, count=1)


def selected_synthetic_images(synthetic_by_pid: dict[int, list[Path]], percentage: int) -> list[Path]:
    selected: list[Path] = []
    for paths in synthetic_by_pid.values():
        count = len(paths) if percentage == 100 else max(1, math.ceil(len(paths) * percentage / 100.0))
        selected.extend(paths[:count])
    return selected


def build_train_dir(
    train_dir: Path,
    ltcc_images: list[Path],
    synthetic_images: list[Path],
    synthetic_offset: int,
    include_ltcc: bool,
) -> None:
    train_dir.mkdir(parents=True, exist_ok=True)

    if include_ltcc:
        for src in ltcc_images:
            link_or_copy(src, train_dir / src.name)

    for src in synthetic_images:
        link_or_copy(src, train_dir / relabeled_synthetic_name(src, synthetic_offset))


def load_base_config() -> dict:
    with BASE_CONFIG.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_config(
    config_path: Path,
    experiment_name: str,
    train_dir: Path,
    num_classes: int,
    args: argparse.Namespace,
) -> None:
    cfg = load_base_config()
    results_dir = EXPERIMENT_ROOT / "results" / experiment_name

    cfg["results_dir"] = str(results_dir)
    cfg["dataset"]["train_dataset_dir"] = str(train_dir)
    cfg["dataset"]["test_dataset_dir"] = str(LTCC_ROOT / "bounding_box_test")
    cfg["dataset"]["query_dataset_dir"] = str(LTCC_ROOT / "query")
    cfg["dataset"]["num_classes"] = num_classes
    cfg["dataset"]["batch_size"] = args.batch_size
    cfg["dataset"]["val_batch_size"] = args.val_batch_size
    cfg["dataset"]["num_workers"] = args.num_workers

    cfg["train"]["num_epochs"] = args.epochs
    cfg["train"]["checkpoint_interval"] = 5
    cfg["train"]["validation_interval"] = 1
    cfg["train"]["resume_training_checkpoint_path"] = None
    cfg["train"]["results_dir"] = str(results_dir / "train")
    cfg["train"]["gpu_ids"] = [0]
    cfg["train"]["num_gpus"] = 1
    cfg["train"]["optim"]["lr_steps"] = [90, 130] if args.epochs >= 150 else [30, 50]

    cfg["evaluate"]["gpu_ids"] = [0]
    cfg["evaluate"]["num_gpus"] = 1
    cfg["evaluate"]["checkpoint"] = "???"
    cfg["evaluate"]["results_dir"] = str(EXPERIMENT_ROOT / "evaluate" / experiment_name)
    cfg["evaluate"]["output_sampled_matches_plot"] = str(
        EXPERIMENT_ROOT / "evaluate" / experiment_name / "sampled_matches.png"
    )
    cfg["evaluate"]["output_cmc_curve_plot"] = str(
        EXPERIMENT_ROOT / "evaluate" / experiment_name / "cmc_curve.png"
    )
    cfg["evaluate"]["test_dataset"] = None
    cfg["evaluate"]["query_dataset"] = None

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def main() -> None:
    args = parse_args()
    ltcc_train = LTCC_ROOT / "bounding_box_train"
    synthetic_train = SYNTHETIC_ROOT / "bounding_box_train"

    if not ltcc_train.is_dir():
        raise SystemExit(f"Missing LTCC train directory: {ltcc_train}")
    if not synthetic_train.is_dir():
        raise SystemExit(f"Missing synthetic train directory: {synthetic_train}")
    if not BASE_CONFIG.is_file():
        raise SystemExit(f"Missing base config: {BASE_CONFIG}")

    generated_data = EXPERIMENT_ROOT / "data"
    if args.rebuild and generated_data.exists():
        shutil.rmtree(generated_data)

    ltcc_images = image_paths(ltcc_train)
    synthetic_images = image_paths(synthetic_train)
    ltcc_pids = sorted({pid_from_name(path) for path in ltcc_images})
    synthetic_by_pid = grouped_by_pid(synthetic_images)
    synthetic_offset = args.synthetic_offset if args.synthetic_offset is not None else max(ltcc_pids) + 1000

    print(f"LTCC images: {len(ltcc_images)}")
    print(f"LTCC unique IDs: {len(ltcc_pids)}")
    print(f"Synthetic images: {len(synthetic_images)}")
    print(f"Synthetic unique IDs: {len(synthetic_by_pid)}")
    print(f"Synthetic PID offset: {synthetic_offset}")

    manifest_rows = []

    for percentage in PERCENTAGES:
        experiment_name = f"ltcc_syntetic_{percentage}"
        train_dir = generated_data / experiment_name / "bounding_box_train"
        selected = selected_synthetic_images(synthetic_by_pid, percentage)
        selected_pids = {pid_from_name(path) for path in selected}
        build_train_dir(train_dir, ltcc_images, selected, synthetic_offset, include_ltcc=True)
        write_config(
            EXPERIMENT_ROOT / "configs" / f"{experiment_name}.yaml",
            experiment_name,
            train_dir,
            len(ltcc_pids) + len(selected_pids),
            args,
        )
        manifest_rows.append((experiment_name, percentage, len(ltcc_images), len(selected), len(ltcc_pids), len(selected_pids)))

    synthetic_only_name = "syntetic_only_100"
    synthetic_only_dir = generated_data / synthetic_only_name / "bounding_box_train"
    selected_all = selected_synthetic_images(synthetic_by_pid, 100)
    build_train_dir(synthetic_only_dir, [], selected_all, synthetic_offset, include_ltcc=False)
    write_config(
        EXPERIMENT_ROOT / "configs" / f"{synthetic_only_name}.yaml",
        synthetic_only_name,
        synthetic_only_dir,
        len({pid_from_name(path) for path in selected_all}),
        args,
    )
    manifest_rows.append((synthetic_only_name, 100, 0, len(selected_all), 0, len(synthetic_by_pid)))

    manifest_path = EXPERIMENT_ROOT / "sweep_manifest.tsv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write("experiment\tpercent\tltcc_images\tsynthetic_images\tltcc_ids\tsynthetic_ids\n")
        for row in manifest_rows:
            handle.write("\t".join(str(item) for item in row) + "\n")

    print(f"Wrote configs: {EXPERIMENT_ROOT / 'configs'}")
    print(f"Wrote data: {generated_data}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
