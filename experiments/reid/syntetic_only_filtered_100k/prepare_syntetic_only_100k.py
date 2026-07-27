#!/usr/bin/env python3
"""Build an exact-size synthetic-only TAO ReID training set."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable

import yaml


ROOT = Path("/mnt/2tb_ssd/TwinProject")
EXP = Path(os.environ.get("EXP_DIR", ROOT / "experiments/reid/syntetic_only_filtered_30k"))
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "syntetic_only_filtered_30k")
DISPLAY_NAME = os.environ.get("DISPLAY_NAME", "Synthetic-Only Filtered 30k")
SYN_ROOT = ROOT / "datasets/final_syntetic_market1501"
SYN_MANIFEST = SYN_ROOT / "manifest.csv"
BASE_CONFIG = ROOT / "experiments/reid/prcc_syntetic_filtered_seq/configs/prcc_plain_swin.yaml"
PRCC_ROOT = ROOT / "experiments/reid/prcc/data"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-images", type=int, default=30_000)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def image_paths(path: Path) -> list[Path]:
    return sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS)


def group_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row["pid"],
        row["camera_id"],
        row["sequence_id"],
        row["frame_id"],
        row["source_box_index"],
    )


def row_sort_key(row: dict[str, str]) -> tuple[int, int, str]:
    return int(row["variant_id"]), int(row["encoded_frame"]), row["output_file"]


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def choose_exact_target(rows: Iterable[dict[str, str]], target: int) -> tuple[list[dict[str, str]], int, int]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    for group_rows in grouped.values():
        group_rows.sort(key=row_sort_key)

    max_available = sum(len(group_rows) for group_rows in grouped.values())
    if target <= 0 or target > max_available:
        raise ValueError(f"Target must be between 1 and {max_available}, got {target}")

    base_cap = 0
    while sum(min(base_cap + 1, len(group_rows)) for group_rows in grouped.values()) <= target:
        base_cap += 1

    selected = []
    candidates_by_pid: dict[str, deque[dict[str, str]]] = defaultdict(deque)
    for key in sorted(grouped, key=lambda item: tuple(int(value) for value in item)):
        group_rows = grouped[key]
        selected.extend(group_rows[:base_cap])
        if len(group_rows) > base_cap:
            candidates_by_pid[key[0]].append(group_rows[base_cap])

    extra_needed = target - len(selected)
    pid_order = sorted(candidates_by_pid, key=int)
    while extra_needed:
        made_progress = False
        for pid in pid_order:
            if extra_needed == 0:
                break
            if candidates_by_pid[pid]:
                selected.append(candidates_by_pid[pid].popleft())
                extra_needed -= 1
                made_progress = True
        if not made_progress:
            raise RuntimeError("Not enough candidates to reach target image count")

    return sorted(selected, key=lambda row: (int(row["pid"]),) + tuple(int(value) for value in group_key(row)[1:]) + row_sort_key(row)), base_cap, target - sum(min(base_cap, len(group_rows)) for group_rows in grouped.values())


def write_config(path: Path, train_dir: Path, num_classes: int, args: argparse.Namespace) -> None:
    with BASE_CONFIG.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    results_dir = EXP / f"results/{EXPERIMENT_NAME}"
    evaluate_dir = EXP / "evaluate/prcc_real_split"
    cfg["results_dir"] = str(results_dir)
    cfg["wandb"]["enable"] = False
    cfg["dataset"]["train_dataset_dir"] = str(train_dir)
    cfg["dataset"]["test_dataset_dir"] = str(PRCC_ROOT / "bounding_box_test")
    cfg["dataset"]["query_dataset_dir"] = str(PRCC_ROOT / "query")
    cfg["dataset"]["num_classes"] = num_classes
    cfg["dataset"]["batch_size"] = args.batch_size
    cfg["dataset"]["val_batch_size"] = args.val_batch_size
    cfg["dataset"]["num_workers"] = args.num_workers
    cfg["train"]["num_epochs"] = args.epochs
    cfg["train"]["checkpoint_interval"] = 10
    cfg["train"]["validation_interval"] = 10
    cfg["train"]["resume_training_checkpoint_path"] = None
    cfg["train"]["results_dir"] = str(results_dir / "train")
    cfg["train"]["gpu_ids"] = [0]
    cfg["train"]["num_gpus"] = 1
    cfg["evaluate"]["gpu_ids"] = [0]
    cfg["evaluate"]["num_gpus"] = 1
    cfg["evaluate"]["checkpoint"] = "???"
    cfg["evaluate"]["results_dir"] = str(evaluate_dir)
    cfg["evaluate"]["output_sampled_matches_plot"] = str(evaluate_dir / "sampled_matches.png")
    cfg["evaluate"]["output_cmc_curve_plot"] = str(evaluate_dir / "cmc_curve.png")
    cfg["evaluate"]["test_dataset"] = None
    cfg["evaluate"]["query_dataset"] = None
    cfg["re_ranking"]["num_query"] = len(image_paths(PRCC_ROOT / "query"))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def main() -> None:
    args = parse_args()
    with SYN_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected, base_cap, extra_groups = choose_exact_target(rows, args.target_images)
    train_dir = EXP / f"data/{EXPERIMENT_NAME}/bounding_box_train"
    if args.rebuild and train_dir.parent.parent.exists():
        shutil.rmtree(train_dir.parent.parent)

    for row in selected:
        src = SYN_ROOT / "bounding_box_train" / row["output_file"]
        if not src.is_file():
            raise FileNotFoundError(src)
        link_or_copy(src, train_dir / row["output_file"])

    filtered_manifest = EXP / "filtered_syntetic_manifest.csv"
    with filtered_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)

    synthetic_ids = {row["pid"] for row in selected}
    config_path = EXP / f"configs/{EXPERIMENT_NAME}.yaml"
    write_config(config_path, train_dir, len(synthetic_ids), args)

    group_sizes = Counter(group_key(row) for row in rows)
    with (EXP / "dataset_summary.tsv").open("w", encoding="utf-8") as handle:
        handle.write("split\timages\tids\tpath\n")
        handle.write(f"syntetic_original_train\t{len(rows)}\t{len({row['pid'] for row in rows})}\t{SYN_ROOT / 'bounding_box_train'}\n")
        handle.write(f"{EXPERIMENT_NAME}\t{len(selected)}\t{len(synthetic_ids)}\t{train_dir}\n")
        handle.write(f"prcc_query_eval_only\t{len(image_paths(PRCC_ROOT / 'query'))}\t71\t{PRCC_ROOT / 'query'}\n")
        handle.write(f"prcc_gallery_eval_only\t{len(image_paths(PRCC_ROOT / 'bounding_box_test'))}\t71\t{PRCC_ROOT / 'bounding_box_test'}\n")

    with (EXP / "DATA_AUDIT.md").open("w", encoding="utf-8") as handle:
        handle.write(f"# {DISPLAY_NAME} Data Audit\n\n")
        handle.write("## Scope\n\n")
        handle.write("Training uses synthetic crops only. Real PRCC query and gallery images are referenced only for later target-domain evaluation.\n\n")
        handle.write("## Filtering Rule\n\n")
        handle.write(f"The original manifest contains `{len(rows)}` crops in `{len(group_sizes)}` person-at-moment groups across `{len(synthetic_ids)}` identities. ")
        handle.write(f"The exact `{len(selected)}`-crop training set keeps up to `{base_cap}` lowest-ID variants from every group, then adds the next variant from `{extra_groups}` groups. ")
        handle.write("Extra groups are selected deterministically in identity-balanced round-robin order.\n\n")
        handle.write("## Training Configuration\n\n")
        handle.write("- Backbone: `swin_base_patch4_window7_224`\n")
        handle.write("- Input: `256x128`\n")
        handle.write(f"- Epochs: `{args.epochs}`\n")
        handle.write(f"- Batch size: `{args.batch_size}`\n")
        handle.write("- Optimizer: `SGD`, base LR `0.0006`, momentum `0.9`, weight decay `0.0001`\n")
        handle.write("- Schedule: LR steps `[40, 70]`, cosine warmup for `20` epochs\n")
        handle.write("- Sampling: `softmax_triplet`, `4` instances per identity\n")

    print(f"Original synthetic crops: {len(rows)}")
    print(f"Selected synthetic crops: {len(selected)}")
    print(f"Synthetic identities: {len(synthetic_ids)}")
    print(f"Base variants per group: {base_cap}")
    print(f"Groups receiving one extra variant: {extra_groups}")
    print(f"Train directory: {train_dir}")
    print(f"Config: {config_path}")


if __name__ == "__main__":
    main()
