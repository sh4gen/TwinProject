#!/usr/bin/env python3
"""Build controlled PRCC plain and PRCC + filtered synthetic TAO experiments."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import yaml


ROOT = Path("/mnt/2tb_ssd/TwinProject")
EXP = ROOT / "experiments/reid/prcc_syntetic_filtered_seq"
PRCC_ROOT = ROOT / "experiments/reid/prcc/data"
SYN_ROOT = ROOT / "datasets/final_syntetic_market1501"
SYN_MANIFEST = SYN_ROOT / "manifest.csv"
BASE_CONFIG = ROOT / "experiments/reid/duke/duke_swin_working_plain_eval.yaml"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PID_RE = re.compile(r"^(?P<pid>\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild generated data directories.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--variants-per-group", type=int, default=3)
    parser.add_argument("--synthetic-offset", type=int, default=None)
    return parser.parse_args()


def image_paths(path: Path) -> list[Path]:
    return sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS)


def pid_from_name(path: Path | str) -> int:
    name = path.name if isinstance(path, Path) else path
    match = PID_RE.match(name)
    if not match:
        raise ValueError(f"Cannot parse Market-1501 PID from filename: {name}")
    return int(match.group("pid"))


def relabel_name(filename: str, offset: int) -> str:
    return PID_RE.sub(f"{pid_from_name(filename) + offset:04d}_", filename, count=1)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def read_synthetic_manifest() -> list[dict[str, str]]:
    with SYN_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def variant_group_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row["pid"],
        row["camera_id"],
        row["sequence_id"],
        row["frame_id"],
        row["source_box_index"],
    )


def choose_representatives(rows: Iterable[dict[str, str]], variants_per_group: int) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[variant_group_key(row)].append(row)

    selected = []
    for group_rows in grouped.values():
        selected.extend(
            sorted(
                group_rows,
                key=lambda row: (int(row["variant_id"]), int(row["encoded_frame"]), row["output_file"]),
            )[:variants_per_group]
        )
    return sorted(
        selected,
        key=lambda row: (
            int(row["pid"]),
            int(row["camera_id"]),
            int(row["sequence_id"]),
            int(row["frame_id"]),
            int(row["source_box_index"]),
        ),
    )


def write_filtered_manifest(selected: list[dict[str, str]], output_path: Path, synthetic_offset: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(selected[0].keys()) + ["relabeled_output_file"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            out = dict(row)
            out["relabeled_output_file"] = relabel_name(row["output_file"], synthetic_offset)
            writer.writerow(out)


def copy_filtered_synthetic(selected: list[dict[str, str]], synthetic_offset: int, filtered_dir: Path) -> None:
    for row in selected:
        src = SYN_ROOT / "bounding_box_train" / row["output_file"]
        dst = filtered_dir / relabel_name(row["output_file"], synthetic_offset)
        if not src.is_file():
            raise FileNotFoundError(src)
        link_or_copy(src, dst)


def copy_combined_train(prcc_images: list[Path], filtered_dir: Path, combined_dir: Path) -> None:
    for src in prcc_images:
        link_or_copy(src, combined_dir / src.name)
    for src in image_paths(filtered_dir):
        link_or_copy(src, combined_dir / src.name)


def load_base_config() -> dict:
    with BASE_CONFIG.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_config(
    config_path: Path,
    series: str,
    train_dir: Path,
    num_classes: int,
    query_count: int,
    args: argparse.Namespace,
) -> None:
    cfg = load_base_config()
    results_dir = EXP / "results" / series
    evaluate_dir = EXP / "evaluate" / series

    cfg["results_dir"] = str(results_dir)
    cfg["wandb"]["enable"] = False
    cfg["model"]["input_width"] = 128
    cfg["model"]["input_height"] = 256
    cfg["model"]["feat_dim"] = 1024
    cfg["model"]["dropout_rate"] = 0.1
    cfg["model"]["drop_path"] = 0.1
    cfg["model"]["drop_out"] = 0.1

    cfg["dataset"]["train_dataset_dir"] = str(train_dir)
    cfg["dataset"]["test_dataset_dir"] = str(PRCC_ROOT / "bounding_box_test")
    cfg["dataset"]["query_dataset_dir"] = str(PRCC_ROOT / "query")
    cfg["dataset"]["num_classes"] = num_classes
    cfg["dataset"]["batch_size"] = args.batch_size
    cfg["dataset"]["val_batch_size"] = args.val_batch_size
    cfg["dataset"]["num_workers"] = args.num_workers
    cfg["dataset"]["pixel_mean"] = [0.5, 0.5, 0.5]
    cfg["dataset"]["pixel_std"] = [0.5, 0.5, 0.5]
    cfg["dataset"]["padding"] = 10
    cfg["dataset"]["prob"] = 0.5
    cfg["dataset"]["re_prob"] = 0.5
    cfg["dataset"]["sampler"] = "softmax_triplet"
    cfg["dataset"]["num_instances"] = 4

    cfg["train"]["num_epochs"] = args.epochs
    cfg["train"]["checkpoint_interval"] = 10
    cfg["train"]["validation_interval"] = 10
    cfg["train"]["resume_training_checkpoint_path"] = None
    cfg["train"]["results_dir"] = str(results_dir / "train")
    cfg["train"]["gpu_ids"] = [0]
    cfg["train"]["num_gpus"] = 1
    cfg["train"]["optim"]["name"] = "SGD"
    cfg["train"]["optim"]["lr_steps"] = [40, 70]
    cfg["train"]["optim"]["gamma"] = 0.1
    cfg["train"]["optim"]["bias_lr_factor"] = 2.0
    cfg["train"]["optim"]["weight_decay"] = 0.0001
    cfg["train"]["optim"]["weight_decay_bias"] = 0.0001
    cfg["train"]["optim"]["warmup_factor"] = 0.01
    cfg["train"]["optim"]["warmup_iters"] = 0
    cfg["train"]["optim"]["warmup_epochs"] = 20
    cfg["train"]["optim"]["warmup_method"] = "cosine"
    cfg["train"]["optim"]["base_lr"] = 0.0006
    cfg["train"]["optim"]["momentum"] = 0.9
    cfg["train"]["optim"]["triplet_loss_margin"] = 0.3

    cfg["evaluate"]["gpu_ids"] = [0]
    cfg["evaluate"]["num_gpus"] = 1
    cfg["evaluate"]["checkpoint"] = "???"
    cfg["evaluate"]["results_dir"] = str(evaluate_dir)
    cfg["evaluate"]["output_sampled_matches_plot"] = str(evaluate_dir / "sampled_matches.png")
    cfg["evaluate"]["output_cmc_curve_plot"] = str(evaluate_dir / "cmc_curve.png")
    cfg["evaluate"]["test_dataset"] = None
    cfg["evaluate"]["query_dataset"] = None
    cfg["re_ranking"]["num_query"] = query_count

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def count_ids(paths: Iterable[Path]) -> int:
    return len({pid_from_name(path) for path in paths})


def main() -> None:
    args = parse_args()
    prcc_train = PRCC_ROOT / "bounding_box_train"
    if not prcc_train.is_dir():
        raise SystemExit(f"Missing PRCC train dir: {prcc_train}")
    if not SYN_MANIFEST.is_file():
        raise SystemExit(f"Missing synthetic manifest: {SYN_MANIFEST}")

    data_root = EXP / "data"
    if args.rebuild and data_root.exists():
        shutil.rmtree(data_root)

    filtered_dir = data_root / "filtered_syntetic" / "bounding_box_train"
    combined_dir = data_root / "prcc_filtered_syntetic" / "bounding_box_train"
    filtered_manifest = EXP / "filtered_syntetic_manifest.csv"
    summary_path = EXP / "dataset_summary.tsv"
    audit_path = EXP / "DATA_AUDIT.md"

    rows = read_synthetic_manifest()
    selected = choose_representatives(rows, args.variants_per_group)
    group_sizes = Counter(variant_group_key(row) for row in rows)
    prcc_images = image_paths(prcc_train)
    prcc_ids = {pid_from_name(path) for path in prcc_images}
    synthetic_ids = {int(row["pid"]) for row in selected}
    synthetic_offset = args.synthetic_offset if args.synthetic_offset is not None else max(prcc_ids) + 1000

    copy_filtered_synthetic(selected, synthetic_offset, filtered_dir)
    copy_combined_train(prcc_images, filtered_dir, combined_dir)
    write_filtered_manifest(selected, filtered_manifest, synthetic_offset)

    filtered_images = image_paths(filtered_dir)
    combined_images = image_paths(combined_dir)
    query_images = image_paths(PRCC_ROOT / "query")
    gallery_images = image_paths(PRCC_ROOT / "bounding_box_test")

    write_config(
        EXP / "configs/prcc_plain_swin.yaml",
        "prcc_plain_swin",
        prcc_train,
        len(prcc_ids),
        len(query_images),
        args,
    )
    write_config(
        EXP / "configs/prcc_filtered_syntetic_swin.yaml",
        "prcc_filtered_syntetic_swin",
        combined_dir,
        len(prcc_ids) + len(synthetic_ids),
        len(query_images),
        args,
    )

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("split\timages\tids\tpath\n")
        handle.write(f"prcc_train\t{len(prcc_images)}\t{len(prcc_ids)}\t{prcc_train}\n")
        handle.write(f"syntetic_original_train\t{len(rows)}\t{len({row['pid'] for row in rows})}\t{SYN_ROOT / 'bounding_box_train'}\n")
        handle.write(f"syntetic_filtered_train\t{len(filtered_images)}\t{count_ids(filtered_images)}\t{filtered_dir}\n")
        handle.write(f"combined_train\t{len(combined_images)}\t{count_ids(combined_images)}\t{combined_dir}\n")
        handle.write(f"prcc_query\t{len(query_images)}\t{count_ids(query_images)}\t{PRCC_ROOT / 'query'}\n")
        handle.write(f"prcc_gallery\t{len(gallery_images)}\t{count_ids(gallery_images)}\t{PRCC_ROOT / 'bounding_box_test'}\n")

    with audit_path.open("w", encoding="utf-8") as handle:
        handle.write("# PRCC Plain And Filtered Synthetic Data Audit\n\n")
        handle.write("## Controlled Comparison\n\n")
        handle.write("Both stages start independently from the same Swin Base pretrained model and use identical optimized hyperparameters. ")
        handle.write("Stage 1 trains on real PRCC train only. Stage 2 trains on real PRCC train plus filtered synthetic crops. ")
        handle.write("Evaluation always uses only real PRCC `query` and `bounding_box_test`.\n\n")
        handle.write("## Filtering Rule\n\n")
        handle.write("Rows were grouped by `pid`, `camera_id`, `sequence_id`, `frame_id`, and `source_box_index`; ")
        handle.write(f"up to `{args.variants_per_group}` crops were retained from each group by choosing the lowest `variant_id` values.\n\n")
        handle.write("## Counts\n\n")
        handle.write(f"- PRCC train images: `{len(prcc_images)}`\n")
        handle.write(f"- PRCC train IDs: `{len(prcc_ids)}`\n")
        handle.write(f"- Original synthetic train images: `{len(rows)}`\n")
        handle.write(f"- Unique person-at-moment synthetic groups: `{len(group_sizes)}`\n")
        handle.write(f"- Synthetic images kept after filtering: `{len(selected)}`\n")
        handle.write(f"- Synthetic IDs kept: `{len(synthetic_ids)}`\n")
        handle.write(f"- Combined train images: `{len(combined_images)}`\n")
        handle.write(f"- Combined train IDs: `{count_ids(combined_images)}`\n")
        handle.write(f"- Synthetic PID offset: `{synthetic_offset}`\n\n")
        handle.write("## Optimized Hyperparameters\n\n")
        handle.write("- Backbone: `swin_base_patch4_window7_224`\n")
        handle.write("- Input: `256x128`\n")
        handle.write(f"- Epochs per stage: `{args.epochs}`\n")
        handle.write(f"- Train batch size: `{args.batch_size}`\n")
        handle.write("- Optimizer: `SGD`, base LR `0.0006`, momentum `0.9`, weight decay `0.0001`\n")
        handle.write("- Schedule: LR steps `[40, 70]`, cosine warmup for `20` epochs\n")
        handle.write("- Sampling: `softmax_triplet`, `4` instances per identity\n")
        handle.write("- Augmentation: horizontal flip `0.5`, random erasing `0.5`, padding `10`\n")
        handle.write("- Re-ranking: `k1=20`, `k2=6`, `lambda=0.3`\n\n")
        handle.write("## Historical Split Note\n\n")
        handle.write("The current local PRCC train directory contains `150` IDs. An older recorded PRCC experiment YAML contains `221` classes, ")
        handle.write("so that historical run came from a different split and must remain a separate reference.\n")

    print(f"Wrote filtered synthetic data: {filtered_dir}")
    print(f"Wrote combined train data: {combined_dir}")
    print(f"Wrote configs: {EXP / 'configs'}")
    print(f"Wrote audit: {audit_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
