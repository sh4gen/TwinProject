#!/usr/bin/env python3
"""Build Duke + filtered synthetic ReID data and a TAO Swin config."""

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
EXP = ROOT / "experiments/reid/duke_syntetic_filtered_seq"
DUKE_ROOT = ROOT / "experiments/reid/duke/data"
SYN_ROOT = ROOT / "datasets/final_syntetic_market1501"
SYN_MANIFEST = SYN_ROOT / "manifest.csv"
BASE_CONFIG = ROOT / "experiments/reid/duke/duke_swin_working_plain_eval.yaml"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PID_RE = re.compile(r"^(?P<pid>\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild generated data directories.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
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
    old_pid = pid_from_name(filename)
    return PID_RE.sub(f"{old_pid + offset:04d}_", filename, count=1)


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
                key=lambda row: (
                    int(row["variant_id"]),
                    int(row["encoded_frame"]),
                    row["output_file"],
                ),
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


def copy_combined_train(duke_images: list[Path], filtered_dir: Path, combined_dir: Path) -> None:
    for src in duke_images:
        link_or_copy(src, combined_dir / src.name)
    for src in image_paths(filtered_dir):
        link_or_copy(src, combined_dir / src.name)


def load_base_config() -> dict:
    with BASE_CONFIG.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_config(config_path: Path, train_dir: Path, num_classes: int, args: argparse.Namespace) -> None:
    cfg = load_base_config()
    results_dir = EXP / "results" / "duke_filtered_syntetic"
    evaluate_dir = EXP / "evaluate" / "duke_filtered_syntetic"

    cfg["results_dir"] = str(results_dir)
    cfg["dataset"]["train_dataset_dir"] = str(train_dir)
    cfg["dataset"]["test_dataset_dir"] = str(DUKE_ROOT / "bounding_box_test")
    cfg["dataset"]["query_dataset_dir"] = str(DUKE_ROOT / "query")
    cfg["dataset"]["num_classes"] = num_classes
    cfg["dataset"]["batch_size"] = args.batch_size
    cfg["dataset"]["val_batch_size"] = args.val_batch_size
    cfg["dataset"]["num_workers"] = args.num_workers

    cfg["train"]["num_epochs"] = args.epochs
    cfg["train"]["checkpoint_interval"] = 10
    cfg["train"]["validation_interval"] = 2
    cfg["train"]["resume_training_checkpoint_path"] = None
    cfg["train"]["results_dir"] = str(results_dir / "train")
    cfg["train"]["gpu_ids"] = [0]
    cfg["train"]["num_gpus"] = 1
    cfg["train"]["optim"]["lr_steps"] = [50, 100] if args.epochs >= 200 else [max(1, int(args.epochs * 0.5)), max(2, int(args.epochs * 0.75))]

    cfg["evaluate"]["gpu_ids"] = [0]
    cfg["evaluate"]["num_gpus"] = 1
    cfg["evaluate"]["checkpoint"] = "???"
    cfg["evaluate"]["results_dir"] = str(evaluate_dir)
    cfg["evaluate"]["output_sampled_matches_plot"] = str(evaluate_dir / "sampled_matches.png")
    cfg["evaluate"]["output_cmc_curve_plot"] = str(evaluate_dir / "cmc_curve.png")
    cfg["evaluate"]["test_dataset"] = None
    cfg["evaluate"]["query_dataset"] = None

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def count_ids(paths: Iterable[Path]) -> int:
    return len({pid_from_name(path) for path in paths})


def main() -> None:
    args = parse_args()

    duke_train = DUKE_ROOT / "bounding_box_train"
    if not duke_train.is_dir():
        raise SystemExit(f"Missing Duke train dir: {duke_train}")
    if not SYN_MANIFEST.is_file():
        raise SystemExit(f"Missing synthetic manifest: {SYN_MANIFEST}")

    data_root = EXP / "data"
    if args.rebuild and data_root.exists():
        shutil.rmtree(data_root)

    filtered_dir = data_root / "filtered_syntetic" / "bounding_box_train"
    combined_dir = data_root / "duke_filtered_syntetic" / "bounding_box_train"
    filtered_manifest = EXP / "filtered_syntetic_manifest.csv"
    audit_path = EXP / "DATA_AUDIT.md"
    summary_path = EXP / "dataset_summary.tsv"
    config_path = EXP / "configs" / "duke_filtered_syntetic.yaml"

    rows = read_synthetic_manifest()
    selected = choose_representatives(rows, args.variants_per_group)
    group_sizes = Counter(variant_group_key(row) for row in rows)

    duke_images = image_paths(duke_train)
    duke_ids = {pid_from_name(path) for path in duke_images}
    synthetic_ids = {int(row["pid"]) for row in selected}
    synthetic_offset = args.synthetic_offset if args.synthetic_offset is not None else max(duke_ids) + 1000

    copy_filtered_synthetic(selected, synthetic_offset, filtered_dir)
    copy_combined_train(duke_images, filtered_dir, combined_dir)
    write_filtered_manifest(selected, filtered_manifest, synthetic_offset)
    write_config(config_path, combined_dir, len(duke_ids) + len(synthetic_ids), args)

    filtered_images = image_paths(filtered_dir)
    combined_images = image_paths(combined_dir)
    duke_query = image_paths(DUKE_ROOT / "query")
    duke_gallery = image_paths(DUKE_ROOT / "bounding_box_test")

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("split\timages\tids\tpath\n")
        handle.write(f"duke_train\t{len(duke_images)}\t{len(duke_ids)}\t{duke_train}\n")
        handle.write(f"syntetic_original_train\t{len(rows)}\t{len({row['pid'] for row in rows})}\t{SYN_ROOT / 'bounding_box_train'}\n")
        handle.write(f"syntetic_filtered_train\t{len(filtered_images)}\t{count_ids(filtered_images)}\t{filtered_dir}\n")
        handle.write(f"combined_train\t{len(combined_images)}\t{count_ids(combined_images)}\t{combined_dir}\n")
        handle.write(f"duke_query\t{len(duke_query)}\t{count_ids(duke_query)}\t{DUKE_ROOT / 'query'}\n")
        handle.write(f"duke_gallery\t{len(duke_gallery)}\t{count_ids(duke_gallery)}\t{DUKE_ROOT / 'bounding_box_test'}\n")

    with audit_path.open("w", encoding="utf-8") as handle:
        handle.write("# Duke Filtered Synthetic Data Audit\n\n")
        handle.write("## Filtering Rule\n\n")
        handle.write("Rows were grouped by `pid`, `camera_id`, `sequence_id`, `frame_id`, and `source_box_index`; ")
        handle.write(f"up to `{args.variants_per_group}` crops were retained from each group by choosing the lowest `variant_id` values.\n\n")
        handle.write("## Counts\n\n")
        handle.write(f"- Original synthetic train images: `{len(rows)}`\n")
        handle.write(f"- Unique person-at-moment groups: `{len(group_sizes)}`\n")
        handle.write(f"- Synthetic images kept after filtering: `{len(selected)}`\n")
        handle.write(f"- Variants kept per group: `{args.variants_per_group}`\n")
        handle.write(f"- Synthetic IDs kept: `{len(synthetic_ids)}`\n")
        handle.write(f"- Duke train images: `{len(duke_images)}`\n")
        handle.write(f"- Combined train images: `{len(combined_images)}`\n")
        handle.write(f"- Combined train IDs: `{count_ids(combined_images)}`\n")
        handle.write(f"- Synthetic PID offset: `{synthetic_offset}`\n\n")
        handle.write("## Variant Group Sizes\n\n")
        handle.write("| Variants per same moment | Number of groups |\n")
        handle.write("| --- | --- |\n")
        for size, count in sorted(Counter(group_sizes.values()).items()):
            handle.write(f"| {size} | {count} |\n")
        handle.write("\n## Evaluation Split Policy\n\n")
        handle.write("Training uses Duke train plus filtered synthetic train. Evaluation uses only Duke `query` and `bounding_box_test`; synthetic query/test are not used.\n")

    print(f"Wrote filtered synthetic data: {filtered_dir}")
    print(f"Wrote combined train data: {combined_dir}")
    print(f"Wrote config: {config_path}")
    print(f"Wrote audit: {audit_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
