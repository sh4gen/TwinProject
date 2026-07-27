#!/usr/bin/env python3
"""Build a namespaced multi-domain ReID train, validation, and stress-test dataset."""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import shutil
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


ROOT = Path("/mnt/2tb_ssd/TwinProject")
EXP = ROOT / "experiments/reid/generalized_reid_swin"
REID = ROOT / "experiments/reid"
SYN_ROOT = ROOT / "datasets/final_syntetic_market1501"
SYN_MANIFEST = SYN_ROOT / "manifest.csv"
BASE_CONFIG = REID / "prcc_syntetic_filtered_seq/configs/prcc_plain_swin.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
NAME_RE = re.compile(r"^(?P<pid>\d+)_c(?P<camera>\d+)")


@dataclass(frozen=True)
class Domain:
    name: str
    code: int
    pid_offset: int
    camera_offset: int

    @property
    def data_root(self) -> Path:
        return REID / self.name / "data"


DOMAINS = (
    Domain("duke", 1, 10_000, 100),
    Domain("ltcc", 2, 20_000, 200),
    Domain("prcc", 3, 30_000, 300),
)
SYNTHETIC = Domain("syntetic", 4, 40_000, 400)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-images", type=int, default=116_920)
    parser.add_argument("--validation-percent", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def image_paths(path: Path) -> list[Path]:
    return sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS)


def parsed_name(path: Path) -> tuple[int, int]:
    match = NAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Cannot parse Market-1501 filename: {path}")
    return int(match.group("pid")), int(match.group("camera"))


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def generated_name(domain: Domain, src: Path, sequence: int) -> str:
    pid, camera = parsed_name(src)
    return f"{pid + domain.pid_offset:05d}_c{camera + domain.camera_offset}s{domain.code:02d}_{sequence:07d}_00{src.suffix.lower()}"


def link_namespaced(domain: Domain, src: Path, dst_dir: Path, sequence: int) -> Path:
    dst = dst_dir / generated_name(domain, src, sequence)
    link_or_copy(src, dst)
    return dst


def group_by_pid(paths: Iterable[Path]) -> dict[int, list[Path]]:
    grouped: dict[int, list[Path]] = defaultdict(list)
    for path in paths:
        pid, _ = parsed_name(path)
        grouped[pid].append(path)
    return {pid: sorted(pid_paths) for pid, pid_paths in grouped.items()}


def choose_validation_ids(paths: Iterable[Path], percent: float, seed: int) -> set[int]:
    grouped = group_by_pid(paths)
    eligible = []
    for pid, pid_paths in grouped.items():
        cameras = {parsed_name(path)[1] for path in pid_paths}
        if len(cameras) >= 2 and len(pid_paths) >= 2:
            eligible.append(pid)
    count = max(1, round(len(grouped) * percent))
    if count > len(eligible):
        raise ValueError(f"Need {count} validation IDs but only {len(eligible)} have cross-camera matches")
    rng = random.Random(seed)
    return set(rng.sample(sorted(eligible), count))


def create_real_train_and_validation(args: argparse.Namespace, data_root: Path) -> tuple[list[dict[str, object]], int, int, int]:
    train_dir = data_root / "train/bounding_box_train"
    val_query = data_root / "validation/query"
    val_gallery = data_root / "validation/bounding_box_test"
    audit_rows = []
    train_id_count = 0
    val_id_count = 0
    sequence = 0

    for index, domain in enumerate(DOMAINS):
        source = image_paths(domain.data_root / "bounding_box_train")
        grouped = group_by_pid(source)
        validation_ids = choose_validation_ids(source, args.validation_percent, args.seed + index)
        train_ids = set(grouped) - validation_ids
        train_images = 0
        val_query_images = 0
        val_gallery_images = 0

        for pid in sorted(grouped):
            pid_paths = grouped[pid]
            if pid in train_ids:
                for src in pid_paths:
                    link_namespaced(domain, src, train_dir, sequence)
                    sequence += 1
                    train_images += 1
                continue

            cameras = sorted({parsed_name(path)[1] for path in pid_paths})
            query_camera = cameras[0]
            query_src = next(path for path in pid_paths if parsed_name(path)[1] == query_camera)
            link_namespaced(domain, query_src, val_query, sequence)
            sequence += 1
            val_query_images += 1
            for src in pid_paths:
                if src == query_src:
                    continue
                link_namespaced(domain, src, val_gallery, sequence)
                sequence += 1
                val_gallery_images += 1

        train_id_count += len(train_ids)
        val_id_count += len(validation_ids)
        audit_rows.append(
            {
                "domain": domain.name,
                "source_train_images": len(source),
                "source_train_ids": len(grouped),
                "train_images": train_images,
                "train_ids": len(train_ids),
                "validation_query_images": val_query_images,
                "validation_gallery_images": val_gallery_images,
                "validation_ids": len(validation_ids),
            }
        )

    return audit_rows, train_id_count, val_id_count, sequence


def synthetic_group_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row["pid"],
        row["camera_id"],
        row["sequence_id"],
        row["frame_id"],
        row["source_box_index"],
    )


def synthetic_sort_key(row: dict[str, str]) -> tuple[int, int, str]:
    return int(row["variant_id"]), int(row["encoded_frame"]), row["output_file"]


def choose_synthetic_exact_target(rows: Iterable[dict[str, str]], target: int) -> tuple[list[dict[str, str]], int, int]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[synthetic_group_key(row)].append(row)
    for group_rows in grouped.values():
        group_rows.sort(key=synthetic_sort_key)

    if target <= 0 or target > sum(map(len, grouped.values())):
        raise ValueError(f"Invalid synthetic target: {target}")

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
    extra_groups = extra_needed
    while extra_needed:
        made_progress = False
        for pid in sorted(candidates_by_pid, key=int):
            if extra_needed == 0:
                break
            if candidates_by_pid[pid]:
                selected.append(candidates_by_pid[pid].popleft())
                extra_needed -= 1
                made_progress = True
        if not made_progress:
            raise RuntimeError("Not enough synthetic groups to reach requested target")

    selected.sort(
        key=lambda row: (
            int(row["pid"]),
            int(row["camera_id"]),
            int(row["sequence_id"]),
            int(row["frame_id"]),
            int(row["source_box_index"]),
            *synthetic_sort_key(row),
        )
    )
    return selected, base_cap, extra_groups


def link_synthetic(selected: list[dict[str, str]], data_root: Path, starting_sequence: int) -> int:
    train_dir = data_root / "train/bounding_box_train"
    sequence = starting_sequence
    for row in selected:
        src = SYN_ROOT / "bounding_box_train" / row["output_file"]
        if not src.is_file():
            raise FileNotFoundError(src)
        link_namespaced(SYNTHETIC, src, train_dir, sequence)
        sequence += 1
    return sequence


def create_official_stress_test(data_root: Path, starting_sequence: int) -> tuple[list[dict[str, object]], int]:
    stress_query = data_root / "official_stress/query"
    stress_gallery = data_root / "official_stress/bounding_box_test"
    audit_rows = []
    sequence = starting_sequence
    for domain in DOMAINS:
        query = image_paths(domain.data_root / "query")
        gallery = image_paths(domain.data_root / "bounding_box_test")
        for src in query:
            link_namespaced(domain, src, stress_query, sequence)
            sequence += 1
        for src in gallery:
            link_namespaced(domain, src, stress_gallery, sequence)
            sequence += 1
        audit_rows.append(
            {
                "domain": domain.name,
                "query_images": len(query),
                "gallery_images": len(gallery),
                "query_ids": len(group_by_pid(query)),
                "gallery_ids": len(group_by_pid(gallery)),
            }
        )
    return audit_rows, sequence


def write_manifest(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t" if path.suffix == ".tsv" else ",")
        writer.writeheader()
        writer.writerows(rows)


def write_config(path: Path, data_root: Path, num_classes: int, val_queries: int, args: argparse.Namespace) -> None:
    with BASE_CONFIG.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    results_dir = EXP / "results/generalized_swin"
    evaluate_dir = EXP / "evaluate/validation"

    cfg["results_dir"] = str(results_dir)
    cfg["wandb"]["enable"] = False
    cfg["dataset"]["train_dataset_dir"] = str(data_root / "train/bounding_box_train")
    cfg["dataset"]["query_dataset_dir"] = str(data_root / "validation/query")
    cfg["dataset"]["test_dataset_dir"] = str(data_root / "validation/bounding_box_test")
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
    cfg["train"]["optim"]["base_lr"] = 0.0006
    cfg["evaluate"]["gpu_ids"] = [0]
    cfg["evaluate"]["num_gpus"] = 1
    cfg["evaluate"]["checkpoint"] = "???"
    cfg["evaluate"]["results_dir"] = str(evaluate_dir)
    cfg["evaluate"]["output_sampled_matches_plot"] = str(evaluate_dir / "sampled_matches.png")
    cfg["evaluate"]["output_cmc_curve_plot"] = str(evaluate_dir / "cmc_curve.png")
    cfg["evaluate"]["query_dataset"] = None
    cfg["evaluate"]["test_dataset"] = None
    cfg["re_ranking"]["num_query"] = val_queries

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def count_ids(path: Path) -> int:
    return len(group_by_pid(image_paths(path)))


def main() -> None:
    args = parse_args()
    data_root = EXP / "data"
    if args.rebuild and data_root.exists():
        shutil.rmtree(data_root)
    if args.rebuild and (EXP / "manifests").exists():
        shutil.rmtree(EXP / "manifests")

    real_audit, real_train_ids, validation_ids, sequence = create_real_train_and_validation(args, data_root)
    with SYN_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        synthetic_rows = list(csv.DictReader(handle))
    selected_synthetic, synthetic_cap, synthetic_extra = choose_synthetic_exact_target(synthetic_rows, args.synthetic_images)
    sequence = link_synthetic(selected_synthetic, data_root, sequence)
    stress_audit, _ = create_official_stress_test(data_root, sequence)

    write_manifest(EXP / "manifests/real_partitions.tsv", real_audit)
    write_manifest(EXP / "manifests/official_stress.tsv", stress_audit)
    write_manifest(EXP / "manifests/selected_synthetic.csv", selected_synthetic)

    train_dir = data_root / "train/bounding_box_train"
    val_query = data_root / "validation/query"
    val_gallery = data_root / "validation/bounding_box_test"
    stress_query = data_root / "official_stress/query"
    stress_gallery = data_root / "official_stress/bounding_box_test"
    synthetic_ids = len({row["pid"] for row in selected_synthetic})
    write_config(
        EXP / "configs/generalized_swin.yaml",
        data_root,
        real_train_ids + synthetic_ids,
        len(image_paths(val_query)),
        args,
    )

    with (EXP / "dataset_summary.tsv").open("w", encoding="utf-8") as handle:
        handle.write("split\timages\tids\tpath\n")
        handle.write(f"generalized_train\t{len(image_paths(train_dir))}\t{count_ids(train_dir)}\t{train_dir}\n")
        handle.write(f"validation_query\t{len(image_paths(val_query))}\t{count_ids(val_query)}\t{val_query}\n")
        handle.write(f"validation_gallery\t{len(image_paths(val_gallery))}\t{count_ids(val_gallery)}\t{val_gallery}\n")
        handle.write(f"official_stress_query\t{len(image_paths(stress_query))}\t{count_ids(stress_query)}\t{stress_query}\n")
        handle.write(f"official_stress_gallery\t{len(image_paths(stress_gallery))}\t{count_ids(stress_gallery)}\t{stress_gallery}\n")

    real_train_images = sum(int(row["train_images"]) for row in real_audit)
    with (EXP / "DATA_AUDIT.md").open("w", encoding="utf-8") as handle:
        handle.write("# Generalized ReID Swin Data Audit\n\n")
        handle.write("## Purpose\n\n")
        handle.write("Train one Swin Base ReID model for real cross-domain generalization. Training combines namespaced real Duke, LTCC, and PRCC identities with a filtered 50% synthetic pool. ")
        handle.write("Checkpoint selection uses a separate identity-disjoint real validation split. Official benchmark query/gallery folders remain untouched for final evaluation.\n\n")
        handle.write("## Namespace Policy\n\n")
        handle.write("| Domain | PID offset | Camera offset |\n| --- | ---: | ---: |\n")
        for domain in (*DOMAINS, SYNTHETIC):
            handle.write(f"| {domain.name} | {domain.pid_offset} | {domain.camera_offset} |\n")
        handle.write("\n")
        handle.write("## Real Partition\n\n")
        handle.write("| Domain | Source train images | Source IDs | Train images | Train IDs | Validation query | Validation gallery | Validation IDs |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in real_audit:
            handle.write(
                f"| {row['domain']} | {row['source_train_images']} | {row['source_train_ids']} | {row['train_images']} | "
                f"{row['train_ids']} | {row['validation_query_images']} | {row['validation_gallery_images']} | {row['validation_ids']} |\n"
            )
        handle.write("\n")
        handle.write("## Synthetic Filter\n\n")
        handle.write(f"- Original crops: `{len(synthetic_rows)}`\n")
        handle.write(f"- Selected crops: `{len(selected_synthetic)}` (`{len(selected_synthetic) / len(synthetic_rows) * 100:.2f}%`)\n")
        handle.write(f"- Synthetic IDs: `{synthetic_ids}`\n")
        handle.write(f"- Retained variants per person-at-moment group: up to `{synthetic_cap}`, plus one extra variant for `{synthetic_extra}` groups\n")
        handle.write("- Extra groups are selected deterministically in identity-balanced round-robin order.\n\n")
        handle.write("## Training Totals\n\n")
        handle.write(f"- Real train images: `{real_train_images}`\n")
        handle.write(f"- Synthetic train images: `{len(selected_synthetic)}`\n")
        handle.write(f"- Total train images: `{len(image_paths(train_dir))}`\n")
        handle.write(f"- Real train IDs: `{real_train_ids}`\n")
        handle.write(f"- Synthetic train IDs: `{synthetic_ids}`\n")
        handle.write(f"- Total classes: `{count_ids(train_dir)}`\n")
        handle.write(f"- Held-out real validation IDs: `{validation_ids}`\n\n")
        handle.write("## Hyperparameters\n\n")
        handle.write("- Backbone: `swin_base_patch4_window7_224`\n")
        handle.write("- Pretrained initialization: Market-1501/AICity Swin Base `.tlt`\n")
        handle.write("- Input: `256x128`\n")
        handle.write(f"- Epochs: `{args.epochs}`\n")
        handle.write(f"- Batch size: `{args.batch_size}`\n")
        handle.write("- Sampler: `softmax_triplet`, `4` instances per identity\n")
        handle.write("- Optimizer: SGD, LR `0.0006`, momentum `0.9`, weight decay `0.0001`\n")
        handle.write("- LR schedule: steps `[40, 70]`, gamma `0.1`, cosine warmup for `20` epochs\n")
        handle.write("- Augmentation: horizontal flip `0.5`, random erasing `0.5`, padding `10`\n")
        handle.write("- Validation and checkpoint interval: every `10` epochs\n\n")
        handle.write("## Evaluation Policy\n\n")
        handle.write("Final evaluation must report Duke, LTCC, and PRCC official query/gallery metrics independently. ")
        handle.write("The namespaced combined official split is an additional cross-domain stress test, not a replacement for the standard per-dataset metrics.\n")

    print(f"Generalized train images: {len(image_paths(train_dir))}")
    print(f"Generalized train classes: {count_ids(train_dir)}")
    print(f"Real train images: {real_train_images}")
    print(f"Synthetic train images: {len(selected_synthetic)}")
    print(f"Validation queries: {len(image_paths(val_query))}")
    print(f"Validation gallery images: {len(image_paths(val_gallery))}")
    print(f"Official stress queries: {len(image_paths(stress_query))}")
    print(f"Official stress gallery images: {len(image_paths(stress_gallery))}")
    print(f"Config: {EXP / 'configs/generalized_swin.yaml'}")


if __name__ == "__main__":
    main()
