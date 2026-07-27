#!/usr/bin/env python3
"""Generate a repository-wide TAO ReID checkpoint evaluation report."""

from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-reid-report")
import matplotlib.pyplot as plt


ROOT = Path("/mnt/2tb_ssd/TwinProject")
REID = ROOT / "experiments/reid"
OUT = REID / "repository_training_results"
GRAPH_DIR = OUT / "graphs"
TABLE_DIR = OUT / "tables"

CHECKPOINT_RE = re.compile(r"(?:model_|archived_)?epoch_(?P<epoch>\d+)(?:_step_(?P<step>\d+))?")
METRIC_RE = re.compile(r"(mAP|Rank-1|Rank-5|Rank-10)\s*│\s*([0-9.]+)%")


@dataclass(frozen=True)
class MetricRow:
    series: str
    checkpoint: str
    epoch: int | None
    step: int | None
    map: float | None
    rank1: float | None
    rank5: float | None
    rank10: float | None
    status: str
    source: str


@dataclass(frozen=True)
class InventoryRow:
    series: str
    checkpoint: str
    epoch: int | None
    step: int | None
    path: str
    size_mib: float
    evaluation_status: str


SERIES_META = {
    "ltcc_resnet_0.1.1": ("LTCC", "LTCC ResNet 0.1.1", "legacy real-only baseline"),
    "ltcc_resnet_0.1.2": ("LTCC", "LTCC ResNet 0.1.2", "legacy real-only baseline"),
    "ltcc_resnet_0.1.3": ("LTCC", "LTCC ResNet 0.1.3", "legacy real-only baseline"),
    "ltcc_resnet_0.1.4": ("LTCC", "LTCC ResNet 0.1.4", "legacy real-only baseline"),
    "ltcc_archive_v1.0.0": ("LTCC", "LTCC archived v1.0.0 curve", "historical per-epoch validation metrics"),
    "ltcc_archive_v1.0.1": ("LTCC", "LTCC archived v1.0.1 curve", "historical per-epoch validation metrics"),
    "ltcc_pretrained_reference": ("LTCC", "LTCC pretrained reference", "historical pretrained benchmark"),
    "ltcc_swin_1.0.1": ("LTCC", "LTCC Swin 1.0.1", "legacy real-only Swin run"),
    "ltcc_swin_misfiled_0.1.4_epoch0": ("LTCC", "LTCC Swin misfiled 0.1.4 epoch 0", "Swin checkpoint stored in later ResNet directory"),
    "ltcc_swin_plain": ("LTCC", "LTCC Swin plain", "real-only Swin baseline"),
    "ltcc_swin_unfiltered_synthetic": ("LTCC", "LTCC + unfiltered synthetic Swin", "older unfiltered synthetic mix"),
    "ltcc_resnet_synthetic_transfer": ("LTCC", "LTCC synthetic transfer ResNet", "synthetic-only transfer attempt"),
    "ltcc_filtered_synthetic_3_variants": ("LTCC", "LTCC + filtered synthetic Swin", "three representatives per moment"),
    "ltcc_syntetic_10": ("LTCC", "LTCC + 10% synthetic Swin", "percentage sweep"),
    "ltcc_syntetic_25": ("LTCC", "LTCC + 25% synthetic Swin", "percentage sweep, interrupted"),
    "ltcc_syntetic_50": ("LTCC", "LTCC + 50% synthetic Swin", "percentage sweep, interrupted"),
    "ltcc_syntetic_75": ("LTCC", "LTCC + 75% synthetic Swin", "percentage sweep, no checkpoint"),
    "ltcc_syntetic_100": ("LTCC", "LTCC + 100% synthetic Swin", "percentage sweep, no checkpoint"),
    "syntetic_only_100": ("LTCC", "Synthetic-only Swin evaluated on LTCC", "percentage sweep"),
    "syntetic_only_filtered_30k_duke_raw": ("Duke", "Synthetic-only filtered 30k on Duke (raw)", "30k synthetic-only training; raw retrieval without re-ranking"),
    "syntetic_only_filtered_30k_ltcc_raw": ("LTCC", "Synthetic-only filtered 30k on LTCC (raw)", "30k synthetic-only training; raw retrieval without re-ranking"),
    "syntetic_only_filtered_30k_prcc_raw": ("PRCC", "Synthetic-only filtered 30k on PRCC (raw)", "30k synthetic-only training; raw retrieval without re-ranking"),
    "syntetic_only_filtered_100k_duke_raw": ("Duke", "Synthetic-only filtered 100k on Duke (raw)", "100k synthetic-only training; raw retrieval without re-ranking"),
    "syntetic_only_filtered_100k_ltcc_raw": ("LTCC", "Synthetic-only filtered 100k on LTCC (raw)", "100k synthetic-only training; raw retrieval without re-ranking"),
    "syntetic_only_filtered_100k_prcc_raw": ("PRCC", "Synthetic-only filtered 100k on PRCC (raw)", "100k synthetic-only training; raw retrieval without re-ranking"),
    "pretrained_swin_market1501_aicity156": ("LTCC", "Pretrained Swin marker", "pretrained reference evaluated on LTCC"),
    "pretrained_swin_ccvid_raw": ("CCVID", "Pretrained Swin on CCVID (raw)", "Market1501 + AICity156 pretrained reference; raw retrieval without re-ranking"),
    "pretrained_swin_duke_raw": ("Duke", "Pretrained Swin on Duke (raw)", "Market1501 + AICity156 pretrained reference; raw retrieval without re-ranking"),
    "pretrained_swin_ltcc_raw": ("LTCC", "Pretrained Swin on LTCC (raw)", "Market1501 + AICity156 pretrained reference; raw retrieval without re-ranking"),
    "pretrained_swin_prcc_raw": ("PRCC", "Pretrained Swin on PRCC (raw)", "Market1501 + AICity156 pretrained reference; raw retrieval without re-ranking"),
    "pretrained_swin_uliri_raw": ("ULIRI", "Pretrained Swin on ULIRI (raw)", "Market1501 + AICity156 pretrained reference; raw retrieval without re-ranking"),
    "pretrained_swin_synthetic_market1501_raw": ("Synthetic Market1501", "Pretrained Swin on synthetic Market1501 (raw)", "Market1501 + AICity156 pretrained reference; raw retrieval without re-ranking"),
    "duke_swin_plain_external": ("Duke", "Duke plain Swin external final", "transferred final checkpoint"),
    "duke_swin_plain_local_partial": ("Duke", "Duke plain Swin local partial", "local interrupted run"),
    "duke_swin_unfiltered_synthetic": ("Duke", "Duke + synthetic Swin", "older synthetic mix"),
    "duke_filtered_synthetic_3_variants": ("Duke", "Duke + filtered synthetic Swin", "three representatives per moment"),
    "prcc_resnet_0.0.1": ("PRCC", "PRCC ResNet 0.0.1", "legacy real-only run"),
    "prcc_pretrained_reference": ("PRCC", "PRCC pretrained reference", "historical pretrained benchmark"),
    "prcc_plain_swin": ("PRCC", "PRCC plain Swin", "real-only Swin baseline"),
    "prcc_filtered_syntetic_swin": ("PRCC", "PRCC + filtered synthetic Swin", "three representatives per moment"),
    "generalized_reid_duke_raw": ("Duke", "Generalized Swin on Duke (raw)", "multi-domain real + 50% filtered synthetic training; raw retrieval without re-ranking"),
    "generalized_reid_ltcc_raw": ("LTCC", "Generalized Swin on LTCC (raw)", "multi-domain real + 50% filtered synthetic training; raw retrieval without re-ranking"),
    "generalized_reid_prcc_raw": ("PRCC", "Generalized Swin on PRCC (raw)", "multi-domain real + 50% filtered synthetic training; raw retrieval without re-ranking"),
    "generalized_reid_combined_stress_raw": ("Combined stress", "Generalized Swin combined stress (raw)", "namespaced combined official splits; raw retrieval without re-ranking"),
    "uliri_resnet_0.0.1": ("ULIRI", "ULIRI ResNet 0.0.1", "legacy real-only run"),
    "uliri_resnet_0.0.1_current_split_epoch13": ("ULIRI", "ULIRI ResNet 0.0.1 epoch 13 on current split", "current local split differs from historical evaluation split"),
    "uliri_resnet_0.0.2": ("ULIRI", "ULIRI ResNet 0.0.2", "legacy interrupted run"),
    "uliri_resnet_0.1.1_status_only": ("ULIRI", "ULIRI ResNet 0.1.1 status-only", "training status evaluation metric; checkpoint absent locally"),
    "uliri_pretrained_reference": ("ULIRI", "ULIRI pretrained reference", "historical pretrained benchmark"),
    "ccvid_status_only": ("CCVID", "CCVID status-only history", "checkpoint files are absent locally"),
    "ltcc_swin_resume_failed": ("LTCC", "LTCC Swin resume failed", "architecture mismatch before training"),
    "ltcc_resnet_resume_failed": ("LTCC", "LTCC ResNet resume failed", "classifier mismatch before training"),
    "ltcc_sweep_smoke": ("LTCC", "LTCC percentage sweep smoke", "short smoke run"),
}

GENERALIZED_TRAIN_DIR = "experiments/reid/generalized_reid_swin/results/generalized_swin/train"
GENERALIZED_SERIES = {
    "duke": "generalized_reid_duke_raw",
    "ltcc": "generalized_reid_ltcc_raw",
    "prcc": "generalized_reid_prcc_raw",
    "combined_stress": "generalized_reid_combined_stress_raw",
}

SYNTHETIC_ONLY_30K_TRAIN_DIR = "experiments/reid/syntetic_only_filtered_30k/results/syntetic_only_filtered_30k/train"
SYNTHETIC_ONLY_30K_SERIES = {
    "duke": "syntetic_only_filtered_30k_duke_raw",
    "ltcc": "syntetic_only_filtered_30k_ltcc_raw",
    "prcc": "syntetic_only_filtered_30k_prcc_raw",
}
SYNTHETIC_ONLY_100K_TRAIN_DIR = "experiments/reid/syntetic_only_filtered_100k/results/syntetic_only_filtered_100k/train"
SYNTHETIC_ONLY_100K_SERIES = {
    "duke": "syntetic_only_filtered_100k_duke_raw",
    "ltcc": "syntetic_only_filtered_100k_ltcc_raw",
    "prcc": "syntetic_only_filtered_100k_prcc_raw",
}
PRETRAINED_CROSS_DATASET_SERIES = {
    "ccvid": "pretrained_swin_ccvid_raw",
    "duke": "pretrained_swin_duke_raw",
    "ltcc": "pretrained_swin_ltcc_raw",
    "prcc": "pretrained_swin_prcc_raw",
    "uliri": "pretrained_swin_uliri_raw",
    "synthetic_market1501": "pretrained_swin_synthetic_market1501_raw",
}

TRAIN_DIR_SERIES = {
    "experiments/reid/ltcc/results_0.1.3/train": "ltcc_resnet_0.1.3",
    "experiments/reid/ltcc/results_0.1.4/train": "ltcc_resnet_0.1.4",
    "experiments/reid/ltcc/results_1.0.1/train": "ltcc_swin_1.0.1",
    "experiments/reid/ltcc/results_swin_plain/train": "ltcc_swin_plain",
    "experiments/reid/ltcc+syntetic/results_0.1.4_syntetic/train": "ltcc_swin_resume_failed",
    "experiments/reid/ltcc+syntetic/results_0.1.4_syntetic_resnet/train": "ltcc_resnet_resume_failed",
    "experiments/reid/ltcc+syntetic/results_0.1.4_syntetic_resnet_transfer/train": "ltcc_resnet_synthetic_transfer",
    "experiments/reid/ltcc+syntetic/results_swin_combined/train": "ltcc_swin_unfiltered_synthetic",
    "experiments/reid/ltcc_syntetic_filtered_seq/results/ltcc_filtered_syntetic/train": "ltcc_filtered_synthetic_3_variants",
    "experiments/reid/ltcc_syntetic_sweep/results/ltcc_syntetic_10/train": "ltcc_syntetic_10",
    "experiments/reid/ltcc_syntetic_sweep/results/ltcc_syntetic_10_smoke_20260522_1845/train": "ltcc_sweep_smoke",
    "experiments/reid/ltcc_syntetic_sweep/results/ltcc_syntetic_25/train": "ltcc_syntetic_25",
    "experiments/reid/ltcc_syntetic_sweep/results/ltcc_syntetic_50/train": "ltcc_syntetic_50",
    "experiments/reid/ltcc_syntetic_sweep/results/syntetic_only_100_bs48_gpu0_detached/train": "syntetic_only_100",
    "experiments/reid/duke/results_plain/train": "duke_swin_plain_local_partial",
    "experiments/reid/duke+syntetic/results_swin_working_lowbatch/train": "duke_swin_unfiltered_synthetic",
    "experiments/reid/duke_syntetic_filtered_seq/results/duke_filtered_syntetic/train": "duke_filtered_synthetic_3_variants",
    "experiments/reid/prcc/results_0.0.1/train": "prcc_resnet_0.0.1",
    "experiments/reid/prcc_syntetic_filtered_seq/results/prcc_plain_swin/train": "prcc_plain_swin",
    "experiments/reid/prcc_syntetic_filtered_seq/results/prcc_filtered_syntetic_swin/train": "prcc_filtered_syntetic_swin",
    "experiments/reid/uliri/results_0.0.1/train": "uliri_resnet_0.0.1",
    "experiments/reid/uliri/results_0.0.2/train": "uliri_resnet_0.0.2",
}


def relative(path: Path | str) -> str:
    value = Path(path)
    try:
        return str(value.relative_to(ROOT))
    except ValueError:
        return str(value)


def checkpoint_parts(name: str) -> tuple[int | None, int | None]:
    match = CHECKPOINT_RE.search(name)
    if not match:
        return None, None
    step = match.group("step")
    return int(match.group("epoch")), int(step) if step is not None else None


def percent(value: str | float | int | None, fraction: bool = False) -> float | None:
    if value is None or value == "" or value == "NA":
        return None
    if isinstance(value, str):
        clean = value.strip().rstrip("%")
        if not clean or clean == "NA":
            return None
        number = float(clean)
        return number if value.strip().endswith("%") else number * (100 if fraction else 1)
    number = float(value)
    return number * (100 if fraction else 1)


def make_row(
    series: str,
    checkpoint: str,
    map_value: str | float | int | None,
    rank1: str | float | int | None,
    rank5: str | float | int | None,
    rank10: str | float | int | None,
    status: str,
    source: Path | str,
    *,
    fraction: bool = False,
) -> MetricRow:
    epoch, step = checkpoint_parts(checkpoint)
    return MetricRow(
        series,
        checkpoint,
        epoch,
        step,
        percent(map_value, fraction),
        percent(rank1, fraction),
        percent(rank5, fraction),
        percent(rank10, fraction),
        status,
        relative(source),
    )


def read_simple_tsv(series: str, path: Path) -> list[MetricRow]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append(
                make_row(
                    series,
                    row["checkpoint"],
                    row.get("mAP"),
                    row.get("Rank-1"),
                    row.get("Rank-5"),
                    row.get("Rank-10"),
                    row.get("status", "passed"),
                    path,
                )
            )
    return rows


def read_sweep_tsv(path: Path) -> list[MetricRow]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            series = row["experiment"]
            if series == "uliri_resnet_0.0.1" and row["checkpoint"] == "model_epoch_013_step_38333":
                series = "uliri_resnet_0.0.1_current_split_epoch13"
            rows.append(
                make_row(
                    series,
                    row["checkpoint"],
                    row.get("mAP"),
                    row.get("Rank-1"),
                    row.get("Rank-5"),
                    row.get("Rank-10"),
                    row.get("status", "passed"),
                    path,
                )
            )
    return rows


def read_generalized_tsv(path: Path) -> list[MetricRow]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            series = GENERALIZED_SERIES.get(row["target"])
            if not series:
                continue
            rows.append(
                make_row(
                    series,
                    row["checkpoint"],
                    row.get("mAP"),
                    row.get("Rank-1"),
                    row.get("Rank-5"),
                    row.get("Rank-10"),
                    row.get("status", "passed"),
                    path,
                )
            )
    return rows


def read_target_mapped_tsv(path: Path, target_series: dict[str, str]) -> list[MetricRow]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            series = target_series.get(row["target"])
            if not series:
                continue
            rows.append(
                make_row(
                    series,
                    row["checkpoint"],
                    row.get("mAP"),
                    row.get("Rank-1"),
                    row.get("Rank-5"),
                    row.get("Rank-10"),
                    row.get("status", "passed"),
                    path,
                )
            )
    return rows


def read_json_summary(series: str, path: Path) -> list[MetricRow]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for checkpoint, metrics in payload.items():
        if checkpoint == "reid_model_latest":
            continue
        cmc = metrics.get("cmc", {})
        rows.append(
            make_row(
                series,
                checkpoint,
                metrics.get("mAP"),
                cmc.get("Rank-1"),
                cmc.get("Rank-5"),
                cmc.get("Rank-10"),
                "passed",
                path,
                fraction=True,
            )
        )
    return rows


def read_status_last_metric(series: str, checkpoint: str, path: Path) -> list[MetricRow]:
    if not path.is_file():
        return []
    last = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("message") in {"Test metrics generated.", "Eval metrics generated."}:
                last = row.get("kpi", {})
    if last is None:
        return []
    return [
        make_row(
            series,
            checkpoint,
            last.get("mAP"),
            last.get("cmc_rank_1"),
            last.get("cmc_rank_5"),
            last.get("cmc_rank_10"),
            "passed",
            path,
            fraction=True,
        )
    ]


def read_archive_metrics(series: str, path: Path) -> list[MetricRow]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for metrics in payload.get("all_epochs", []):
        epoch = int(metrics["epoch"])
        rows.append(
            make_row(
                series,
                f"archived_epoch_{epoch:03d}",
                metrics.get("mAP"),
                metrics.get("rank1"),
                metrics.get("rank5"),
                metrics.get("rank10"),
                "passed",
                path,
            )
        )
    return rows


def read_metric_text(series: str, checkpoint: str, path: Path) -> list[MetricRow]:
    if not path.is_file():
        return []
    metrics = dict(METRIC_RE.findall(path.read_text(encoding="utf-8", errors="replace")))
    if "mAP" not in metrics:
        return []
    return [
        make_row(
            series,
            checkpoint,
            metrics.get("mAP"),
            metrics.get("Rank-1"),
            metrics.get("Rank-5"),
            metrics.get("Rank-10"),
            "passed",
            path,
        )
    ]


def deduplicate(rows: Iterable[MetricRow]) -> list[MetricRow]:
    selected: dict[tuple[str, str], MetricRow] = {}
    for row in rows:
        key = (row.series, row.checkpoint)
        prior = selected.get(key)
        if prior is None or (prior.status != "passed" and row.status == "passed"):
            selected[key] = row
    return sorted(
        selected.values(),
        key=lambda row: (row.series, row.epoch if row.epoch is not None else 10**9, row.checkpoint),
    )


def collect_metrics() -> list[MetricRow]:
    rows: list[MetricRow] = []
    for version in ("0.1.1", "0.1.2", "0.1.3", "0.1.4", "1.0.1"):
        series = "ltcc_swin_1.0.1" if version == "1.0.1" else f"ltcc_resnet_{version}"
        rows.extend(
            read_json_summary(
                series,
                REID / f"ltcc/evaluation_results_{version}/evaluation_summary.json",
            )
        )
    rows.extend(read_json_summary("prcc_resnet_0.0.1", REID / "prcc/evaluation_results_0.0.1/evaluation_summary.json"))
    rows.extend(read_json_summary("uliri_resnet_0.0.1", REID / "uliri/evaluation_results_0.0.1/evaluation_summary.json"))
    rows.extend(read_json_summary("uliri_resnet_0.0.2", REID / "uliri/evaluation_results_0.0.2/evaluation_summary.json"))
    rows.extend(
        read_status_last_metric(
            "uliri_resnet_0.1.1_status_only",
            "training_status_latest",
            REID / "uliri/results_0.1.1/train/status.json",
        )
    )
    rows.extend(read_archive_metrics("ltcc_archive_v1.0.0", REID / "ltcc/archives/v1.0.0_20251120_083952/results/metrics.json"))
    rows.extend(read_archive_metrics("ltcc_archive_v1.0.1", REID / "ltcc/archives/v1.0.1_20251120_104004/results/metrics.json"))
    rows.extend(read_metric_text("ltcc_pretrained_reference", "pretrained_reference", REID / "ltcc/pretrained_results.txt"))
    rows.extend(read_metric_text("prcc_pretrained_reference", "pretrained_reference", REID / "prcc/pretrained_results.txt"))
    rows.extend(read_metric_text("uliri_pretrained_reference", "pretrained_reference", REID / "uliri/pretrained_results.txt"))
    rows.extend(read_simple_tsv("ltcc_swin_plain", REID / "ltcc/evaluation_results_swin_plain/summary.tsv"))
    rows.extend(read_simple_tsv("ltcc_swin_unfiltered_synthetic", REID / "ltcc+syntetic/evaluation_results_swin_combined/summary.tsv"))
    rows.extend(read_simple_tsv("ltcc_resnet_synthetic_transfer", REID / "ltcc+syntetic/evaluation_results_transfer/summary.tsv"))
    rows.extend(read_simple_tsv("ltcc_filtered_synthetic_3_variants", REID / "ltcc_syntetic_filtered_seq/evaluate/ltcc_filtered_syntetic/summary.tsv"))
    rows.extend(read_sweep_tsv(REID / "ltcc_syntetic_sweep/evaluation_full_gpu1/summary.tsv"))
    rows.extend(read_sweep_tsv(REID / "ltcc_syntetic_sweep/evaluation_progress_gpu1/summary.tsv"))
    for path in sorted((REID / "repository_training_results/additional_validation").glob("summary_worker*.tsv")):
        rows.extend(read_sweep_tsv(path))
    rows.extend(read_simple_tsv("duke_swin_plain_external", REID / "duke/evaluation_results_swin_working_plain/summary.tsv"))
    rows.extend(read_simple_tsv("duke_swin_unfiltered_synthetic", REID / "duke+syntetic/evaluation_results_swin_working_lowbatch/summary.tsv"))
    rows.extend(read_simple_tsv("duke_filtered_synthetic_3_variants", REID / "duke_syntetic_filtered_seq/evaluate/duke_filtered_syntetic/summary_reverse.tsv"))
    rows.extend(read_sweep_tsv(REID / "prcc_syntetic_filtered_seq/evaluate/prcc_real_split/summary_reverse.tsv"))
    rows.extend(read_generalized_tsv(REID / "generalized_reid_swin/evaluate/all_checkpoints/summary.tsv"))
    rows.extend(read_target_mapped_tsv(REID / "syntetic_only_filtered_30k/evaluate/all_targets_raw/summary.tsv", SYNTHETIC_ONLY_30K_SERIES))
    rows.extend(read_target_mapped_tsv(REID / "syntetic_only_filtered_100k/evaluate/all_targets_raw/summary.tsv", SYNTHETIC_ONLY_100K_SERIES))
    rows.extend(read_target_mapped_tsv(REID / "pretrained_cross_dataset/evaluate/all_targets_raw/summary.tsv", PRETRAINED_CROSS_DATASET_SERIES))
    rows.extend(read_status_last_metric("ccvid_status_only", "model_epoch_029_step_50729", REID / "ccvid/results/evaluate/status.json"))
    return deduplicate(rows)


def collect_inventory(metrics: list[MetricRow]) -> list[InventoryRow]:
    metric_status = {(row.series, row.checkpoint): row.status for row in metrics}
    rows = []
    for path in sorted(REID.glob("**/train/model_epoch_*.pth")):
        if path.name.startswith("sanitized_"):
            continue
        train_dir = relative(path.parent)
        if "/tao_temp_output/" in str(path):
            continue
        series_names = [TRAIN_DIR_SERIES.get(train_dir, f"unclassified:{train_dir}")]
        if train_dir == GENERALIZED_TRAIN_DIR:
            series_names = list(GENERALIZED_SERIES.values())
        if train_dir == SYNTHETIC_ONLY_30K_TRAIN_DIR:
            series_names = list(SYNTHETIC_ONLY_30K_SERIES.values())
        if train_dir == SYNTHETIC_ONLY_100K_TRAIN_DIR:
            series_names = list(SYNTHETIC_ONLY_100K_SERIES.values())
        if relative(path) == "experiments/reid/ltcc/results_0.1.4/train/model_epoch_000_step_00000.pth":
            series_names = ["ltcc_swin_misfiled_0.1.4_epoch0"]
        if relative(path) == "experiments/reid/uliri/results_0.0.1/train/model_epoch_013_step_38333.pth":
            series_names = ["uliri_resnet_0.0.1_current_split_epoch13"]
        checkpoint = path.stem
        epoch, step = checkpoint_parts(checkpoint)
        for series in series_names:
            rows.append(
                InventoryRow(
                    series,
                    checkpoint,
                    epoch,
                    step,
                    relative(path),
                    round(path.stat().st_size / 1024 / 1024, 1),
                    metric_status.get((series, checkpoint), "missing"),
                )
            )
    return rows


def display_name(series: str) -> str:
    return SERIES_META.get(series, ("Unknown", series, ""))[1]


def domain(series: str) -> str:
    return SERIES_META.get(series, ("Unknown", series, ""))[0]


def fmt_metric(value: float | None) -> str:
    return "NA" if value is None else f"{value:.1f}%"


def metric_value(row: MetricRow, metric: str) -> float | None:
    return {
        "mAP": row.map,
        "Rank-1": row.rank1,
        "Rank-5": row.rank5,
        "Rank-10": row.rank10,
    }[metric]


def best_row(rows: Iterable[MetricRow]) -> MetricRow | None:
    valid = [row for row in rows if row.status == "passed" and row.map is not None]
    return max(valid, key=lambda row: row.map) if valid else None


def write_csv(path: Path, header: list[str], rows: Iterable[Iterable[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def plot_lines(
    path: Path,
    title: str,
    metric: str,
    series_names: list[str],
    metrics_by_series: dict[str, list[MetricRow]],
    *,
    ylabel: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    plotted = False
    for series in series_names:
        rows = [
            row
            for row in metrics_by_series.get(series, [])
            if row.status == "passed" and row.epoch is not None and metric_value(row, metric) is not None
        ]
        rows.sort(key=lambda row: row.epoch)
        if not rows:
            continue
        ax.plot(
            [row.epoch for row in rows],
            [metric_value(row, metric) for row in rows],
            marker="o",
            markersize=4,
            linewidth=1.7,
            label=display_name(series),
        )
        plotted = True
    ax.set_title(title)
    ax.set_xlabel("Checkpoint epoch")
    ax.set_ylabel(ylabel or f"{metric} (%)")
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_coverage(path: Path, inventory: list[InventoryRow]) -> None:
    grouped: dict[str, list[InventoryRow]] = defaultdict(list)
    for row in inventory:
        grouped[row.series].append(row)
    series = sorted(grouped, key=lambda item: (domain(item), display_name(item)))
    validated = [sum(row.evaluation_status == "passed" for row in grouped[item]) for item in series]
    missing = [sum(row.evaluation_status != "passed" for row in grouped[item]) for item in series]
    fig_height = max(7, len(series) * 0.36)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    y = list(range(len(series)))
    ax.barh(y, validated, label="Validated", color="#2b8a3e")
    ax.barh(y, missing, left=validated, label="Missing or failed", color="#c92a2a")
    ax.set_yticks(y, [display_name(item) for item in series], fontsize=8)
    ax.set_xlabel("Local stable checkpoint count")
    ax.set_title("Local checkpoint evaluation coverage")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def read_dataset_summary(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def status_tail(path: Path) -> tuple[str, str]:
    if not path.is_file():
        return "missing", "No status file"
    last = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    if not last:
        return "unknown", "No parseable status row"
    message = str(last.get("message", "")).replace("\r", "").replace("\n", "<br>")
    if message.endswith("finished successfully."):
        return "SUCCESS", message
    return str(last.get("status", "unknown")), message


def metric_log_count() -> int:
    return sum(
        bool(METRIC_RE.search(path.read_text(encoding="utf-8", errors="replace")))
        for path in REID.rglob("*.log")
    )


def markdown_table(headers: list[str], rows: Iterable[Iterable[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def merge_checkpoint_rows(series: str, metrics: list[MetricRow], inventory: list[InventoryRow]) -> list[tuple[object, ...]]:
    eval_rows = {row.checkpoint: row for row in metrics if row.series == series}
    inv_rows = {row.checkpoint: row for row in inventory if row.series == series}
    checkpoints = sorted(
        set(eval_rows) | set(inv_rows),
        key=lambda checkpoint: (
            checkpoint_parts(checkpoint)[0] if checkpoint_parts(checkpoint)[0] is not None else 10**9,
            checkpoint,
        ),
    )
    rows = []
    for checkpoint in checkpoints:
        metric = eval_rows.get(checkpoint)
        inv = inv_rows.get(checkpoint)
        rows.append(
            (
                checkpoint_parts(checkpoint)[0] if checkpoint_parts(checkpoint)[0] is not None else "NA",
                checkpoint,
                fmt_metric(metric.map) if metric else "NA",
                fmt_metric(metric.rank1) if metric else "NA",
                fmt_metric(metric.rank5) if metric else "NA",
                fmt_metric(metric.rank10) if metric else "NA",
                metric.status if metric else "missing",
                inv.path if inv else "checkpoint not stored locally",
            )
        )
    return rows


def write_report(metrics: list[MetricRow], inventory: list[InventoryRow]) -> None:
    metrics_by_series: dict[str, list[MetricRow]] = defaultdict(list)
    inventory_by_series: dict[str, list[InventoryRow]] = defaultdict(list)
    for row in metrics:
        metrics_by_series[row.series].append(row)
    for row in inventory:
        inventory_by_series[row.series].append(row)

    all_series = sorted(set(metrics_by_series) | set(inventory_by_series), key=lambda item: (domain(item), display_name(item)))
    coverage_rows = []
    for series in all_series:
        local = inventory_by_series.get(series, [])
        passed = sum(row.evaluation_status == "passed" for row in local)
        failed = sum(row.evaluation_status == "failed" for row in local)
        missing = len(local) - passed - failed
        external = len(metrics_by_series.get(series, [])) - sum(
            row.checkpoint in {item.checkpoint for item in local} for row in metrics_by_series.get(series, [])
        )
        best = best_row(metrics_by_series.get(series, []))
        coverage_rows.append(
            (
                domain(series),
                display_name(series),
                len(local),
                passed,
                failed,
                missing,
                external,
                best.checkpoint if best else "NA",
                fmt_metric(best.map) if best else "NA",
            )
        )

    ltcc_filtered = read_dataset_summary(REID / "ltcc_syntetic_filtered_seq/dataset_summary.tsv")
    duke_filtered = read_dataset_summary(REID / "duke_syntetic_filtered_seq/dataset_summary.tsv")
    prcc_filtered = read_dataset_summary(REID / "prcc_syntetic_filtered_seq/dataset_summary.tsv")
    generalized_summary = read_dataset_summary(REID / "generalized_reid_swin/dataset_summary.tsv")
    generalized_partitions = read_dataset_summary(REID / "generalized_reid_swin/manifests/real_partitions.tsv")
    sweep_manifest = read_dataset_summary(REID / "ltcc_syntetic_sweep/sweep_manifest.tsv")
    status_rows = []
    for label, path in [
        ("CCVID training", REID / "ccvid/results/train/status.json"),
        ("CCVID evaluation", REID / "ccvid/results/evaluate/status.json"),
        ("LTCC filtered synthetic", REID / "ltcc_syntetic_filtered_seq/results/ltcc_filtered_syntetic/train/status.json"),
        ("Duke filtered synthetic", REID / "duke_syntetic_filtered_seq/results/duke_filtered_syntetic/train/status.json"),
        ("PRCC plain Swin", REID / "prcc_syntetic_filtered_seq/results/prcc_plain_swin/train/status.json"),
        ("PRCC filtered synthetic Swin", REID / "prcc_syntetic_filtered_seq/results/prcc_filtered_syntetic_swin/train/status.json"),
        ("Generalized multi-domain Swin", REID / "generalized_reid_swin/results/generalized_swin/train/status.json"),
        ("LTCC sweep 10%", REID / "ltcc_syntetic_sweep/results/ltcc_syntetic_10/train/status.json"),
        ("LTCC sweep 25%", REID / "ltcc_syntetic_sweep/results/ltcc_syntetic_25/train/status.json"),
        ("LTCC sweep 50%", REID / "ltcc_syntetic_sweep/results/ltcc_syntetic_50/train/status.json"),
        ("LTCC sweep 75%", REID / "ltcc_syntetic_sweep/results/ltcc_syntetic_75/train/status.json"),
        ("LTCC sweep 100%", REID / "ltcc_syntetic_sweep/results/ltcc_syntetic_100/train/status.json"),
        ("Synthetic only", REID / "ltcc_syntetic_sweep/results/syntetic_only_100_bs48_gpu0_detached/train/status.json"),
        ("Synthetic-only filtered 30k", REID / "syntetic_only_filtered_30k/results/syntetic_only_filtered_30k/train/status.json"),
        ("Synthetic-only filtered 100k", REID / "syntetic_only_filtered_100k/results/syntetic_only_filtered_100k/train/status.json"),
        ("ULIRI 0.0.2", REID / "uliri/results_0.0.2/train/status.json"),
    ]:
        status, message = status_tail(path)
        status_rows.append((label, status, message, relative(path)))

    ltcc_filtered_best = best_row(metrics_by_series.get("ltcc_filtered_synthetic_3_variants", []))
    ltcc_plain_best = best_row(metrics_by_series.get("ltcc_swin_plain", []))
    ltcc_unfiltered_best = best_row(metrics_by_series.get("ltcc_swin_unfiltered_synthetic", []))
    duke_filtered_best = best_row(metrics_by_series.get("duke_filtered_synthetic_3_variants", []))
    duke_plain_best = best_row(metrics_by_series.get("duke_swin_plain_external", []))
    prcc_plain_best = best_row(metrics_by_series.get("prcc_plain_swin", []))
    prcc_filtered_best = best_row(metrics_by_series.get("prcc_filtered_syntetic_swin", []))
    generalized_final = {
        target: max(
            (
                row
                for row in metrics_by_series.get(series, [])
                if row.status == "passed" and row.epoch is not None
            ),
            key=lambda row: row.epoch,
            default=None,
        )
        for target, series in GENERALIZED_SERIES.items()
    }
    generalized_standard_rows = [generalized_final[target] for target in ("duke", "ltcc", "prcc") if generalized_final[target]]
    generalized_macro_map = (
        sum(row.map for row in generalized_standard_rows if row.map is not None) / len(generalized_standard_rows)
        if len(generalized_standard_rows) == 3
        else None
    )
    filtered_stats = {row["split"]: row for row in ltcc_filtered}
    original_synthetic = int(filtered_stats.get("syntetic_original_train", {}).get("images", 0))
    retained_synthetic = int(filtered_stats.get("syntetic_filtered_train", {}).get("images", 0))
    retained_percent = retained_synthetic / original_synthetic * 100 if original_synthetic else 0.0
    audited_metric_logs = metric_log_count()

    lines = [
        "# Repository-Wide TAO ReIdentification Training Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Scope",
        "",
        "This document inventories the TAO ReIdentification experiment artifacts under `experiments/reid`. "
        "It reports stable checkpoint evaluations, identifies missing or failed validations, preserves source-file traceability, "
        "and plots every available checkpoint metric as a line-series point. Metrics are grouped by target dataset because LTCC, "
        "Duke, PRCC, ULIRI, and CCVID have different query/gallery splits and should not be compared as if they were one benchmark. "
        "The generalized-model section adds separate raw-metric curves for Duke, LTCC, PRCC, and an explicitly labeled combined stress split.",
        "",
        "## Evaluation Integrity Policy",
        "",
        "- Synthetic images are training augmentation only for LTCC, Duke, and PRCC synthetic-mix experiments.",
        "- LTCC synthetic experiments are evaluated only on LTCC `query` and LTCC `bounding_box_test`.",
        "- Duke synthetic experiments are evaluated only on Duke `query` and Duke `bounding_box_test`.",
        "- PRCC synthetic experiments are evaluated only on PRCC `query` and PRCC `bounding_box_test`.",
        "- Generalized-model curves are raw retrieval metrics: TAO CPU-side re-ranking and sampled match-grid generation are disabled so every checkpoint can be evaluated efficiently.",
        "- The generalized combined stress split namespaces identity and camera IDs across Duke, LTCC, and PRCC. It is an additional stress measurement, not a replacement for each official target-domain benchmark.",
        "- Checkpoint plots use stable `model_epoch_*.pth` artifacts. Mutable `reid_model_latest.pth` files are excluded.",
        "- A row marked `missing` means a local stable checkpoint exists but no matching comparable external evaluation was found.",
        "- A row marked `failed` means evaluation was attempted but did not produce a valid metric row.",
            "- Historical evaluation JSON rows can appear without a local checkpoint when an earlier checkpoint file was removed after evaluation.",
            "- Archived per-epoch metrics and pretrained benchmark text files are retained as historical-only rows with their original source paths.",
        "",
        "## Executive Findings",
        "",
    ]
    if ltcc_filtered_best and ltcc_plain_best:
        lines.append(
            f"- The best filtered LTCC synthetic checkpoint currently reaches **{fmt_metric(ltcc_filtered_best.map)} mAP** "
            f"and **{fmt_metric(ltcc_filtered_best.rank1)} Rank-1**, compared with the plain LTCC Swin best of "
            f"**{fmt_metric(ltcc_plain_best.map)} mAP** and **{fmt_metric(ltcc_plain_best.rank1)} Rank-1**. "
            f"That is a **{ltcc_filtered_best.map - ltcc_plain_best.map:+.1f} point mAP** change."
        )
    if ltcc_unfiltered_best:
        if ltcc_filtered_best and ltcc_filtered_best.map > ltcc_unfiltered_best.map:
            lines.append(
                f"- The older unfiltered LTCC synthetic mix peaks at **{fmt_metric(ltcc_unfiltered_best.map)} mAP** "
                f"and **{fmt_metric(ltcc_unfiltered_best.rank1)} Rank-1**. The filtered run is stronger by "
                f"**{ltcc_filtered_best.map - ltcc_unfiltered_best.map:+.1f} mAP points** while using a much smaller, "
                "less repetitive synthetic subset."
            )
        else:
            lines.append(
                f"- The older unfiltered LTCC synthetic mix currently has the strongest LTCC Swin checkpoint at "
                f"**{fmt_metric(ltcc_unfiltered_best.map)} mAP** and **{fmt_metric(ltcc_unfiltered_best.rank1)} Rank-1**. "
                "Its data policy is much larger and more repetitive, so the filtered run is the cleaner efficiency comparison."
            )
    if duke_filtered_best and duke_plain_best:
        lines.append(
            f"- The best filtered Duke synthetic checkpoint reaches **{fmt_metric(duke_filtered_best.map)} mAP** "
            f"and **{fmt_metric(duke_filtered_best.rank1)} Rank-1**. The transferred plain Duke final remains stronger at "
            f"**{fmt_metric(duke_plain_best.map)} mAP** and **{fmt_metric(duke_plain_best.rank1)} Rank-1**."
        )
    if prcc_filtered_best and prcc_plain_best:
        lines.append(
            f"- The best filtered PRCC synthetic checkpoint reaches **{fmt_metric(prcc_filtered_best.map)} mAP** "
            f"and **{fmt_metric(prcc_filtered_best.rank1)} Rank-1**, compared with the plain PRCC Swin best of "
            f"**{fmt_metric(prcc_plain_best.map)} mAP** and **{fmt_metric(prcc_plain_best.rank1)} Rank-1**. "
            f"That is a **{prcc_filtered_best.map - prcc_plain_best.map:+.1f} point mAP** change."
        )
    if generalized_macro_map is not None and generalized_final["combined_stress"]:
        lines.append(
            f"- The generalized Swin final epoch 119 raw retrieval results are Duke **{fmt_metric(generalized_final['duke'].map)} mAP**, "
            f"LTCC **{fmt_metric(generalized_final['ltcc'].map)} mAP**, and PRCC **{fmt_metric(generalized_final['prcc'].map)} mAP**. "
            f"The three-dataset macro mAP is **{generalized_macro_map:.1f}%**. Its namespaced combined stress split reaches "
            f"**{fmt_metric(generalized_final['combined_stress'].map)} mAP**."
        )
    lines.extend(
        [
            f"- The original synthetic training directory contains {original_synthetic:,} crops across 39 IDs. "
            f"The filtered policy keeps {retained_synthetic:,} crops, or **{retained_percent:.2f}%**, by retaining at most "
            "three visual variants for each underlying person-at-moment group.",
            "- The percentage sweep is intentionally documented separately from the filtered policy. The sweep samples a percentage from each synthetic ID, while the filtered policy removes near-duplicate moment variants using manifest metadata.",
            "- CCVID is status-only in this checkout: the status history records evaluation attempts and a final mAP value, but its checkpoint files are not present locally.",
            "",
            "## Historical Artifact Audit",
            "",
            f"- A repository-wide text scan found {audited_metric_logs} TAO evaluation logs containing mAP output. Versioned JSON summaries or TSV summaries cover the established LTCC, Duke, PRCC, ULIRI, filtered-synthetic, and percentage-sweep log families.",
            "- `experiments/reid/ltcc/archives/v1.0.0_20251120_083952/results/metrics.json` contributes a 34-epoch historical LTCC curve.",
            "- `experiments/reid/ltcc/archives/v1.0.1_20251120_104004/results/metrics.json` contributes a 23-epoch historical LTCC curve.",
            "- LTCC, PRCC, and ULIRI `pretrained_results.txt` files contribute historical pretrained benchmark references.",
            "- `experiments/reid/uliri/results_0.1.1/train/status.json` contributes a status-only ULIRI evaluation reference because no matching stable checkpoint is present locally.",
            "- The remaining unusual metric logs are the three already parsed LTCC percentage-sweep progress logs. No additional unmerged per-checkpoint evaluation-log family was found.",
            "",
            "## Graphs",
            "",
            "### LTCC Swin mAP",
            "",
            "![LTCC Swin mAP](graphs/ltcc_swin_map.png)",
            "",
            "### LTCC Swin Rank-1",
            "",
            "![LTCC Swin Rank-1](graphs/ltcc_swin_rank1.png)",
            "",
            "### LTCC Legacy And Transfer mAP",
            "",
            "![LTCC legacy mAP](graphs/ltcc_legacy_map.png)",
            "",
            "### Duke mAP",
            "",
            "![Duke mAP](graphs/duke_map.png)",
            "",
            "### Duke Rank-1",
            "",
            "![Duke Rank-1](graphs/duke_rank1.png)",
            "",
            "### PRCC And ULIRI Historical mAP",
            "",
            "![PRCC and ULIRI mAP](graphs/prcc_uliri_map.png)",
            "",
            "### PRCC Swin mAP",
            "",
            "![PRCC Swin mAP](graphs/prcc_swin_map.png)",
            "",
            "### PRCC Swin Rank-1",
            "",
            "![PRCC Swin Rank-1](graphs/prcc_swin_rank1.png)",
            "",
            "### Generalized Swin Cross-Domain Raw mAP",
            "",
            "![Generalized Swin cross-domain raw mAP](graphs/generalized_reid_map.png)",
            "",
            "### Generalized Swin Cross-Domain Raw Rank-1",
            "",
            "![Generalized Swin cross-domain raw Rank-1](graphs/generalized_reid_rank1.png)",
            "",
            "### Synthetic-Only Cross-Dataset Raw mAP",
            "",
            "![Synthetic-only cross-dataset raw mAP](graphs/synthetic_only_cross_dataset_map.png)",
            "",
            "### Synthetic-Only Cross-Dataset Raw Rank-1",
            "",
            "![Synthetic-only cross-dataset raw Rank-1](graphs/synthetic_only_cross_dataset_rank1.png)",
            "",
            "### Local Checkpoint Evaluation Coverage",
            "",
            "![Checkpoint evaluation coverage](graphs/checkpoint_coverage.png)",
            "",
            "## Experiment Coverage Summary",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Target", "Experiment", "Local checkpoints", "Passed", "Failed", "Missing", "Historical-only", "Best checkpoint", "Best mAP"],
            coverage_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Generalized Multi-Domain Swin Construction",
            "",
            "The generalized Swin experiment trains one model from namespaced Duke, LTCC, and PRCC real training identities plus a deterministic 50% selection of the filtered synthetic pool. "
            "Real validation identities are held out before training. Official Duke, LTCC, and PRCC query/gallery folders remain untouched for final target-domain evaluation.",
            "",
            "Synthetic, real-training, validation, and combined-stress files use explicit PID and camera namespaces: Duke offsets start at PID `10000` and camera `100`; "
            "LTCC at PID `20000` and camera `200`; PRCC at PID `30000` and camera `300`; synthetic data at PID `40000` and camera `400`. "
            "The combined stress split is therefore collision-free, but it remains an additional cross-domain stress check rather than an official single-dataset protocol.",
            "",
            "### Generalized Dataset Summary",
            "",
        ]
    )
    lines.extend(markdown_table(["Split", "Images", "IDs", "Path"], [(row["split"], row["images"], row["ids"], row["path"]) for row in generalized_summary]))
    lines.extend(["", "### Identity-Disjoint Real Partition", ""])
    lines.extend(
        markdown_table(
            ["Domain", "Source train images", "Source train IDs", "Train images", "Train IDs", "Validation query", "Validation gallery", "Validation IDs"],
            [
                (
                    row["domain"],
                    row["source_train_images"],
                    row["source_train_ids"],
                    row["train_images"],
                    row["train_ids"],
                    row["validation_query_images"],
                    row["validation_gallery_images"],
                    row["validation_ids"],
                )
                for row in generalized_partitions
            ],
        )
    )
    lines.extend(["", "### Generalized Final Raw Retrieval Metrics", ""])
    lines.extend(
        markdown_table(
            ["Target split", "Checkpoint", "mAP", "Rank-1", "Rank-5", "Rank-10"],
            [
                (
                    target,
                    generalized_final[target].checkpoint,
                    fmt_metric(generalized_final[target].map),
                    fmt_metric(generalized_final[target].rank1),
                    fmt_metric(generalized_final[target].rank5),
                    fmt_metric(generalized_final[target].rank10),
                )
                for target in ("duke", "ltcc", "prcc", "combined_stress")
                if generalized_final[target]
            ],
        )
    )
    lines.extend(
        [
            "",
            "The generalized checkpoint curves intentionally use raw retrieval metrics without re-ranking. This makes the 48-checkpoint audit practical while training and evaluation share the workstation. "
            "Compare these curves internally across generalized checkpoints; do not treat them as directly interchangeable with older re-ranked score tables.",
            "",
            "## Filtered Synthetic Data Construction",
            "",
        ]
    )
    lines.extend(
        [
            "The filtered LTCC and Duke experiments use the manifest at `datasets/final_syntetic_market1501/manifest.csv`. "
            "Rows are grouped by `pid`, `camera_id`, `sequence_id`, `frame_id`, and `source_box_index`. "
            "Within each group, the preparation scripts sort by `variant_id`, then keep the three lowest variants. "
            "This directly limits repeated versions of the same visual moment while preserving all 39 synthetic identities.",
            "",
            "Synthetic PIDs are offset before the files are linked into a combined Market-1501-style training directory. "
            "This avoids identity collisions with real LTCC or Duke PIDs. Synthetic query and gallery folders are not used in target-domain validation.",
            "",
            "### LTCC Filtered Dataset",
            "",
        ]
    )
    lines.extend(markdown_table(["Split", "Images", "IDs", "Path"], [(row["split"], row["images"], row["ids"], row["path"]) for row in ltcc_filtered]))
    lines.extend(["", "### Duke Filtered Dataset", ""])
    lines.extend(markdown_table(["Split", "Images", "IDs", "Path"], [(row["split"], row["images"], row["ids"], row["path"]) for row in duke_filtered]))
    lines.extend(["", "### PRCC Filtered Dataset", ""])
    lines.extend(markdown_table(["Split", "Images", "IDs", "Path"], [(row["split"], row["images"], row["ids"], row["path"]) for row in prcc_filtered]))
    lines.extend(
        [
            "",
            "### Original Synthetic Variant Distribution",
            "",
            "| Variants available for one person-at-moment group | Group count |",
            "| --- | --- |",
            "| 1 | 5 |",
            "| 47 | 1 |",
            "| 50 | 734 |",
            "| 149 | 12 |",
            "| 150 | 1302 |",
            "",
            f"The three-variant filter removes **{100.0 - retained_percent:.2f}%** of the original synthetic crop volume while preserving the synthetic identity set.",
            "",
            "## LTCC Percentage Sweep Construction",
            "",
            "The earlier LTCC percentage sweep uses a different policy. It groups synthetic crops by PID and takes the requested percentage from every PID. "
            "This keeps every synthetic identity represented while progressively increasing synthetic image volume. The sweep does not use manifest moment grouping, "
            "so near-duplicate moment variants remain present.",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Experiment", "Synthetic %", "LTCC images", "Synthetic images", "LTCC IDs", "Synthetic IDs"],
            [
                (
                    row["experiment"],
                    row["percent"],
                    row["ltcc_images"],
                    row["synthetic_images"],
                    row["ltcc_ids"],
                    row["synthetic_ids"],
                )
                for row in sweep_manifest
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Hardware And Evaluation Execution",
            "",
            "| Device | Memory | Current report role |",
            "| --- | --- | --- |",
            "| NVIDIA GeForce RTX 3090 | 24,576 MiB | Filtered-LTCC recovery queues and generalized multi-domain Swin training |",
            "| NVIDIA GeForce RTX 5070 | 12,227 MiB | LTCC recovery queues and generalized raw checkpoint validation while GPU 0 trained |",
            "",
            "TAO evaluation runs inside `nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt`. The checkpoint queues use stable checkpoint files and execute on separate GPU-isolated Docker containers. "
            "The report parser deduplicates repeated TSV rows, preferring a passed result when an earlier attempt failed.",
            "",
            "## Training Configuration Notes",
            "",
            "| Experiment family | Backbone | Input | Epochs | Checkpoint interval | Optimizer | Base LR | Notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
            "| LTCC plain Swin | `swin_base_patch4_window7_224` | `384x192` | 150 | 5 | SGD | `0.00035` | Real LTCC train only |",
            "| LTCC filtered synthetic | `swin_base_patch4_window7_224` | `384x192` | 150 | 5 | SGD | `0.00035` | LTCC train plus 6,152 filtered synthetic crops |",
            "| LTCC percentage sweep | `swin_base_patch4_window7_224` | `384x192` | 150 planned | 5 | SGD | `0.00035` | Interrupted at different points; synthetic-only rerun used batch 48 |",
            "| Duke filtered synthetic | `swin_base_patch4_window7_224` | `256x128` | 200 | 10 | SGD | `0.001` | Duke train plus 6,152 filtered synthetic crops |",
            "| Duke older synthetic mix | `swin_base_patch4_window7_224` | `256x128` | 200 | 10 | SGD | transferred config | Older combined-data policy |",
            "| PRCC plain Swin | `swin_base_patch4_window7_224` | `256x128` | 120 | 10 | SGD | `0.0006` | Real PRCC train only |",
            "| PRCC filtered synthetic | `swin_base_patch4_window7_224` | `256x128` | 120 | 10 | SGD | `0.0006` | PRCC train plus 6,152 filtered synthetic crops |",
            "| Generalized multi-domain Swin | `swin_base_patch4_window7_224` | `256x128` | 120 | 10 | SGD | `0.0006` | Namespaced Duke + LTCC + PRCC real train, identity-disjoint real validation, and 116,920 synthetic crops selected from the filtered pool |",
            "| Synthetic-only filtered 30k | `swin_base_patch4_window7_224` | `256x128` | 120 | 10 | SGD | `0.0006` | Exactly 30,000 synthetic crops, no real training images; evaluated on real Duke/LTCC/PRCC raw target splits |",
            "| Synthetic-only filtered 100k | `swin_base_patch4_window7_224` | `256x128` | 120 | 10 | SGD | `0.0006` | Exactly 100,000 synthetic crops using lower filtering; training started on GPU 0, evaluation pending |",
            "",
            "## Run Status And Known Failures",
            "",
        ]
    )
    lines.extend(markdown_table(["Run", "Latest status", "Latest message", "Status file"], status_rows))
    lines.extend(
        [
            "",
            "Notable failure history:",
            "",
            "- `ltcc_syntetic_10` epoch 19 is a truncated checkpoint. Repeated evaluation attempts fail with `EOFError: Ran out of input`, so it remains marked failed.",
            "- `ltcc_syntetic_10` epoch 139 initially failed on the RTX 5070 during evaluation, then passed on the RTX 3090 at `34.0%` mAP. The final tables retain the successful comparable result.",
            "- `ltcc_syntetic_50` epoch 24 stalled for over an hour during CPU-side re-ranking on the RTX 5070. It passed on the RTX 3090 recovery queue with 27.0% mAP, and the remaining sweep evaluations completed under per-checkpoint timeouts.",
            "- LTCC `results_0.1.4/train/model_epoch_000_step_00000.pth` is a Swin artifact stored beside later ResNet checkpoints. It is evaluated and reported as a separate misfiled series.",
            "- ULIRI `results_0.0.1/train/model_epoch_013_step_38333.pth` carries a 92-class classifier head while the current local config declares 80 classes. The recovery evaluator loads it with `dataset.num_classes=92`.",
            "- ULIRI epoch 13 is reported separately from the older ULIRI `0.0.1` curve because the current local split contains 5,355 query and 12,565 gallery images, while the historical evaluation logs used a smaller split. The current-split retry reached 99.8% mAP and 100.0% Rank-1.",
            "- LTCC `results_0.1.4_syntetic` attempted to resume a ResNet checkpoint into a Swin model and failed with state-dict architecture mismatches.",
            "- LTCC `results_0.1.4_syntetic_resnet` then matched the backbone but failed because the checkpoint classifier had 77 classes while the synthetic-only training set had 7 classes.",
            "- CCVID evaluation history contains repeated GPU out-of-memory attempts before the final recorded status-only metric.",
            "- ULIRI `0.0.2` training history contains a GPU out-of-memory failure.",
            "- The generalized checkpoint sweep disables TAO CPU-side re-ranking and sampled match-grid images. Re-ranking made repeated full checkpoint validation CPU-bound, while sampled match-grid generation created one visual row per query and added avoidable overhead.",
            "",
            "## Full Checkpoint Tables",
            "",
            "Every available evaluated checkpoint is listed below. Local checkpoints without a comparable evaluation are included with `missing` status.",
            "",
        ]
    )
    for series in all_series:
        rows = merge_checkpoint_rows(series, metrics, inventory)
        if not rows:
            continue
        lines.extend([f"### {display_name(series)}", ""])
        lines.extend(markdown_table(["Epoch", "Checkpoint", "mAP", "Rank-1", "Rank-5", "Rank-10", "Status", "Local checkpoint path"], rows))
        lines.append("")
    lines.extend(
        [
            "## Generated Machine-Readable Tables",
            "",
            "- `tables/checkpoint_metrics.csv`: deduplicated comparable evaluation metrics.",
            "- `tables/checkpoint_inventory.csv`: every local stable checkpoint and its evaluation coverage state.",
            "- `tables/experiment_summary.csv`: one coverage and best-checkpoint row per experiment series.",
            "",
            "## Reproducibility Files",
            "",
            "- `experiments/reid/ltcc_syntetic_filtered_seq/prepare_filtered_dataset.py`",
            "- `experiments/reid/ltcc_syntetic_filtered_seq/evaluate_all_checkpoints_container.sh`",
            "- `experiments/reid/ltcc_syntetic_sweep/prepare_ltcc_syntetic_sweep.py`",
            "- `experiments/reid/ltcc_syntetic_sweep/evaluate_all_available_container.sh`",
            "- `experiments/reid/duke_syntetic_filtered_seq/prepare_filtered_dataset.py`",
            "- `experiments/reid/duke_syntetic_filtered_seq/evaluate_all_reverse.sh`",
            "- `experiments/reid/prcc_syntetic_filtered_seq/prepare_prcc_experiments.py`",
            "- `experiments/reid/prcc_syntetic_filtered_seq/evaluate_all_reverse.sh`",
            "- `experiments/reid/generalized_reid_swin/prepare_generalized_experiment.py`",
            "- `experiments/reid/generalized_reid_swin/start_train_detached.sh`",
            "- `experiments/reid/generalized_reid_swin/start_evaluate_all_gpu1_detached.sh`",
            "",
            "## Limitations",
            "",
            "- This report reflects artifacts available in the current checkout. Removed checkpoint files cannot be revalidated locally.",
            "- External plain Duke has only its transferred final checkpoint, so it appears as a single reference point rather than a full learning curve.",
            "- CCVID checkpoint files are absent locally; its final status-only mAP is not a full CMC result.",
            "- A metric curve is comparable only within the same target-domain query/gallery protocol.",
            "- Generalized-model curves use raw retrieval metrics without re-ranking. They should not be directly compared with older re-ranked metrics as though the scoring protocol were identical.",
            "- A missing graph point does not mean zero performance. It means no comparable completed evaluation row exists for that checkpoint.",
        ]
    )
    (OUT / "TRAINING_RESULTS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_csv(
        TABLE_DIR / "experiment_summary.csv",
        ["target", "experiment", "local_checkpoints", "passed", "failed", "missing", "historical_only", "best_checkpoint", "best_map"],
        coverage_rows,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    metrics = collect_metrics()
    inventory = collect_inventory(metrics)
    metrics_by_series: dict[str, list[MetricRow]] = defaultdict(list)
    for row in metrics:
        metrics_by_series[row.series].append(row)

    write_csv(
        TABLE_DIR / "checkpoint_metrics.csv",
        ["target", "series", "experiment", "checkpoint", "epoch", "step", "mAP", "Rank-1", "Rank-5", "Rank-10", "status", "source"],
        [
            [
                domain(row.series),
                row.series,
                display_name(row.series),
                row.checkpoint,
                row.epoch if row.epoch is not None else "",
                row.step if row.step is not None else "",
                row.map if row.map is not None else "",
                row.rank1 if row.rank1 is not None else "",
                row.rank5 if row.rank5 is not None else "",
                row.rank10 if row.rank10 is not None else "",
                row.status,
                row.source,
            ]
            for row in metrics
        ],
    )
    write_csv(
        TABLE_DIR / "checkpoint_inventory.csv",
        ["target", "series", "experiment", "checkpoint", "epoch", "step", "size_mib", "evaluation_status", "path"],
        [
            [
                domain(row.series),
                row.series,
                display_name(row.series),
                row.checkpoint,
                row.epoch if row.epoch is not None else "",
                row.step if row.step is not None else "",
                row.size_mib,
                row.evaluation_status,
                row.path,
            ]
            for row in inventory
        ],
    )

    ltcc_swin = [
        "ltcc_swin_plain",
        "ltcc_swin_unfiltered_synthetic",
        "ltcc_filtered_synthetic_3_variants",
        "ltcc_syntetic_10",
        "ltcc_syntetic_25",
        "ltcc_syntetic_50",
        "syntetic_only_100",
    ]
    ltcc_legacy = [
        "ltcc_resnet_0.1.1",
        "ltcc_resnet_0.1.2",
        "ltcc_resnet_0.1.3",
        "ltcc_resnet_0.1.4",
        "ltcc_archive_v1.0.0",
        "ltcc_archive_v1.0.1",
        "ltcc_swin_1.0.1",
        "ltcc_resnet_synthetic_transfer",
    ]
    duke = [
        "duke_swin_plain_external",
        "duke_swin_unfiltered_synthetic",
        "duke_filtered_synthetic_3_variants",
    ]
    prcc = [
        "prcc_plain_swin",
        "prcc_filtered_syntetic_swin",
    ]
    generalized = list(GENERALIZED_SERIES.values())
    synthetic_only_cross_dataset = list(SYNTHETIC_ONLY_30K_SERIES.values()) + list(SYNTHETIC_ONLY_100K_SERIES.values())
    pretrained_cross_dataset = list(PRETRAINED_CROSS_DATASET_SERIES.values())
    plot_lines(GRAPH_DIR / "ltcc_swin_map.png", "LTCC target evaluation: Swin mAP by stable checkpoint", "mAP", ltcc_swin, metrics_by_series)
    plot_lines(GRAPH_DIR / "ltcc_swin_rank1.png", "LTCC target evaluation: Swin Rank-1 by stable checkpoint", "Rank-1", ltcc_swin, metrics_by_series)
    plot_lines(GRAPH_DIR / "ltcc_legacy_map.png", "LTCC target evaluation: legacy and transfer mAP", "mAP", ltcc_legacy, metrics_by_series)
    plot_lines(GRAPH_DIR / "duke_map.png", "Duke target evaluation: mAP by stable checkpoint", "mAP", duke, metrics_by_series)
    plot_lines(GRAPH_DIR / "duke_rank1.png", "Duke target evaluation: Rank-1 by stable checkpoint", "Rank-1", duke, metrics_by_series)
    plot_lines(GRAPH_DIR / "prcc_swin_map.png", "PRCC target evaluation: Swin mAP by stable checkpoint", "mAP", prcc, metrics_by_series)
    plot_lines(GRAPH_DIR / "prcc_swin_rank1.png", "PRCC target evaluation: Swin Rank-1 by stable checkpoint", "Rank-1", prcc, metrics_by_series)
    plot_lines(GRAPH_DIR / "generalized_reid_map.png", "Generalized Swin: cross-domain raw mAP without re-ranking", "mAP", generalized, metrics_by_series)
    plot_lines(GRAPH_DIR / "generalized_reid_rank1.png", "Generalized Swin: cross-domain raw Rank-1 without re-ranking", "Rank-1", generalized, metrics_by_series)
    plot_lines(GRAPH_DIR / "synthetic_only_cross_dataset_map.png", "Synthetic-only filtered 30k/100k: real-dataset raw mAP without re-ranking", "mAP", synthetic_only_cross_dataset, metrics_by_series)
    plot_lines(GRAPH_DIR / "synthetic_only_cross_dataset_rank1.png", "Synthetic-only filtered 30k/100k: real-dataset raw Rank-1 without re-ranking", "Rank-1", synthetic_only_cross_dataset, metrics_by_series)
    plot_lines(GRAPH_DIR / "pretrained_cross_dataset_map.png", "Pretrained Swin: cross-dataset raw mAP without re-ranking", "mAP", pretrained_cross_dataset, metrics_by_series)
    plot_lines(GRAPH_DIR / "pretrained_cross_dataset_rank1.png", "Pretrained Swin: cross-dataset raw Rank-1 without re-ranking", "Rank-1", pretrained_cross_dataset, metrics_by_series)
    plot_lines(
        GRAPH_DIR / "prcc_uliri_map.png",
        "PRCC and ULIRI historical mAP curves (separate target datasets)",
        "mAP",
        [
            "prcc_resnet_0.0.1",
            "uliri_resnet_0.0.1",
            "uliri_resnet_0.0.1_current_split_epoch13",
            "uliri_resnet_0.0.2",
            "uliri_resnet_0.1.1_status_only",
        ],
        metrics_by_series,
    )
    plot_coverage(GRAPH_DIR / "checkpoint_coverage.png", inventory)
    write_report(metrics, inventory)
    print(f"Wrote report: {OUT / 'TRAINING_RESULTS_REPORT.md'}")
    print(f"Wrote graphs: {GRAPH_DIR}")
    print(f"Wrote tables: {TABLE_DIR}")


if __name__ == "__main__":
    main()
