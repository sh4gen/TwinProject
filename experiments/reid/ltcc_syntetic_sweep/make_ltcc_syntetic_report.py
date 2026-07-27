#!/usr/bin/env python3
"""Build a markdown report for the LTCC synthetic sweep."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path("/mnt/2tb_ssd/TwinProject")
EXP = ROOT / "experiments/reid/ltcc_syntetic_sweep"
LTCC = ROOT / "experiments/reid/ltcc/data"
SYNTHETIC = ROOT / "datasets/final_syntetic_market1501"


def pct(value: str) -> float:
    if not value or value == "NA":
        return -1.0
    return float(value.rstrip("%"))


def image_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.rglob("*") if item.suffix.lower() in {".jpg", ".jpeg", ".png"})


def id_count(path: Path) -> int:
    ids = set()
    for item in path.rglob("*"):
        if item.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        match = re.match(r"^(\d+)_", item.name)
        if match:
            ids.add(match.group(1))
    return len(ids)


def read_manifest() -> list[dict[str, str]]:
    path = EXP / "sweep_manifest.tsv"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_summaries() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(EXP.glob("evaluation*_gpu*/summary.tsv")):
        with path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                row["summary_file"] = str(path)
                rows.append(row)
    return rows


def latest_status(path: Path) -> str:
    if not path.is_file():
        return "not started"
    last = ""
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last = line.strip()
        if not last:
            return "empty status"
        data = json.loads(last)
        message = data.get("message", "")
        status = data.get("status", "")
        date = data.get("date", "")
        time = data.get("time", "")
        return f"{status}: {message} ({date} {time})"
    except Exception as exc:  # noqa: BLE001
        return f"unreadable: {exc}"


def best_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_experiment: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "passed":
            by_experiment[row["experiment"]].append(row)

    best = []
    for experiment, experiment_rows in sorted(by_experiment.items()):
        best.append(max(experiment_rows, key=lambda row: pct(row.get("mAP", ""))))
    return best


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EXP / "LTCC_SYNTHETIC_SWEEP_REPORT.md")
    args = parser.parse_args()

    manifest = read_manifest()
    summaries = read_summaries()
    best = best_rows(summaries)
    synthetic_cfg = read_yaml(EXP / "configs/syntetic_only_100.yaml")
    ltcc_cfg = read_yaml(ROOT / "experiments/reid/ltcc/ltcc_swin_plain.yaml")

    lines: list[str] = []
    lines.append("# LTCC Synthetic ReID Sweep")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}".rstrip())
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "We are measuring how synthetic person crops affect an LTCC ReID model. "
        "Training sets mix LTCC train identities with different percentages of the synthetic train split, "
        "but every evaluation uses only the real LTCC query and gallery/test split. "
        "This keeps the metric tied to real LTCC performance and prevents synthetic images from leaking into evaluation."
    )
    lines.append("")
    lines.append("## Dataset Separation")
    lines.append("")
    lines.append(
        "The synthetic dataset was converted to Market-1501-style ReID files before this sweep. "
        "Only `bounding_box_train` from `datasets/final_syntetic_market1501` is used for training. "
        "Synthetic `query` and `bounding_box_test` folders are intentionally ignored in this LTCC experiment."
    )
    lines.append("")
    lines.extend(
        table(
            ["Split", "Path", "Images", "IDs"],
            [
                ["LTCC train", str(LTCC / "bounding_box_train"), str(image_count(LTCC / "bounding_box_train")), str(id_count(LTCC / "bounding_box_train"))],
                ["LTCC query", str(LTCC / "query"), str(image_count(LTCC / "query")), str(id_count(LTCC / "query"))],
                ["LTCC gallery/test", str(LTCC / "bounding_box_test"), str(image_count(LTCC / "bounding_box_test")), str(id_count(LTCC / "bounding_box_test"))],
                ["Synthetic train", str(SYNTHETIC / "bounding_box_train"), str(image_count(SYNTHETIC / "bounding_box_train")), str(id_count(SYNTHETIC / "bounding_box_train"))],
            ],
        )
    )
    lines.append("")
    lines.append("## How The Mixed Train Sets Were Built")
    lines.append("")
    lines.append(
        "`prepare_ltcc_syntetic_sweep.py` groups synthetic images by person ID and takes the requested percentage from each synthetic identity. "
        "That means 10%, 25%, 50%, 75%, and 100% keep all synthetic identities represented instead of randomly dropping complete people. "
        "Synthetic person IDs are offset by `max(LTCC PID) + 1000` before linking into generated train folders, so synthetic IDs cannot collide with LTCC IDs."
    )
    lines.append("")
    if manifest:
        lines.extend(
            table(
                ["Experiment", "Synthetic %", "LTCC images", "Synthetic images", "LTCC IDs", "Synthetic IDs"],
                [
                    [
                        row["experiment"],
                        row["percent"],
                        row["ltcc_images"],
                        row["synthetic_images"],
                        row["ltcc_ids"],
                        row["synthetic_ids"],
                    ]
                    for row in manifest
                ],
            )
        )
        lines.append("")
    lines.append("## Common Training Configuration")
    lines.append("")
    model = synthetic_cfg.get("model", {})
    dataset = synthetic_cfg.get("dataset", {})
    train = synthetic_cfg.get("train", {})
    optim = train.get("optim", {})
    re_ranking = synthetic_cfg.get("re_ranking", {})
    lines.extend(
        [
            f"- Backbone: `{model.get('backbone', ltcc_cfg.get('model', {}).get('backbone'))}`",
            f"- Pretrained file: `{model.get('pretrained_model_path')}`",
            f"- Input size: `{model.get('input_height')}x{model.get('input_width')}`",
            f"- Losses: ID `{model.get('id_loss_type')}` plus metric `{model.get('metric_loss_type')}`",
            f"- Optimizer: `{optim.get('name')}`, base LR `{optim.get('base_lr')}`, momentum `{optim.get('momentum')}`, weight decay `{optim.get('weight_decay')}`",
            f"- Current synthetic-only run: `{train.get('num_epochs')}` epochs, batch `{dataset.get('batch_size')}`, validation batch `{dataset.get('val_batch_size')}`, workers `{dataset.get('num_workers')}`",
            f"- Re-ranking: `{re_ranking.get('re_ranking')}`, k1 `{re_ranking.get('k1')}`, k2 `{re_ranking.get('k2')}`, lambda `{re_ranking.get('lambda_value')}`",
        ]
    )
    lines.append("")
    lines.append("## Evaluation Policy")
    lines.append("")
    lines.append(
        "All checkpoint evaluations override the config evaluation paths to LTCC query and LTCC `bounding_box_test`. "
        "The currently running synthetic-only training is not stopped; GPU0 remains assigned to training and GPU1 is used for evaluation."
    )
    lines.append("")
    lines.append("The pretrained Swin `.tlt` baseline is evaluated on LTCC by overriding `dataset.num_classes=857`, because the checkpoint classifier has 857 rows while LTCC has 77 train IDs. This affects checkpoint loading only; the reported metrics still come from LTCC query/gallery embeddings.")
    lines.append("")
    if summaries:
        lines.append("## Best Evaluated Checkpoint Per Experiment")
        lines.append("")
        lines.extend(
            table(
                ["Experiment", "Best checkpoint", "mAP", "Rank-1", "Rank-5", "Rank-10"],
                [
                    [
                        row["experiment"],
                        row["checkpoint"],
                        row.get("mAP", "NA"),
                        row.get("Rank-1", "NA"),
                        row.get("Rank-5", "NA"),
                        row.get("Rank-10", "NA"),
                    ]
                    for row in best
                ],
            )
        )
        lines.append("")
        lines.append("## All Evaluation Rows")
        lines.append("")
        lines.extend(
            table(
                ["Experiment", "Checkpoint", "mAP", "Rank-1", "Rank-5", "Rank-10", "Status"],
                [
                    [
                        row.get("experiment", ""),
                        row.get("checkpoint", ""),
                        row.get("mAP", "NA"),
                        row.get("Rank-1", "NA"),
                        row.get("Rank-5", "NA"),
                        row.get("Rank-10", "NA"),
                        row.get("status", ""),
                    ]
                    for row in sorted(summaries, key=lambda item: (item.get("experiment", ""), item.get("checkpoint", "")))
                ],
            )
        )
        lines.append("")
    lines.append("## Current Run Status")
    lines.append("")
    status_paths = [
        EXP / "results/syntetic_only_100_bs48_gpu0_detached/train/status.json",
        EXP / "evaluation_full_gpu1/pretrained/status.json",
    ]
    lines.extend(table(["Status file", "Latest status"], [[str(path), latest_status(path)] for path in status_paths]))
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.extend(
        [
            f"- Sweep data builder: `{EXP / 'prepare_ltcc_syntetic_sweep.py'}`",
            f"- Training launcher: `{EXP / 'run_ltcc_syntetic_sweep.sh'}`",
            f"- Synthetic-only GPU0 launcher: `{EXP / 'start_syntetic_only_gpu0.sh'}`",
            f"- Full GPU1 evaluator: `{EXP / 'evaluate_all_available_gpu1.sh'}`",
            f"- Full evaluation summary: `{EXP / 'evaluation_full_gpu1/summary.tsv'}`",
            f"- Earlier progress summary: `{EXP / 'evaluation_progress_gpu1/summary.tsv'}`",
        ]
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- GPU0 synthetic-only training is compute-bound: Swin Base saturates the RTX 3090, while disk and dataloader wait are low.")
    lines.append("- GPU1 is used for evaluation only because previous TAO training on the RTX 5070 was unstable.")
    lines.append("- `reid_model_latest.pth` is not evaluated while training is running because it can be overwritten mid-read; only stable `model_epoch_*.pth` checkpoints are queued.")
    lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
