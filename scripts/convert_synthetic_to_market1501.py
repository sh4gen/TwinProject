#!/usr/bin/env python3
"""Convert synthetic ReID JSON annotations into Market-1501 cropped images."""

from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image


FILENAME_RE = re.compile(
    r"^(?P<pid>\d+)_c(?P<camera>\d+)s(?P<sequence>\d+)_(?P<frame>\d+)_(?P<variant>\d+)\.jpg$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop synthetic person boxes into a Market-1501 style dataset."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("final_syntetic_dataset") / "SYNTHETIC DATAS",
        help="Directory containing synthetic scene folders and JSON annotations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("final_syntetic_market1501"),
        help="Market-1501 output dataset root.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker processes used for cropping.",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=1,
        help="Minimum valid crop width in pixels.",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=1,
        help="Minimum valid crop height in pixels.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for cropped output images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output crop files.",
    )
    return parser.parse_args()


def character_to_pid(value: Any, fallback: int) -> int:
    if value is None:
        return fallback
    match = re.search(r"\d+", str(value))
    if not match:
        return fallback
    return int(match.group(0))


def filename_info(image_name: str, annotation: dict[str, Any]) -> dict[str, int]:
    match = FILENAME_RE.match(image_name)
    if match:
        return {key: int(value) for key, value in match.groupdict().items()}

    return {
        "pid": int(annotation.get("person_id", 0)),
        "camera": int(annotation.get("camera_id", 0)),
        "sequence": int(annotation.get("sequence_id", 0)),
        "frame": int(annotation.get("frame_id", 0)),
        "variant": int(annotation.get("variant_id", 0)),
    }


def clamp_box(box: dict[str, Any], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    x_min = max(0, min(width, int(round(float(box.get("x_min", 0))))))
    y_min = max(0, min(height, int(round(float(box.get("y_min", 0))))))
    x_max = max(0, min(width, int(round(float(box.get("x_max", 0))))))
    y_max = max(0, min(height, int(round(float(box.get("y_max", 0))))))
    return x_min, y_min, x_max, y_max


def process_annotation(task: tuple[str, str, int, int, int, bool]) -> dict[str, Any]:
    json_path_raw, output_train_raw, min_width, min_height, jpeg_quality, overwrite = task
    json_path = Path(json_path_raw)
    output_train = Path(output_train_raw)

    result: dict[str, Any] = {
        "json": str(json_path),
        "processed_images": 0,
        "written": 0,
        "skipped_invalid": 0,
        "skipped_non_person": 0,
        "skipped_missing_image": 0,
        "skipped_existing": 0,
        "errors": [],
        "rows": [],
    }

    try:
        with json_path.open("r", encoding="utf-8") as handle:
            annotation = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"json_read_error: {exc}")
        return result

    image_name = annotation.get("image_file") or f"{json_path.stem}.jpg"
    image_path = json_path.with_name(image_name)
    if not image_path.exists():
        result["skipped_missing_image"] += 1
        result["errors"].append(f"missing_image: {image_path}")
        return result

    boxes = annotation.get("annotations", {}).get("boxes", [])
    if not boxes:
        return result

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_size = image.size
            base_info = filename_info(image_name, annotation)

            for source_box_index, box in enumerate(boxes):
                if box.get("label_info", {}).get("class") != "person":
                    result["skipped_non_person"] += 1
                    continue

                x_min, y_min, x_max, y_max = clamp_box(box, image_size)
                crop_width = x_max - x_min
                crop_height = y_max - y_min
                if crop_width < min_width or crop_height < min_height:
                    result["skipped_invalid"] += 1
                    continue

                pid = character_to_pid(box.get("character_id"), base_info["pid"])
                camera_id = int(annotation.get("camera_id", base_info["camera"]))
                sequence_id = int(annotation.get("sequence_id", base_info["sequence"]))
                frame_id = int(annotation.get("frame_id", base_info["frame"]))
                variant_id = int(annotation.get("variant_id", base_info["variant"]))
                encoded_frame = frame_id * 1000 + variant_id
                output_name = (
                    f"{pid:04d}_c{camera_id}s{sequence_id}_"
                    f"{encoded_frame:06d}_{source_box_index:02d}.jpg"
                )
                output_path = output_train / output_name

                if output_path.exists() and not overwrite:
                    result["skipped_existing"] += 1
                else:
                    crop = image.crop((x_min, y_min, x_max, y_max))
                    crop.save(output_path, quality=jpeg_quality, optimize=True)
                    result["written"] += 1

                result["rows"].append(
                    {
                        "output_file": output_name,
                        "source_image": str(image_path),
                        "source_json": str(json_path),
                        "pid": pid,
                        "character_id": box.get("character_id", ""),
                        "semantic_id": box.get("semantic_id", ""),
                        "camera_id": camera_id,
                        "sequence_id": sequence_id,
                        "frame_id": frame_id,
                        "variant_id": variant_id,
                        "encoded_frame": encoded_frame,
                        "source_box_index": source_box_index,
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                        "width": crop_width,
                        "height": crop_height,
                    }
                )

            result["processed_images"] = 1
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"image_process_error: {exc}")

    return result


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output_train = output / "bounding_box_train"
    output_test = output / "bounding_box_test"
    output_query = output / "query"

    output_train.mkdir(parents=True, exist_ok=True)
    output_test.mkdir(parents=True, exist_ok=True)
    output_query.mkdir(parents=True, exist_ok=True)

    duplicate_name_re = re.compile(r"^(?P<base>.+)\(\d+\)$")
    json_paths = []
    duplicate_annotation_files = 0
    for path in sorted(source.rglob("*.json")):
        if "_meta" in path.stem:
            continue
        duplicate_match = duplicate_name_re.match(path.stem)
        if duplicate_match and path.with_name(f"{duplicate_match.group('base')}.json").exists():
            duplicate_annotation_files += 1
            continue
        json_paths.append(path)
    tasks = [
        (
            str(path),
            str(output_train),
            args.min_width,
            args.min_height,
            args.jpeg_quality,
            args.overwrite,
        )
        for path in json_paths
    ]

    totals = {
        "annotations": len(tasks),
        "duplicate_annotation_files": duplicate_annotation_files,
        "processed_images": 0,
        "written": 0,
        "skipped_invalid": 0,
        "skipped_non_person": 0,
        "skipped_missing_image": 0,
        "skipped_existing": 0,
        "errors": 0,
    }

    manifest_path = output / "manifest.csv"
    summary_path = output / "summary.json"
    error_path = output / "errors.log"
    fieldnames = [
        "output_file",
        "source_image",
        "source_json",
        "pid",
        "character_id",
        "semantic_id",
        "camera_id",
        "sequence_id",
        "frame_id",
        "variant_id",
        "encoded_frame",
        "source_box_index",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "width",
        "height",
    ]

    print(f"Source: {source}")
    print(f"Output: {output}")
    print(f"Annotations: {len(tasks)}")
    print(f"Workers: {args.workers}")

    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_handle:
        writer = csv.DictWriter(manifest_handle, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_annotation, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                for key in (
                    "processed_images",
                    "written",
                    "skipped_invalid",
                    "skipped_non_person",
                    "skipped_missing_image",
                    "skipped_existing",
                ):
                    totals[key] += int(result[key])
                totals["errors"] += len(result["errors"])
                writer.writerows(result["rows"])

                if index % 1000 == 0 or index == len(futures):
                    print(
                        "Processed "
                        f"{index}/{len(futures)} annotations, "
                        f"written={totals['written']}, "
                        f"skipped_invalid={totals['skipped_invalid']}, "
                        f"errors={totals['errors']}",
                        flush=True,
                    )

                if result["errors"]:
                    with error_path.open("a", encoding="utf-8") as error_handle:
                        for error in result["errors"]:
                            error_handle.write(f"{result['json']}\t{error}\n")

    with summary_path.open("w", encoding="utf-8") as summary_handle:
        json.dump(totals, summary_handle, indent=2)
        summary_handle.write("\n")

    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
