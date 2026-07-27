# Final Synthetic Market-1501 Dataset

Converted from `final_syntetic_dataset/SYNTHETIC DATAS`.

## Structure

```text
final_syntetic_market1501/
  bounding_box_train/
  bounding_box_test/
  query/
  manifest.csv
  summary.json
```

All synthetic crops are in `bounding_box_train`. `bounding_box_test` and `query`
are intentionally empty so this dataset is used only for training, not for
evaluation.

## Conversion Rules

- Crops were extracted from the non-meta JSON annotation files.
- Person identity comes from each box's `character_id`.
- Camera, sequence, frame, and variant come from the JSON/source filename.
- Output filename format is `PPPP_cCsS_FFFFFF_BB.jpg`.
- `FFFFFF` encodes `frame_id * 1000 + variant_id`.
- `manifest.csv` preserves the original source image, JSON path, bbox
  coordinates, character ID, semantic ID, camera ID, sequence ID, frame ID, and
  variant ID for every crop.

## Summary

```json
{
  "annotations": 67900,
  "duplicate_annotation_files": 34,
  "processed_images": 67749,
  "written": 233840,
  "skipped_invalid": 2100,
  "skipped_non_person": 0,
  "skipped_missing_image": 0,
  "skipped_existing": 0,
  "errors": 0
}
```

Unique training identities: 39.

## References

- NVIDIA TAO data annotation format: Market-1501 uses `bounding_box_train`,
  `bounding_box_test`, and `query`.
- NVIDIA TAO ReIdentificationNet docs: TAO expects Market-1501 format for
  training and evaluation.
