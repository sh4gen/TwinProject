# Synthetic Data Audit

Checked dataset: `experiments/reid/syntetic/synthetic_market1501`

Format:

- Directory layout matches NVIDIA TAO ReID/Market-1501 convention: `bounding_box_train`, `bounding_box_test`, `query`.
- All image filenames match the expected identity/camera pattern: `0001_c10s1_000002_00.jpg`.
- Non-image metadata file present: `dataset_info.json`.

Counts:

- Synthetic train images used by this experiment: 3,486
- Synthetic train identities used by this experiment: 7 (`0001`, `0002`, `0003`, `0007`, `0013`, `0014`, `0015`)
- Source synthetic train images: 2,324
- Source synthetic gallery/test images added to train: 1,162
- Synthetic query images: 80
- LTCC validation gallery images: 7,026
- LTCC validation query images: 493

Issues and risks:

- The merged synthetic train split has only 7 identities. This is usable for short adaptation but still high risk for overfitting.
- Synthetic camera ids range up to 150. The filename format is valid, but this is much larger than normal LTCC camera count.
- Synthetic IDs do not correspond to LTCC person IDs. Validation on original LTCC query/gallery is correct for measuring whether synthetic adaptation helps the real target domain.
- Do not use the synthetic query/gallery as the primary score for this experiment unless you specifically want synthetic-domain performance.
