# LTCC Filtered Synthetic Sequential Experiment

This experiment filters repeated synthetic variants before combining the synthetic training data with LTCC.

## Filtering

The synthetic manifest contains many versions of the same underlying moment. The preparation script groups rows by:

- `pid`
- `camera_id`
- `sequence_id`
- `frame_id`
- `source_box_index`

It keeps the three lowest `variant_id` values from each group. Synthetic IDs are offset before combining with LTCC so they do not collide with LTCC IDs.

## Data Policy

Training uses:

- LTCC `bounding_box_train`
- filtered synthetic `bounding_box_train`

Evaluation uses only:

- LTCC `query`
- LTCC `bounding_box_test`

Synthetic query/test images are not used.

## Commands

Prepare data and start training detached on RTX 3090:

```bash
cd /mnt/2tb_ssd/TwinProject
experiments/reid/ltcc_syntetic_filtered_seq/start_train_detached.sh
```

Run attached instead:

```bash
cd /mnt/2tb_ssd/TwinProject
experiments/reid/ltcc_syntetic_filtered_seq/run_sequential_rtx3090.sh
```

Evaluate latest stable checkpoint on LTCC:

```bash
cd /mnt/2tb_ssd/TwinProject
experiments/reid/ltcc_syntetic_filtered_seq/evaluate_latest.sh
```
