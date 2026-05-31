# Duke Filtered Synthetic Sequential Experiment

This experiment mirrors the latest LTCC filtered-synthetic setup for Duke.

## Filtering

The synthetic manifest contains many versions of the same underlying moment. The preparation script groups rows by:

- `pid`
- `camera_id`
- `sequence_id`
- `frame_id`
- `source_box_index`

It retains the three lowest `variant_id` values from each group. Synthetic IDs are offset before merging with Duke so they do not collide with Duke IDs.

## Data Policy

Training uses:

- Duke `bounding_box_train`
- filtered synthetic `bounding_box_train`

Evaluation uses only:

- Duke `query`
- Duke `bounding_box_test`

Synthetic query/test images are not used.

## Configuration

The model configuration is derived from the transferred Duke Swin setup that previously worked locally:

- Swin Base backbone
- input size `128x256`
- feature dimension `1024`
- `200` epochs
- GPU0 / RTX 3090
- configurable default batch size `32`

## Commands

Prepare data and start detached training:

```bash
cd /mnt/2tb_ssd/TwinProject
experiments/reid/duke_syntetic_filtered_seq/start_train_detached.sh
```

Run attached:

```bash
cd /mnt/2tb_ssd/TwinProject
experiments/reid/duke_syntetic_filtered_seq/run_sequential_rtx3090.sh
```

Evaluate the latest checkpoint on Duke query/gallery using GPU1:

```bash
cd /mnt/2tb_ssd/TwinProject
experiments/reid/duke_syntetic_filtered_seq/evaluate_latest.sh
```

Evaluate all checkpoints from newest to oldest using GPU0 and GPU1 in parallel:

```bash
cd /mnt/2tb_ssd/TwinProject
experiments/reid/duke_syntetic_filtered_seq/start_evaluate_all_reverse_detached.sh
```

Monitor both evaluator workers:

```bash
tail -f experiments/reid/duke_syntetic_filtered_seq/evaluate/reverse_evaluator_gpu0.log
tail -f experiments/reid/duke_syntetic_filtered_seq/evaluate/reverse_evaluator_gpu1.log
```

Merge the two worker summaries:

```bash
experiments/reid/duke_syntetic_filtered_seq/merge_reverse_summaries.sh
```
