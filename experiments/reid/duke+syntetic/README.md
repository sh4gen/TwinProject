# Duke + Synthetic ReID Experiment

This mirrors the LTCC synthetic experiment for Duke.

## Data

- Training data: `/mnt/2tb_ssd/TwinProject/experiments/reid/duke/data_merged/bounding_box_train`
- Duke train images: 8784
- Synthetic images in merged train: 2324
- Total merged train images: 11108
- Training classes: 706, meaning 702 Duke IDs plus 4 synthetic IDs
- Evaluation data stays Duke-only:
  - Query: `/mnt/2tb_ssd/TwinProject/experiments/reid/duke/data/query`
  - Gallery: `/mnt/2tb_ssd/TwinProject/experiments/reid/duke/data/bounding_box_test`

## Train

```bash
cd /mnt/2tb_ssd/TwinProject/experiments/reid/duke+syntetic
./start_train_swin_combined_150.sh
```

## Evaluate

After training, sanitize checkpoints first if TAO evaluation has trouble loading raw checkpoints:

```bash
cd /mnt/2tb_ssd/TwinProject/experiments/reid/duke+syntetic
docker run --rm --gpus all -v /mnt/2tb_ssd/TwinProject:/mnt/2tb_ssd/TwinProject nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt python /mnt/2tb_ssd/TwinProject/experiments/reid/duke+syntetic/sanitize_swin_checkpoints.py
./evaluate_all_swin_checkpoints.sh
```

The evaluation summary is written to:

`/mnt/2tb_ssd/TwinProject/experiments/reid/duke+syntetic/evaluation_results_swin_combined/summary.tsv`
