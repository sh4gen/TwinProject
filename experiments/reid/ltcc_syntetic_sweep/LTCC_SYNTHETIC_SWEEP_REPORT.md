# LTCC Synthetic ReID Sweep

Generated: 2026-05-31 15:15:34

## Purpose

We are measuring how synthetic person crops affect an LTCC ReID model. Training sets mix LTCC train identities with different percentages of the synthetic train split, but every evaluation uses only the real LTCC query and gallery/test split. This keeps the metric tied to real LTCC performance and prevents synthetic images from leaking into evaluation.

## Dataset Separation

The synthetic dataset was converted to Market-1501-style ReID files before this sweep. Only `bounding_box_train` from `datasets/final_syntetic_market1501` is used for training. Synthetic `query` and `bounding_box_test` folders are intentionally ignored in this LTCC experiment.

| Split | Path | Images | IDs |
| --- | --- | --- | --- |
| LTCC train | /mnt/2tb_ssd/TwinProject/experiments/reid/ltcc/data/bounding_box_train | 9576 | 77 |
| LTCC query | /mnt/2tb_ssd/TwinProject/experiments/reid/ltcc/data/query | 493 | 75 |
| LTCC gallery/test | /mnt/2tb_ssd/TwinProject/experiments/reid/ltcc/data/bounding_box_test | 7026 | 75 |
| Synthetic train | /mnt/2tb_ssd/TwinProject/datasets/final_syntetic_market1501/bounding_box_train | 233840 | 39 |

## How The Mixed Train Sets Were Built

`prepare_ltcc_syntetic_sweep.py` groups synthetic images by person ID and takes the requested percentage from each synthetic identity. That means 10%, 25%, 50%, 75%, and 100% keep all synthetic identities represented instead of randomly dropping complete people. Synthetic person IDs are offset by `max(LTCC PID) + 1000` before linking into generated train folders, so synthetic IDs cannot collide with LTCC IDs.

| Experiment | Synthetic % | LTCC images | Synthetic images | LTCC IDs | Synthetic IDs |
| --- | --- | --- | --- | --- | --- |
| ltcc_syntetic_10 | 10 | 9576 | 23385 | 77 | 39 |
| ltcc_syntetic_25 | 25 | 9576 | 58470 | 77 | 39 |
| ltcc_syntetic_50 | 50 | 9576 | 116924 | 77 | 39 |
| ltcc_syntetic_75 | 75 | 9576 | 175391 | 77 | 39 |
| ltcc_syntetic_100 | 100 | 9576 | 233840 | 77 | 39 |
| syntetic_only_100 | 100 | 0 | 233840 | 0 | 39 |

## Common Training Configuration

- Backbone: `swin_base_patch4_window7_224`
- Pretrained file: `/mnt/2tb_ssd/TwinProject/models/reid/swin_base_market1501_aicity156_featuredim1024.tlt`
- Input size: `384x192`
- Losses: ID `softmax` plus metric `triplet`
- Optimizer: `SGD`, base LR `0.00035`, momentum `0.9`, weight decay `0.0005`
- Current synthetic-only run: `100` epochs, batch `48`, validation batch `128`, workers `8`
- Re-ranking: `True`, k1 `20`, k2 `6`, lambda `0.3`

## Evaluation Policy

All checkpoint evaluations override the config evaluation paths to LTCC query and LTCC `bounding_box_test`. The currently running synthetic-only training is not stopped; GPU0 remains assigned to training and GPU1 is used for evaluation.

The pretrained Swin `.tlt` baseline is evaluated on LTCC by overriding `dataset.num_classes=857`, because the checkpoint classifier has 857 rows while LTCC has 77 train IDs. This affects checkpoint loading only; the reported metrics still come from LTCC query/gallery embeddings.

## Best Evaluated Checkpoint Per Experiment

| Experiment | Best checkpoint | mAP | Rank-1 | Rank-5 | Rank-10 |
| --- | --- | --- | --- | --- | --- |
| ltcc_syntetic_10 | model_epoch_029_step_59207 | 40.7% | 72.6% | 80.3% | 82.4% |
| ltcc_syntetic_25 | model_epoch_009_step_40440 | 34.9% | 66.7% | 74.4% | 78.9% |
| ltcc_syntetic_50 | model_epoch_014_step_112349 | 27.1% | 56.0% | 67.1% | 72.2% |
| pretrained_swin_market1501_aicity156 | swin_base_market1501_aicity156_featuredim1024 | 7.5% | 23.1% | 35.7% | 42.6% |
| syntetic_only_100 | model_epoch_004_step_22005 | 6.4% | 15.6% | 26.2% | 31.2% |

## All Evaluation Rows

| Experiment | Checkpoint | mAP | Rank-1 | Rank-5 | Rank-10 | Status |
| --- | --- | --- | --- | --- | --- | --- |
| ltcc_syntetic_10 | model_epoch_004_step_09879 | 38.5% | 70.8% | 77.7% | 79.7% | passed |
| ltcc_syntetic_10 | model_epoch_009_step_19730 | 39.7% | 72.4% | 77.5% | 79.7% | passed |
| ltcc_syntetic_10 | model_epoch_014_step_29612 | 39.6% | 70.6% | 77.7% | 80.3% | passed |
| ltcc_syntetic_10 | model_epoch_019_step_39478 | NA | NA | NA | NA | failed |
| ltcc_syntetic_10 | model_epoch_019_step_39478 | NA | NA | NA | NA | failed |
| ltcc_syntetic_10 | model_epoch_019_step_39478 | NA | NA | NA | NA | failed |
| ltcc_syntetic_10 | model_epoch_019_step_39478 | NA | NA | NA | NA | failed |
| ltcc_syntetic_10 | model_epoch_024_step_49336 | 40.2% | 73.2% | 78.9% | 81.1% | passed |
| ltcc_syntetic_10 | model_epoch_029_step_59207 | 40.7% | 72.6% | 80.3% | 82.4% | passed |
| ltcc_syntetic_10 | model_epoch_034_step_69098 | 40.5% | 73.2% | 79.1% | 81.9% | passed |
| ltcc_syntetic_10 | model_epoch_039_step_78942 | 39.6% | 72.8% | 79.9% | 82.8% | passed |
| ltcc_syntetic_10 | model_epoch_044_step_88790 | 39.5% | 72.6% | 79.3% | 81.1% | passed |
| ltcc_syntetic_10 | model_epoch_049_step_98636 | 38.4% | 70.2% | 77.9% | 81.7% | passed |
| ltcc_syntetic_10 | model_epoch_054_step_108508 | 40.5% | 72.8% | 79.3% | 82.2% | passed |
| ltcc_syntetic_10 | model_epoch_059_step_118324 | 38.9% | 72.0% | 77.1% | 80.9% | passed |
| ltcc_syntetic_10 | model_epoch_064_step_128213 | 39.0% | 71.8% | 78.9% | 80.9% | passed |
| ltcc_syntetic_10 | model_epoch_069_step_138039 | 38.8% | 72.4% | 79.1% | 81.9% | passed |
| ltcc_syntetic_10 | model_epoch_074_step_147920 | 38.7% | 72.2% | 79.7% | 82.6% | passed |
| ltcc_syntetic_10 | model_epoch_079_step_157768 | 37.1% | 70.2% | 78.3% | 81.3% | passed |
| ltcc_syntetic_10 | model_epoch_084_step_167599 | 37.6% | 71.6% | 80.1% | 82.2% | passed |
| ltcc_syntetic_10 | model_epoch_089_step_177469 | 37.3% | 71.8% | 79.1% | 81.9% | passed |
| ltcc_syntetic_10 | model_epoch_094_step_187354 | 36.8% | 69.2% | 75.7% | 78.7% | passed |
| ltcc_syntetic_10 | model_epoch_099_step_197194 | 37.1% | 71.6% | 77.5% | 80.7% | passed |
| ltcc_syntetic_10 | model_epoch_104_step_207090 | 37.0% | 70.0% | 77.9% | 81.5% | passed |
| ltcc_syntetic_10 | model_epoch_109_step_216983 | 35.0% | 68.0% | 78.7% | 80.7% | passed |
| ltcc_syntetic_10 | model_epoch_114_step_226838 | 36.8% | 69.8% | 76.9% | 81.3% | passed |
| ltcc_syntetic_10 | model_epoch_119_step_236720 | 35.0% | 67.3% | 75.7% | 79.3% | passed |
| ltcc_syntetic_10 | model_epoch_124_step_246598 | 35.4% | 69.0% | 75.5% | 78.3% | passed |
| ltcc_syntetic_10 | model_epoch_129_step_256450 | 33.8% | 64.3% | 74.6% | 78.7% | passed |
| ltcc_syntetic_10 | model_epoch_134_step_266302 | 35.1% | 67.7% | 75.9% | 80.5% | passed |
| ltcc_syntetic_10 | model_epoch_139_step_276159 | NA | NA | NA | NA | failed |
| ltcc_syntetic_10 | model_epoch_139_step_276159 | 34.0% | 66.7% | 75.5% | 77.5% | passed |
| ltcc_syntetic_10 | model_epoch_144_step_286037 | 34.5% | 67.3% | 75.7% | 78.3% | passed |
| ltcc_syntetic_10 | model_epoch_149_step_295897 | 33.1% | 65.1% | 72.8% | 77.3% | passed |
| ltcc_syntetic_10 | model_epoch_149_step_295897 | 33.1% | 65.1% | 72.8% | 77.3% | passed |
| ltcc_syntetic_25 | model_epoch_004_step_20234 | 31.1% | 62.7% | 71.4% | 76.5% | passed |
| ltcc_syntetic_25 | model_epoch_009_step_40440 | 34.9% | 66.7% | 74.4% | 78.9% | passed |
| ltcc_syntetic_25 | model_epoch_009_step_40440 | 34.9% | 66.7% | 74.4% | 78.9% | passed |
| ltcc_syntetic_50 | model_epoch_004_step_37518 | 15.3% | 35.5% | 46.0% | 51.3% | passed |
| ltcc_syntetic_50 | model_epoch_009_step_74889 | 24.2% | 52.3% | 63.9% | 68.0% | passed |
| ltcc_syntetic_50 | model_epoch_014_step_112349 | 27.1% | 56.0% | 67.1% | 72.2% | passed |
| ltcc_syntetic_50 | model_epoch_019_step_149773 | 27.0% | 57.0% | 67.1% | 71.0% | passed |
| ltcc_syntetic_50 | model_epoch_024_step_187175 | 27.0% | 56.0% | 67.3% | 72.6% | passed |
| ltcc_syntetic_50 | model_epoch_029_step_224523 | 25.8% | 55.6% | 64.5% | 70.2% | passed |
| ltcc_syntetic_50 | model_epoch_034_step_262083 | 24.5% | 48.9% | 62.7% | 67.5% | passed |
| ltcc_syntetic_50 | model_epoch_039_step_299555 | 25.3% | 52.9% | 65.9% | 70.2% | passed |
| ltcc_syntetic_50 | model_epoch_044_step_336978 | 25.0% | 52.9% | 68.2% | 71.6% | passed |
| ltcc_syntetic_50 | model_epoch_049_step_374323 | 22.7% | 48.9% | 62.3% | 67.1% | passed |
| ltcc_syntetic_50 | model_epoch_054_step_411865 | 22.1% | 49.5% | 62.1% | 65.5% | passed |
| ltcc_syntetic_50 | model_epoch_059_step_449245 | 25.1% | 53.3% | 66.5% | 70.8% | passed |
| ltcc_syntetic_50 | model_epoch_064_step_486751 | 22.2% | 46.9% | 60.6% | 66.7% | passed |
| ltcc_syntetic_50 | model_epoch_069_step_524265 | 24.8% | 52.3% | 63.3% | 69.4% | passed |
| ltcc_syntetic_50 | model_epoch_074_step_561684 | 23.1% | 50.1% | 63.9% | 67.5% | passed |
| ltcc_syntetic_50 | model_epoch_074_step_561684 | 23.1% | 50.3% | 63.9% | 67.5% | passed |
| ltcc_syntetic_50 | model_epoch_079_step_599146 | 22.8% | 48.9% | 60.9% | 67.3% | passed |
| ltcc_syntetic_50 | model_epoch_084_step_636649 | 23.0% | 50.3% | 63.1% | 68.2% | passed |
| ltcc_syntetic_50 | model_epoch_084_step_636649 | 23.0% | 50.3% | 63.1% | 68.2% | passed |
| ltcc_syntetic_50 | model_epoch_089_step_673942 | 23.8% | 51.7% | 64.9% | 69.0% | passed |
| ltcc_syntetic_50 | model_epoch_094_step_711395 | 22.7% | 50.9% | 63.7% | 67.7% | passed |
| ltcc_syntetic_50 | model_epoch_099_step_748797 | 21.3% | 48.1% | 60.6% | 66.7% | passed |
| pretrained_swin_market1501_aicity156 | swin_base_market1501_aicity156_featuredim1024 | 7.5% | 23.1% | 35.7% | 42.6% | passed |
| syntetic_only_100 | model_epoch_004_step_22005 | 6.4% | 15.6% | 26.2% | 31.2% | passed |
| syntetic_only_100 | model_epoch_004_step_22005 | 6.4% | 15.6% | 26.2% | 31.2% | passed |
| syntetic_only_100 | model_epoch_009_step_44054 | 4.7% | 11.6% | 21.9% | 28.4% | passed |
| syntetic_only_100 | model_epoch_014_step_66096 | 4.1% | 10.8% | 21.7% | 26.8% | passed |

## Current Run Status

| Status file | Latest status |
| --- | --- |
| /mnt/2tb_ssd/TwinProject/experiments/reid/ltcc_syntetic_sweep/results/syntetic_only_100_bs48_gpu0_detached/train/status.json | RUNNING: Train metrics generated. (5/24/2026 16:53:1) |
| /mnt/2tb_ssd/TwinProject/experiments/reid/ltcc_syntetic_sweep/evaluation_full_gpu1/pretrained/status.json | RUNNING: Evaluate finished successfully. (5/24/2026 10:52:35) |

## Files

- Sweep data builder: `/mnt/2tb_ssd/TwinProject/experiments/reid/ltcc_syntetic_sweep/prepare_ltcc_syntetic_sweep.py`
- Training launcher: `/mnt/2tb_ssd/TwinProject/experiments/reid/ltcc_syntetic_sweep/run_ltcc_syntetic_sweep.sh`
- Synthetic-only GPU0 launcher: `/mnt/2tb_ssd/TwinProject/experiments/reid/ltcc_syntetic_sweep/start_syntetic_only_gpu0.sh`
- Full GPU1 evaluator: `/mnt/2tb_ssd/TwinProject/experiments/reid/ltcc_syntetic_sweep/evaluate_all_available_gpu1.sh`
- Full evaluation summary: `/mnt/2tb_ssd/TwinProject/experiments/reid/ltcc_syntetic_sweep/evaluation_full_gpu1/summary.tsv`
- Earlier progress summary: `/mnt/2tb_ssd/TwinProject/experiments/reid/ltcc_syntetic_sweep/evaluation_progress_gpu1/summary.tsv`

## Notes

- GPU0 synthetic-only training is compute-bound: Swin Base saturates the RTX 3090, while disk and dataloader wait are low.
- GPU1 is used for evaluation only because previous TAO training on the RTX 5070 was unstable.
- `reid_model_latest.pth` is not evaluated while training is running because it can be overwritten mid-read; only stable `model_epoch_*.pth` checkpoints are queued.
