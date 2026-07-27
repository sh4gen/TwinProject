# PRCC Plain And Filtered Synthetic Data Audit

## Controlled Comparison

Both stages start independently from the same Swin Base pretrained model and use identical optimized hyperparameters. Stage 1 trains on real PRCC train only. Stage 2 trains on real PRCC train plus filtered synthetic crops. Evaluation always uses only real PRCC `query` and `bounding_box_test`.

## Filtering Rule

Rows were grouped by `pid`, `camera_id`, `sequence_id`, `frame_id`, and `source_box_index`; up to `3` crops were retained from each group by choosing the lowest `variant_id` values.

## Counts

- PRCC train images: `22898`
- PRCC train IDs: `150`
- Original synthetic train images: `233840`
- Unique person-at-moment synthetic groups: `2054`
- Synthetic images kept after filtering: `6152`
- Synthetic IDs kept: `39`
- Combined train images: `29050`
- Combined train IDs: `189`
- Synthetic PID offset: `1332`

## Optimized Hyperparameters

- Backbone: `swin_base_patch4_window7_224`
- Input: `256x128`
- Epochs per stage: `120`
- Train batch size: `48`
- Optimizer: `SGD`, base LR `0.0006`, momentum `0.9`, weight decay `0.0001`
- Schedule: LR steps `[40, 70]`, cosine warmup for `20` epochs
- Sampling: `softmax_triplet`, `4` instances per identity
- Augmentation: horizontal flip `0.5`, random erasing `0.5`, padding `10`
- Re-ranking: `k1=20`, `k2=6`, `lambda=0.3`

## Historical Split Note

The current local PRCC train directory contains `150` IDs. An older recorded PRCC experiment YAML contains `221` classes, so that historical run came from a different split and must remain a separate reference.
