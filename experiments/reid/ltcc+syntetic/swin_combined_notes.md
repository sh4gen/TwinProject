# Swin Combined LTCC + Synthetic

Strategy:

- Backbone: `swin_base_patch4_window7_224`
- Pretrained model: `models/reid/swin_base_market1501_aicity156_featuredim1024.tlt`
- Train data: LTCC `bounding_box_train` plus merged synthetic `bounding_box_train` and `bounding_box_test`
- Validation/evaluation data: LTCC `query` and `bounding_box_test` only

Combined train set:

- Total images: 13,062
- Total identities: 84
- LTCC identities: 77
- Synthetic identities: 7, remapped to `9001` through `9007` to avoid collisions with LTCC IDs

Run:

```bash
./start_train_swin_combined.sh
```

Evaluate:

```bash
./evaluate_swin_checkpoint.sh /mnt/2tb_ssd/TwinProject/experiments/reid/ltcc+syntetic/results_swin_combined/train/<checkpoint>.pth
```
