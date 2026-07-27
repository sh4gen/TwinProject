# LTCC + Synthetic ReID Fine-Tune

This experiment starts from the best saved LTCC TAO checkpoint found locally:

- Source run: `experiments/reid/ltcc/results_0.1.4`
- Checkpoint: `model_epoch_059_step_32791.pth`
- Evaluation: mAP `0.236`, Rank-1 `0.467`, Rank-5 `0.619`, Rank-10 `0.692`

Dataset wiring:

- Train: `data/bounding_box_train` -> merged synthetic Market-1501 train + test/gallery split
- Validate/evaluate gallery: `data/bounding_box_test` -> original LTCC gallery split
- Validate/evaluate query: `data/query` -> original LTCC query split

Current default strategy is `ltcc_syntetic_swin_combined.yaml`: train Swin Base on LTCC train plus synthetic train/gallery, then validate/evaluate only on LTCC query/gallery.

The combined Swin train set uses 13,062 images and 84 identities. Synthetic IDs are remapped to `9001` through `9007` to avoid collisions with LTCC IDs.

The older `ltcc_syntetic_transfer.yaml` trained only on synthetic images and performed poorly on LTCC validation.

Use `ltcc_syntetic_transfer.yaml` only if you specifically want synthetic-only transfer. The synthetic train set has 7 identities, so TAO builds a 7-class classifier head and a strict Lightning resume from the 77-class LTCC checkpoint fails.

`ltcc_syntetic_resume.yaml` is kept only as a strict-resume reference for cases where the train identity set matches the original 77-class LTCC head.

The selected LTCC checkpoint stores a ResNet-50 model even though one archived experiment spec listed Swin settings. The active synthetic fine-tune specs use `resnet_50` to match the actual checkpoint state dict.

Example commands:

```bash
tao model re_identification train \
  -e /mnt/2tb_ssd/TwinProject/experiments/reid/ltcc+syntetic/ltcc_syntetic_swin_combined.yaml

tao model re_identification evaluate \
  -e /mnt/2tb_ssd/TwinProject/experiments/reid/ltcc+syntetic/ltcc_syntetic_swin_combined.yaml \
  evaluate.checkpoint=/mnt/2tb_ssd/TwinProject/experiments/reid/ltcc+syntetic/results_swin_combined/train/<checkpoint>.pth \
  evaluate.query_dataset=/mnt/2tb_ssd/TwinProject/experiments/reid/ltcc+syntetic/data/query \
  evaluate.test_dataset=/mnt/2tb_ssd/TwinProject/experiments/reid/ltcc+syntetic/data/bounding_box_test
```

The synthetic set is small in identity count, so compare every saved checkpoint against the original LTCC query/gallery split and keep the best validation checkpoint, not necessarily the final epoch.
