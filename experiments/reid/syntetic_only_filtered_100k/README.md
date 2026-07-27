# Synthetic-Only Filtered 100k Swin

This experiment trains a Swin Base ReID model on exactly `100,000` synthetic crops and no real training images. It is the lowered-filtering follow-up to the completed 30k synthetic-only diagnostic run.

Real datasets are reserved for evaluation only. Training uses synthetic images from `datasets/final_syntetic_market1501/bounding_box_train`.

Start detached training on GPU 0:

```bash
experiments/reid/syntetic_only_filtered_100k/start_train_detached.sh
```

Check progress:

```bash
experiments/reid/syntetic_only_filtered_100k/check_progress.sh
```
