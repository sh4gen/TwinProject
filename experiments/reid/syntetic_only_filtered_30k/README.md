# Synthetic-Only Filtered 30k Swin

This NVIDIA TAO ReID experiment trains a Swin Base model on exactly `30,000`
filtered synthetic crops and `39` synthetic identities. It does not include
real PRCC images in training. Real PRCC query/gallery directories are reserved
for later target-domain evaluation.

The dataset filter retains up to `14` variants from every person-at-moment
group, then adds a fifteenth variant from `1,309` groups using deterministic
identity-balanced round-robin selection.

Start detached training on the RTX 3090:

```bash
experiments/reid/syntetic_only_filtered_30k/start_train_detached.sh
```

Check progress:

```bash
experiments/reid/syntetic_only_filtered_30k/check_progress.sh
```
