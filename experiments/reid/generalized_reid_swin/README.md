# Generalized Multi-Domain ReID Swin

This experiment trains one Swin Base ReID model using:

- namespaced Duke, LTCC, and PRCC real training identities
- an exact `116,920`-crop filtered synthetic pool, equal to `50%` of the original synthetic manifest
- an identity-disjoint real validation split built from the three real training datasets

Official Duke, LTCC, and PRCC query/gallery folders remain untouched for final
standard evaluation. A separate namespaced combined split is generated as an
additional cross-domain stress test.

Start detached training on the RTX 3090:

```bash
experiments/reid/generalized_reid_swin/start_train_detached.sh
```

Check progress:

```bash
experiments/reid/generalized_reid_swin/check_progress.sh
```

After stable checkpoints exist, evaluate the latest one on GPU1:

```bash
GPU_DEVICE=1 experiments/reid/generalized_reid_swin/evaluate_latest_all_targets.sh
```

Evaluate every currently available stable checkpoint on Duke, LTCC, PRCC, and
the combined stress test while GPU0 training continues:

```bash
experiments/reid/generalized_reid_swin/start_evaluate_all_gpu1_detached.sh
```

The evaluator is resumable. Running it again after later checkpoints appear
reuses completed PASS rows and evaluates only missing target/checkpoint pairs.
The checkpoint sweep disables re-ranking by default because TAO performs that
stage on CPU and it is too expensive for every checkpoint, especially for the
combined stress-test gallery. Use re-ranking only for selected final
checkpoints. It also disables TAO sampled-match image grids because those plots
render one row per query and do not affect mAP or CMC metrics. Compact CMC curve
plots remain enabled.

Check concurrent evaluation progress:

```bash
experiments/reid/generalized_reid_swin/check_evaluation_progress.sh
```

While training is still running, start the detached follower once. It waits for
the current queue, discovers later checkpoints, and performs a final resumable
sweep after the training container exits:

```bash
experiments/reid/generalized_reid_swin/start_follow_training_checkpoints_gpu1_detached.sh
```
