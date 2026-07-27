# PRCC Plain Then Filtered Synthetic Swin

This workflow runs two controlled NVIDIA TAO ReID training stages sequentially on the RTX 3090:

1. Plain PRCC train only.
2. PRCC train plus synthetic crops filtered to at most three variants for one person-at-moment group.

Both stages start independently from the same Swin Base pretrained model. Evaluation must use only the real PRCC `query` and `bounding_box_test` directories.

Start detached training:

```bash
cd /mnt/2tb_ssd/TwinProject
experiments/reid/prcc_syntetic_filtered_seq/start_train_detached.sh
```

Follow progress:

```bash
docker logs -f tao_prcc_plain_then_filtered_gpu0
```

Compact status:

```bash
experiments/reid/prcc_syntetic_filtered_seq/check_progress.sh
```

Evaluate every stable checkpoint on the real PRCC query/gallery split using
both GPUs, then regenerate the repository-wide report:

```bash
experiments/reid/prcc_syntetic_filtered_seq/start_evaluate_all_reverse_detached.sh
```

Evaluation progress is written to:

```bash
tail -f experiments/reid/prcc_syntetic_filtered_seq/evaluate/reverse_evaluator_gpu0.log
tail -f experiments/reid/prcc_syntetic_filtered_seq/evaluate/reverse_evaluator_gpu1.log
```
