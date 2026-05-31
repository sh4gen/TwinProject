# Duke Filtered Synthetic Evaluation

## Best Checkpoint

The reverse checkpoint sweep completed successfully across all 20 saved checkpoints.

| Item | Value |
| --- | --- |
| Best checkpoint | `model_epoch_099_step_43340.pth` |
| mAP | 86.4% |
| Rank-1 | 89.7% |
| Rank-5 | 94.4% |
| Rank-10 | 95.3% |

## Final Checkpoint

Evaluation completed successfully on the Duke-only query/gallery split.

| Item | Value |
| --- | --- |
| Checkpoint | `model_epoch_199_step_86706.pth` |
| Query | Duke `query`, 702 images |
| Gallery | Duke `bounding_box_test`, 10,541 images |
| Distance | Euclidean distance with re-ranking |
| mAP | 84.7% |
| Rank-1 | 89.5% |
| Rank-5 | 93.2% |
| Rank-10 | 94.6% |

## Comparison

| Experiment | Checkpoint | mAP | Rank-1 | Rank-5 | Rank-10 |
| --- | --- | ---: | ---: | ---: | ---: |
| Plain Duke Swin baseline | `plain_duke_model_epoch_199_step_10800` | 89.0% | 90.6% | 95.6% | 96.6% |
| Previous Duke + small synthetic subset, best checkpoint | `model_epoch_109_step_61202` | 85.9% | 89.3% | 93.4% | 95.6% |
| Duke + filtered synthetic, best checkpoint | `model_epoch_099_step_43340` | 86.4% | 89.7% | 94.4% | 95.3% |
| Duke + filtered synthetic, final checkpoint | `model_epoch_199_step_86706` | 84.7% | 89.5% | 93.2% | 94.6% |

## Notes

- Synthetic images were used only in training. Evaluation used Duke query/gallery only.
- The filtered experiment trained with 8,784 Duke images and 6,152 filtered synthetic images, totaling 14,936 images and 741 classes.
- All 20 checkpoints were evaluated from newest to oldest using GPU0 and GPU1.
- The filtered run peaked at epoch 99 and then regressed by 1.7 mAP points by epoch 199.
- The full sweep table is saved at `evaluate/duke_filtered_syntetic/summary_reverse.tsv`.
