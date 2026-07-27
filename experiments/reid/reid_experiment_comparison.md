# ReID LTCC Synthetic Experiment Comparison

Generated on 2026-05-06 after evaluating the plain LTCC Swin checkpoints.

## Scope

All comparable numbers below use the LTCC query/gallery split for evaluation:

- Query: `experiments/reid/ltcc/data/query`, 493 images, 75 IDs
- Gallery: `experiments/reid/ltcc/data/bounding_box_test`, 7026 images, 75 IDs
- Metrics: TAO Re-Identification evaluation with feature normalization, Euclidean distance, and re-ranking

The main question was whether synthetic data helps LTCC ReID when fine-tuning with NVIDIA TAO.

## Best Results

| Experiment | Backbone | Training data | Best checkpoint | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---|---|---:|---:|---:|---:|
| LTCC + synthetic Swin, best mAP | Swin base | LTCC train + synthetic train | `model_epoch_024` | 43.1% | 74.4% | 81.7% | 84.6% |
| LTCC + synthetic Swin, best Rank-1 | Swin base | LTCC train + synthetic train | `model_epoch_034` | 42.3% | 75.7% | 81.1% | 83.2% |
| Plain LTCC Swin, best mAP | Swin base | LTCC train only | `model_epoch_099` | 23.8% | 50.3% | 62.3% | 67.5% |
| Plain LTCC Swin, best Rank-1 | Swin base | LTCC train only | `model_epoch_124` | 22.5% | 51.1% | 61.5% | 66.1% |
| Previous LTCC baseline | ResNet | LTCC train only | `model_epoch_059` | 23.6% | 46.7% | 61.9% | 69.2% |
| Synthetic-only transfer, best mAP | ResNet | Synthetic train only | `model_epoch_004` | 2.8% | 5.5% | 15.0% | 23.5% |

## Deltas

Using the best mAP checkpoint for each experiment:

| Comparison | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| LTCC + synthetic Swin vs previous LTCC ResNet | +19.5 | +27.7 | +19.8 | +15.4 |
| LTCC + synthetic Swin vs plain LTCC Swin | +19.3 | +24.1 | +19.4 | +17.1 |
| Plain LTCC Swin vs previous LTCC ResNet | +0.2 | +3.6 | +0.4 | -1.7 |
| Synthetic-only transfer vs previous LTCC ResNet | -20.8 | -41.2 | -46.9 | -45.7 |

Using the best Rank-1 checkpoint for the Swin experiments:

| Comparison | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| LTCC + synthetic Swin vs plain LTCC Swin | +19.8 | +24.6 | +19.6 | +17.1 |

## Findings

The current best model is the combined LTCC + synthetic Swin run. It reaches 43.1% mAP and 74.4% Rank-1 at epoch 24, while the best Rank-1 appears at epoch 34 with 75.7%.

Plain LTCC Swin did not materially beat the old LTCC ResNet baseline on mAP. It improved Rank-1 from 46.7% to 50.3%, but mAP stayed almost flat and Rank-10 was lower than the old ResNet result.

Synthetic-only fine-tuning was not useful for LTCC. It collapsed to 2.8% mAP and about 5.5% Rank-1, which is consistent with domain shift and/or forgetting when real LTCC training data is replaced by synthetic data.

The useful strategy is to mix synthetic samples with the real LTCC training split, then validate only on LTCC query/gallery. In this setup, synthetic data appears to add meaningful coverage without replacing the LTCC distribution.
