# LTCC ReID Experiment Comparison

All metrics below were measured on the LTCC query/gallery split. Higher is better.

## Completed And Baseline Runs

| Experiment | Training data | Checkpoint | Synthetic train images | mAP | Rank-1 | Rank-5 | Rank-10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| External pretrained Swin baseline | External pretrained model only | `swin_base_market1501_aicity156_featuredim1024` | 0 | 7.5% | 23.1% | 35.7% | 42.6% | Evaluated directly on LTCC |
| Previous LTCC ResNet baseline | LTCC only | `model_epoch_059` | 0 | 23.6% | 46.7% | 61.9% | 69.2% | Earlier ResNet experiment |
| Plain LTCC Swin baseline | LTCC only | `model_epoch_099` | 0 | 23.8% | 50.3% | 62.3% | 67.5% | Best plain LTCC Swin mAP |
| Previous LTCC + synthetic Swin | LTCC + older unfiltered synthetic data | `model_epoch_024` | Not recorded here | **43.1%** | 74.4% | **81.7%** | **84.6%** | Best overall mAP |
| Previous LTCC + synthetic Swin | LTCC + older unfiltered synthetic data | `model_epoch_034` | Not recorded here | 42.3% | **75.7%** | 81.1% | 83.2% | Best overall Rank-1 |
| LTCC + 10% synthetic sweep | LTCC + 10% of converted synthetic data | `model_epoch_029_step_59207` | 23,385 | 40.7% | 72.6% | 80.3% | 82.4% | Best observed checkpoint |
| LTCC + filtered synthetic, 3 variants per moment | LTCC + filtered synthetic data | `model_epoch_149_step_69529` | 6,152 | 41.4% | 73.6% | 81.1% | 83.6% | Completed 150 epochs |

## Partial Or Diagnostic Runs

| Experiment | Training data | Checkpoint | mAP | Rank-1 | Rank-5 | Rank-10 | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Synthetic-only ResNet transfer | Synthetic only | `model_epoch_004` | 2.8% | 5.5% | 15.0% | 23.5% | Earlier diagnostic run |
| Synthetic-only Swin sweep | Synthetic only | `model_epoch_014_step_66096` | 4.1% | 10.8% | 21.7% | 26.8% | Interrupted snapshot |
| LTCC + 10% synthetic sweep terminal model | LTCC + 10% synthetic | `model_epoch_149_step_295897` | 33.1% | 65.1% | 72.8% | 77.3% | Completed, but regressed after its best checkpoint |
| LTCC + 25% synthetic sweep | LTCC + 25% synthetic | `model_epoch_009_step_40440` | 34.9% | 66.7% | 74.4% | 78.9% | Interrupted snapshot |
| LTCC + 50% synthetic sweep | LTCC + 50% synthetic | `model_epoch_084_step_636649` | 23.0% | 50.3% | 63.1% | 68.2% | Interrupted snapshot |

## Filtered Dataset Result

The filtered experiment retained at most three variants for each synthetic person-at-moment crop:

| Dataset | Images | IDs |
| --- | ---: | ---: |
| Original converted synthetic train | 233,840 | 39 |
| Filtered synthetic train | 6,152 | 39 |
| LTCC train | 9,576 | 77 |
| Combined filtered train | 15,728 | 116 |

The filtered run is close to the older best unfiltered combined model while using a much smaller synthetic set:

| Comparison | mAP delta | Rank-1 delta | Rank-5 delta | Rank-10 delta |
| --- | ---: | ---: | ---: | ---: |
| Filtered combined vs plain LTCC Swin | +17.6 | +23.3 | +18.8 | +16.1 |
| Filtered combined vs 10% sweep best checkpoint | +0.7 | +1.0 | +0.8 | +1.2 |
| Filtered combined vs older unfiltered combined best-mAP checkpoint | -1.7 | -0.8 | -0.6 | -1.0 |

## Notes

- The interrupted sweep rows are progress snapshots and must not be treated as fully trained experiment results.
- The older unfiltered combined run used an earlier generated dataset/configuration. Its LTCC evaluation split is comparable, but its training input is not identical to the new converted synthetic dataset.
- The 10% sweep shows late-epoch regression: its best observed mAP was 40.7% at epoch 29, while the terminal epoch 149 model reached 33.1%.
