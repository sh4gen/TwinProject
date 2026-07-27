# Duke ReID Experiment Comparison

Date: 2026-05-07

## Evaluation Setup

All results below are evaluated on Duke query/gallery only. Synthetic data was used only for the Duke+syntetic training run and was not used in evaluation.

Important dataset caveat: the current Duke+syntetic training run used the existing merged training folder, which TAO reported as 706 train IDs / 11,108 train images. That is Duke train plus a small synthetic subset, not the entire synthetic Duke source folder that exists under `experiments/reid/syntetic`.

## Headline Result

The external plain Duke Swin model is still stronger than the Duke+syntetic Swin run.

| Model | Checkpoint | mAP | Rank-1 | Rank-5 | Rank-10 |
| --- | --- | ---: | ---: | ---: | ---: |
| Plain Duke Swin | `plain_duke_model_epoch_199_step_10800` | 89.0% | 90.6% | 95.6% | 96.6% |
| Best Duke+syntetic Swin | `model_epoch_109_step_61202` | 85.9% | 89.3% | 93.4% | 95.6% |
| Delta, Duke+syntetic - plain | `model_epoch_109_step_61202` | -3.1 | -1.3 | -2.2 | -1.0 |

Best Duke+syntetic checkpoint by mAP and Rank-1: `model_epoch_109_step_61202`.

## Plain Duke Baseline

| Checkpoint | mAP | Rank-1 | Rank-5 | Rank-10 |
| --- | ---: | ---: | ---: | ---: |
| `plain_duke_model_epoch_199_step_10800` | 89.0% | 90.6% | 95.6% | 96.6% |

The moved Downloads folder path contains a non-ASCII character, which Hydra could not parse reliably. Evaluation used this ASCII symlink before checkpoint sanitization:
`experiments/reid/duke/external_plain_checkpoints/plain_duke_model_epoch_199_step_10800.pth`

Evaluated sanitized checkpoint:
`experiments/reid/duke/external_plain_checkpoints/sanitized_plain_duke_model_epoch_199_step_10800.pth`

## Duke+Syntetic Checkpoint Sweep

| Checkpoint | mAP | Rank-1 | Rank-5 | Rank-10 |
| --- | ---: | ---: | ---: | ---: |
| `model_epoch_009_step_05566` | 81.8% | 86.5% | 91.9% | 93.9% |
| `model_epoch_019_step_11134` | 82.2% | 84.9% | 92.3% | 93.7% |
| `model_epoch_029_step_16701` | 82.8% | 85.5% | 92.6% | 94.7% |
| `model_epoch_039_step_22270` | 83.9% | 86.6% | 92.5% | 94.3% |
| `model_epoch_049_step_27833` | 83.3% | 86.2% | 92.5% | 93.9% |
| `model_epoch_059_step_33396` | 83.9% | 87.7% | 92.7% | 94.3% |
| `model_epoch_069_step_38962` | 84.7% | 87.7% | 93.2% | 95.3% |
| `model_epoch_079_step_44521` | 84.6% | 86.9% | 92.6% | 95.0% |
| `model_epoch_089_step_50084` | 85.3% | 88.0% | 93.3% | 94.3% |
| `model_epoch_099_step_55641` | 85.8% | 87.9% | 93.2% | 94.4% |
| `model_epoch_109_step_61202` | 85.9% | 89.3% | 93.4% | 95.6% |
| `model_epoch_119_step_66761` | 85.4% | 87.2% | 93.6% | 95.3% |
| `model_epoch_129_step_72328` | 85.2% | 86.9% | 93.2% | 94.6% |
| `model_epoch_139_step_77890` | 85.7% | 88.0% | 93.3% | 95.4% |
| `model_epoch_149_step_83456` | 85.5% | 88.3% | 93.3% | 95.0% |
| `model_epoch_159_step_89021` | 85.6% | 88.6% | 93.0% | 94.6% |
| `model_epoch_169_step_94588` | 85.3% | 87.5% | 93.4% | 95.3% |
| `model_epoch_179_step_100150` | 84.6% | 87.6% | 93.6% | 95.0% |
| `model_epoch_189_step_105716` | 84.6% | 87.6% | 93.3% | 94.0% |
| `model_epoch_199_step_111283` | 84.3% | 86.9% | 92.5% | 94.4% |

## Interpretation

The Duke+syntetic run improves steadily until roughly epoch 109, then plateaus and declines. The synthetic-augmented run did not beat the plain Duke baseline on the Duke-only evaluation set.

The most likely issue is not GPU usage or evaluation setup. The evaluation used CUDA, and the query/gallery set was Duke only. The bigger experimental issue is data composition: this run did not include the whole synthetic Duke source dataset, only the current small synthetic subset already present in the merged train folder. A follow-up run should rebuild the Duke+syntetic train folder from Duke train plus the intended full synthetic train subset, then repeat the same Duke-only evaluation.

## Artifacts

Plain baseline summary:
`experiments/reid/duke/evaluation_results_swin_working_plain/summary.tsv`

Duke+syntetic sweep summary:
`experiments/reid/duke+syntetic/evaluation_results_swin_working_lowbatch/summary.tsv`

Duke+syntetic training config:
`experiments/reid/duke+syntetic/duke_syntetic_swin_working_lowbatch.yaml`

Duke+syntetic evaluation script:
`experiments/reid/duke+syntetic/evaluate_all_swin_working_checkpoints.sh`
