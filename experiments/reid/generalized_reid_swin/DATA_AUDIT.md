# Generalized ReID Swin Data Audit

## Purpose

Train one Swin Base ReID model for real cross-domain generalization. Training combines namespaced real Duke, LTCC, and PRCC identities with a filtered 50% synthetic pool. Checkpoint selection uses a separate identity-disjoint real validation split. Official benchmark query/gallery folders remain untouched for final evaluation.

## Namespace Policy

| Domain | PID offset | Camera offset |
| --- | ---: | ---: |
| duke | 10000 | 100 |
| ltcc | 20000 | 200 |
| prcc | 30000 | 300 |
| syntetic | 40000 | 400 |

## Real Partition

| Domain | Source train images | Source IDs | Train images | Train IDs | Validation query | Validation gallery | Validation IDs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| duke | 8784 | 702 | 7892 | 632 | 70 | 822 | 70 |
| ltcc | 9576 | 77 | 8699 | 69 | 8 | 869 | 8 |
| prcc | 22898 | 150 | 20747 | 135 | 15 | 2136 | 15 |

## Synthetic Filter

- Original crops: `233840`
- Selected crops: `116920` (`50.00%`)
- Synthetic IDs: `39`
- Retained variants per person-at-moment group: up to `61`, plus one extra variant for `14` groups
- Extra groups are selected deterministically in identity-balanced round-robin order.

## Training Totals

- Real train images: `37338`
- Synthetic train images: `116920`
- Total train images: `154258`
- Real train IDs: `836`
- Synthetic train IDs: `39`
- Total classes: `875`
- Held-out real validation IDs: `93`

## Hyperparameters

- Backbone: `swin_base_patch4_window7_224`
- Pretrained initialization: Market-1501/AICity Swin Base `.tlt`
- Input: `256x128`
- Epochs: `120`
- Batch size: `48`
- Sampler: `softmax_triplet`, `4` instances per identity
- Optimizer: SGD, LR `0.0006`, momentum `0.9`, weight decay `0.0001`
- LR schedule: steps `[40, 70]`, gamma `0.1`, cosine warmup for `20` epochs
- Augmentation: horizontal flip `0.5`, random erasing `0.5`, padding `10`
- Validation and checkpoint interval: every `10` epochs

## Evaluation Policy

Final evaluation must report Duke, LTCC, and PRCC official query/gallery metrics independently. The namespaced combined official split is an additional cross-domain stress test, not a replacement for the standard per-dataset metrics.
