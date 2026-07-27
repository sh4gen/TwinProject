# Synthetic-Only Filtered 30k Data Audit

## Scope

Training uses synthetic crops only. Real PRCC query and gallery images are referenced only for later target-domain evaluation.

## Filtering Rule

The original manifest contains `233840` crops in `2054` person-at-moment groups across `39` identities. The exact `30000`-crop training set keeps up to `14` lowest-ID variants from every group, then adds the next variant from `1309` groups. Extra groups are selected deterministically in identity-balanced round-robin order.

## Training Configuration

- Backbone: `swin_base_patch4_window7_224`
- Input: `256x128`
- Epochs: `120`
- Batch size: `48`
- Optimizer: `SGD`, base LR `0.0006`, momentum `0.9`, weight decay `0.0001`
- Schedule: LR steps `[40, 70]`, cosine warmup for `20` epochs
- Sampling: `softmax_triplet`, `4` instances per identity
