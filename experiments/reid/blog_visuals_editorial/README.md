# NVIDIA Blog Editorial Visuals

This directory contains the publication-oriented replacement set for the
earlier dashboard and slide-style visuals.

## Selected Assets

| Figure | File | Purpose |
| --- | --- | --- |
| 1 | `figure_1_editorial_hero_final.png` | Lead image: synthetic annotation, eight-to-three filtering, Swin-style embedding, and LTCC clothing-change retrieval |
| 2 | `figure_2_reid_pipeline_architecture.png` | Simplified deployment workflow: TensorRT-optimized YOLO detection and Swin ReID, per-camera tracking, and global identity association |
| 3 | `figure_3_editorial_filtering_final.png` | Method image: repeated synthetic variants reduced to three representative crops |
| 5 | `figure_5_editorial_ltcc_impact.png` | Exact LTCC real-only, unfiltered, and filtered synthetic comparison |
| 6 | `figure_6_editorial_cross_domain_impact.png` | Exact LTCC, PRCC, and Duke comparison showing domain-dependent synthetic impact |

## Suggested Captions

**Figure 1.** The project converts JSON-annotated synthetic scenes into
person crops, removes repetitive variants, and trains a Swin-based NVIDIA TAO
ReIdentification model whose embeddings associate the same real person across
cameras and clothing changes.

**Figure 2.** Camera streams pass through TensorRT-optimized YOLO person
detection and a TensorRT-optimized, TAO-trained Swin ReID model. Per-camera
tracks are then associated with one persistent cross-camera identity.

**Figure 3.** Synthetic crops were grouped by identity, camera, sequence,
source frame, and bounding box. Retaining at most three variants per group
reduced the pool from 233,840 to 6,152 crops while preserving all 39 synthetic
identities.

**Figure 5.** Adding the filtered synthetic subset improved LTCC mAP from
23.8% to 43.8% and Rank-1 accuracy from 50.3% to 76.1%, using 2.63% of the
original synthetic crop volume.

**Figure 6.** The same filtered subset produced a large improvement on LTCC,
almost no change on PRCC, and a small regression on Duke, showing that
synthetic augmentation is domain-dependent.

## Accuracy Notes

- The synthetic scene is an actual repository frame.
- Its green boxes come from the corresponding project JSON annotation.
- The repeated synthetic woman crops are from one person, camera, frame, and
  bounding-box group.
- The hero and filtering image show exactly three retained variants.
- The real retrieval sequence is one LTCC identity in black and white
  clothing across different cameras.
- Figure 2 presents the optimized deployment workflow at a conceptual level:
  YOLO person detection, person cropping, TAO-trained Swin ReID, per-camera
  tracking, and downstream global identity association.
- All evaluations shown in the charts use real benchmark query/gallery
  splits. Synthetic images are training inputs only.
- No dashboard, fictional metric, invented dataset, or fabricated deployment
  component is shown.

## Production

The hero, pipeline, and filtering figures were generated with the built-in
OpenAI image generation tool using repository images as references. The
pipeline prompt is stored in `figure_2_pipeline_prompt.md`; the other exact
prompts are stored in `PROMPTS.md`.

Private source/reference crops, intermediate alternatives, and proof sheets
remain local. The selected publication figures above are the versioned assets.

The two metric figures are deterministic Matplotlib outputs. Regenerate them
with:

```bash
MPLCONFIGDIR=/tmp/twinproject-mpl-cache \
python3 experiments/reid/blog_visuals_editorial/render_editorial_charts.py
```
