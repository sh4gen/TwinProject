# LTCC Filtered Synthetic Data Audit

## Filtering Rule

The original synthetic training set had many visual variants of the same moment. Rows were grouped by `pid`, `camera_id`, `sequence_id`, `frame_id`, and `source_box_index`; up to `3` representative crops were kept from each group, choosing the lowest `variant_id` values.

## Counts

- Original synthetic train images: `233840`
- Unique person-at-moment groups: `2054`
- Synthetic images kept after filtering: `6152`
- Variants kept per group: `3`
- Synthetic IDs kept: `39`
- LTCC train images: `9576`
- Combined train images: `15728`
- Combined train IDs: `116`
- Synthetic PID offset: `1151`

## Variant Group Sizes

| Variants per same moment | Number of groups |
| --- | --- |
| 1 | 5 |
| 47 | 1 |
| 50 | 734 |
| 149 | 12 |
| 150 | 1302 |

## Evaluation Split Policy

Training uses LTCC train plus filtered synthetic train. Evaluation uses only LTCC `query` and `bounding_box_test`; synthetic query/test are not used.
