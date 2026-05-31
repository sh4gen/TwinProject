# Duke Filtered Synthetic Data Audit

## Filtering Rule

Rows were grouped by `pid`, `camera_id`, `sequence_id`, `frame_id`, and `source_box_index`; up to `3` crops were retained from each group by choosing the lowest `variant_id` values.

## Counts

- Original synthetic train images: `233840`
- Unique person-at-moment groups: `2054`
- Synthetic images kept after filtering: `6152`
- Variants kept per group: `3`
- Synthetic IDs kept: `39`
- Duke train images: `8784`
- Combined train images: `14936`
- Combined train IDs: `741`
- Synthetic PID offset: `8140`

## Variant Group Sizes

| Variants per same moment | Number of groups |
| --- | --- |
| 1 | 5 |
| 47 | 1 |
| 50 | 734 |
| 149 | 12 |
| 150 | 1302 |

## Evaluation Split Policy

Training uses Duke train plus filtered synthetic train. Evaluation uses only Duke `query` and `bounding_box_test`; synthetic query/test are not used.
