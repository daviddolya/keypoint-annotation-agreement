# keypoint-annotation

Annotation agreement for human skeletons: 17 COCO keypoints per person.
14 val2017 frames were annotated by hand, blind to the
ground truth. Three different things are measured: coordinate accuracy
(OKS), accuracy per individual joint (PCK), and agreement on the
visibility flag -- the last of which no coordinate metric can see.
Stage A4 of an annotation-quality portfolio.

## Result

| | |
|---|---|
| frames | 14 |
| people, mine / ground truth | 47 / 47 |
| matched pairs | 43 |
| **OKS over common points** | **0.895** |
| OKS COCO-style | 0.868 |
| PCK@1 sigma | 0.954 |
| **visibility-flag agreement** | **0.822** |
| Cohen's kappa on the flag | 0.345 |

People are matched by the box built from the annotated keypoints, IoU threshold 0.3: that axis stays independent of
the metric, otherwise a person with swapped left and right sides finds
no match and the worst case simply vanishes from the report.

Unmatched: 4 in the ground truth, 4 of mine (2 of them match a person the ground-truth filter dropped, 2 match nothing in the ground truth).

## What a coordinate metric cannot see

The visibility flag is a separate axis of the annotation, and OKS does
not touch it at all: an annotation with perfect coordinates in which
every point is marked visible scores exactly 1.000. Flag agreement is
therefore computed on its own, with kappa next to it: raw agreement
looks high by itself whenever one flag value dominates the rest.

| GT \ mine | v=0 not annotated | v=1 not visible | v=2 visible |
|---|---|---|---|
| v=0 not annotated | 0 | 23 | 13 |
| v=1 not visible | 8 | 25 | 19 |
| v=2 visible | 8 | 39 | 482 |

![flag disagreement by joint](reports/review/flag_by_joint.png)

## Where the hand misses

The PCK tolerance is one COCO sigma for that joint at that scale, i.e.
a contribution to OKS of no less than exp(-1/2) = 0.607. The threshold
was fixed in advance and never tuned to the result.

![PCK by joint](reports/review/pck_by_joint.png)

| worst joints | PCK | points | mean offset, px |
|---|---|---|---|
| right_hip | 0.90 | 40 | 13.6 |
| left_wrist | 0.91 | 33 | 11.1 |
| right_shoulder | 0.93 | 41 | 7.9 |
| left_ankle | 0.94 | 33 | 9.9 |
| right_wrist | 0.94 | 35 | 12.3 |

| person size | pairs | OKS |
|---|---|---|
| small | 11 | 0.895 |
| medium | 22 | 0.902 |
| large | 10 | 0.879 |

## The worst pairs

Blue is the ground truth, orange is mine. A point is filled when it
is marked visible and hollow when it is marked not visible.

### 000000551820.jpg - ground truth #1247453

OKS 0.404

![000000551820_1247453](reports/review/01_000000551820_gt1247453.jpg)

### 000000233771.jpg - ground truth #220854

OKS 0.669

![000000233771_220854](reports/review/02_000000233771_gt220854.jpg)

### 000000100624.jpg - ground truth #500399

OKS 0.774

![000000100624_500399](reports/review/03_000000100624_gt500399.jpg)

### 000000023899.jpg - ground truth #448028

OKS 0.798

![000000023899_448028](reports/review/04_000000023899_gt448028.jpg)

### 000000080340.jpg - ground truth #542289

OKS 0.822

![000000080340_542289](reports/review/05_000000080340_gt542289.jpg)

### 000000132544.jpg - ground truth #1676854

OKS 0.840

![000000132544_1676854](reports/review/06_000000132544_gt1676854.jpg)

### 000000551820.jpg - ground truth #1248862

OKS 0.853

![000000551820_1248862](reports/review/07_000000551820_gt1248862.jpg)

### 000000474078.jpg - ground truth #1729803

OKS 0.865

![000000474078_1729803](reports/review/08_000000474078_gt1729803.jpg)

## Reproduce

Python 3.10+ and Pillow; Pillow is needed for the rendering only, all
metrics run on the standard library (the Hungarian algorithm included).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# the four self-test cases with answers known in advance
.venv/bin/python common/oks.py --selftest

# ground truth (10 MB, not stored in the repository)
.venv/bin/python tools/fetch_keypoints.py --out data/coco

# sanity-check the export before computing anything
.venv/bin/python tools/check_export.py \
    --mine annotation/person_keypoints_default.json \
    --selection data/subset/selection_keypoints.json

# the numbers in this README
.venv/bin/python annotation/keypoint_agreement.py \
    --gt data/coco/person_keypoints_val2017.json \
    --mine annotation/person_keypoints_default.json \
    --selection data/subset/selection_keypoints.json \
    --out reports/keypoint_metrics.json

# the pictures above, then this README
.venv/bin/python tools/render_skeletons.py \
    --gt data/coco/person_keypoints_val2017.json \
    --mine annotation/person_keypoints_default.json \
    --selection data/subset/selection_keypoints.json \
    --images data/subset/frames --out reports/review
.venv/bin/python tools/build_readme.py --repo keypoint-annotation
```

The 14 frames and the selection manifest are committed, so the numbers
can be reproduced without rebuilding the subset.

## What else is here

- Annotation guidelines and the disputed-case decisions -- [annotation/GUIDELINES.md](annotation/GUIDELINES.md)
- Full report -- [reports/keypoint_report.md](reports/keypoint_report.md)
- Code I did not write myself, and what I owe an explanation for -- [DEBT.md](DEBT.md)

This README is generated by `tools/build_readme.py` from
`reports/keypoint_metrics.json`; edit the report, not this file.
