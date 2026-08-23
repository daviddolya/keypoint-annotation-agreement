# Working notes

## Step 1 -- OKS

2026-08-23, `common/oks.py --selftest`: all four cases match.
OKS 1.0000 / 0.6065 / 0.0767 / 1.0000; flag agreement in the fourth case 0.867.

What follows from that (my own words):

## Step 2 -- the price of a pixel

The `tools/oks_sensitivity.py` run on my own subset: 47 people, an 8 px offset
of every point gives mean OKS 0.817; small people 0.705, large ones 0.922;
at 8 px the nose retains 0.435 of its contribution and the hip 0.952.

**Decision 1. Lower area threshold -- 4000 px².** A person smaller than that
is not annotated, and this is written into the guidelines. Rationale: the 14
frames hold 94 person figures, and flawless hand work (a 2 px tremor) caps OKS
at 0.68 below 1000 px², 0.84 between 1000 and 2000, 0.92 between 2000 and 4000,
and 0.96+ above 4000. Below the threshold the metric would measure the
resolution of the picture, not my accuracy. The COCO ground truth itself
annotated none of the eleven people below 1000 px². The threshold coincides
with the frame-selection filter and with `--min-area` of the comparison
script, so the report states a single number.

**Decision 2. Zoom on the face and the wrists.** Nose, eyes and ears have a
one-sigma tolerance of 6-8 px on a person of median size: an 8 px error leaves
0.435 of the nose contribution and 0.632 of the ear. Those get placed zoomed
in. Wrists (15 px tolerance) get zoomed when needed -- they are occluded more
often. Shoulders, hips and ankles (19-26 px) are placed at normal zoom: the
same error costs 0.91-0.95 there, and the extra minute does not pay off.

## Step 5 -- computing agreement

`annotation/keypoint_agreement.py`, default filters (min-kp 8, min-area 4000,
matching by the keypoint box, threshold 0.3). Numbers only, conclusions in
step 6.

14 frames; 47 people in the ground truth, 47 of mine. 43 pairs, 4 unmatched on
each side: of my unmatched, 2 were dropped by the ground-truth filter and 2
correspond to nothing in the ground truth. Matching by OKS produced the same
number of pairs (43) at OKS 0.899 against 0.895 -- a slightly different
partition, but the worst pairs did not drop out of the count.

OKS over common points 0.895 (median 0.916, minimum 0.404), OKS COCO-style
0.868, PCK@1 sigma 0.954.

Worst joints by PCK: right_hip 0.90 (13.6 px), left_wrist 0.91 (11.1 px),
right_shoulder 0.93 (7.9 px). Best: eyes, ears and nose (PCK 0.96-1.00,
offsets 1.8-3.1 px).

Visibility-flag agreement 0.822, Cohen's kappa 0.345, 617 slots.

| GT \ mine | v=0 | v=1 | v=2 |
|---|---|---|---|
| v=0 | 0 | 23 | 13 |
| v=1 | 8 | 25 | 19 |
| v=2 | 8 | 39 | 482 |

OKS by size: small 0.895 (11 pairs), medium 0.902 (22), large 0.879 (10).

Worst pairs: 000000551820.jpg #1247453 <-> #43 OKS 0.404 (12/12 points);
000000233771.jpg #220854 <-> #25 OKS 0.669; 000000100624.jpg #500399 <-> #12
OKS 0.774; 000000023899.jpg #448028 <-> #1 OKS 0.798; 000000080340.jpg
#542289 <-> #8 OKS 0.822.

Metrics: `reports/keypoint_metrics.json`.

## Step 6 -- systematic disagreements

Three of them, each in the form "number -- volume -- conclusion".

**1. Swapped sides, one figure.** `000000551820.jpg`, ground truth #1247453:
OKS 0.404, rising to 0.880 when left and right are swapped back. Tested by
swapping across all 43 pairs -- it improves exactly one and makes the other 42
worse, so there is no systematic confusion. Volume: 1 figure of 43, cost 0.011
of the overall OKS (0.895 instead of 0.906). Conclusion: not fixable by the
guidelines -- the rule about sides was there in version 1 and it is correct;
what was not done was the colour check before handing the figure in. The
annotation stays as it is: correcting it after comparing against the ground
truth would be fitting to the ground truth. The report states the number.

**2. Ears: 14 points.** The ground truth does not annotate an ear hidden by
the head or the hair at all (41% of ears in val2017 carry v=0); I marked them
occluded. That is 14 of the 110 flag disagreements, i.e. 13%. Conclusion:
fixable by the guidelines, and I adopt the COCO rule -- an ear I cannot see is
not annotated. Goes into version 2.

**3. The visible / not-visible boundary: 58 points.** I marked 39 points
occluded where the ground truth considers them visible, and 19 the other way
round (8 of those are ankles). That is 53% of all flag disagreements.
Conclusion: not fixable by the guidelines. The boundary is subjective for any
annotator, and stating kappa 0.345 and the 3x3 matrix is more honest than
promising a fix. Indirect confirmation: on points the ground truth marked as
not visible my offset is 11.0 px against 7.8 px on visible ones, PCK 0.864
against 0.962 -- guessing at a hidden joint is objectively more expensive.

**What did not happen.** The image border produced no disagreement: of the 36
points placed where the ground truth places nothing, exactly one sits near the
border. The rule was stated before annotating -- compare with the tracking
stage, where the same amodality question produced 72 of 77 misses.
