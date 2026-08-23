# Skeleton annotation agreement: 17 COCO keypoints

Stage A4 of an annotation-quality portfolio. The annotation was done by hand
in CVAT, blind to the ground truth, and compared against
`person_keypoints_val2017`.

## 1. What was annotated

14 COCO val2017 frames, selected for holding three or four annotated people
each; the frames of the earlier detection stage were excluded from the
selection so as not to annotate the same photographs a third time. 47 person
figures, 17 COCO keypoints per figure, roughly 620 points placed by hand.

The ground truth was not opened until the annotation was finished -- neither
the points themselves nor how many of them there were.

The thresholds were fixed in advance and never tuned to the result:

| threshold | value | where it comes from |
|---|---|---|
| minimum annotated points per figure | 8 | the ground-truth filter and the frame selection |
| minimum figure area | 4000 px² | decided before annotating, rationale below |
| person matching, box IoU | 0.3 | the script default |
| PCK tolerance | 1 COCO sigma for the joint | contribution to OKS no lower than 0.607 |

The area threshold was not chosen to make the work easier. On a person
smaller than 2000 px², flawless hand work -- a two-pixel tremor -- caps OKS at
0.84, and below 1000 px² at 0.68. Below the threshold the metric would be
measuring the resolution of the photograph rather than the accuracy of the
annotator. The ground truth itself annotated none of the eleven people below
1000 px² that appear in these frames.

## 2. Method

Three different things are measured, and none of them replaces the others.

**OKS** -- coordinate accuracy. The distance to the ground-truth point is
normalised by the area of the figure and by a per-joint sigma, so the same
error costs differently on a large and a small person, and differently on the
nose and on the hip. The sigmas are a measurement of the spread with which
humans re-annotate a joint, not a tuning knob. Computed in two modes: over
common points (averaged over what both sides annotated) and COCO-style (a
point I did not place contributes zero).

**Per-joint PCK** -- the share of points inside the tolerance, separately for
each of the 17 joints. The tolerance is expressed in sigmas rather than
pixels, which keeps PCK and OKS comparable.

**Visibility-flag agreement and Cohen's kappa** -- an axis that boxes,
polygons and tracks do not have. Beyond its coordinates, a point carries a
decision: annotated and visible (`v=2`), annotated but not visible (`v=1`),
not annotated at all (`v=0`). OKS is indifferent to that decision.

**People are matched by the box built from the annotated points, not by OKS**
-- and that matters more than it looks. COCOeval matches by OKS, but there the
other side is a model. Here it would be wrong: a person with swapped sides
scores an OKS of about 0.1-0.4, fails the matching threshold and drops out of
the count together with their error, and the metric goes up as a result. The
matching axis is deliberately kept separate. Both variants were computed and
the difference is stated in section 5.

The implementation uses the Python standard library; the Hungarian algorithm
is hand-written and carried over from the tracking stage. It was verified
before any computation on four cases with known answers
(`common/oks.py --selftest`), including the control one: offsetting every
point by exactly one sigma must yield exp(-1/2) = 0.6065 regardless of joint
and person size.

## 3. Result

| | |
|---|---|
| frames | 14 |
| people, mine / ground truth | 47 / 47 |
| matched pairs | 43 |
| **OKS over common points** | **0.895** (median 0.916, minimum 0.404) |
| OKS COCO-style | 0.868 |
| PCK@1 sigma | 0.954 |
| **visibility-flag agreement** | **0.822** |
| Cohen's kappa on the flag | 0.345 (617 slots) |

Unmatched: 4 in the ground truth, 4 of mine -- of the latter, 2 correspond to
a person dropped by the ground-truth filter (annotated with fewer than eight
points, or below the area threshold) and 2 correspond to nothing in the
ground truth at all.

The flag matrix, rows are the ground truth, columns are mine:

| GT \ mine | v=0 not annotated | v=1 not visible | v=2 visible |
|---|---|---|---|
| v=0 not annotated | 0 | 23 | 13 |
| v=1 not visible | 8 | 25 | 19 |
| v=2 visible | 8 | 39 | 482 |

Worst joints by PCK, with the mean offset:

| joint | PCK | points | mean offset, px |
|---|---|---|---|
| right_hip | 0.90 | 40 | 13.6 |
| left_wrist | 0.91 | 33 | 11.1 |
| right_shoulder | 0.93 | 41 | 7.9 |
| left_ankle | 0.94 | 33 | 9.9 |
| right_wrist | 0.94 | 35 | 12.3 |

The face agrees best: eyes 1.8 px, ears 2.3-3.1 px, nose 2.7 px, at PCK
0.96-1.00. That is a direct consequence of a decision taken before annotating:
face points were placed zoomed in, because the tolerance there is 6-8 px
against 19-26 px on the hip and the ankle.

OKS is flat across figure size: small 0.895 (11 pairs), medium 0.902 (22),
large 0.879 (10). The area threshold did its job -- no dependence of the
metric on resolution is left.

## 4. Systematic disagreements

**Swapped sides -- one figure out of 43.** `000000551820.jpg`, ground truth
#1247453: OKS 0.404, and 0.880 when left and right are swapped back. Tested by
swapping across all 43 pairs: it improves exactly one and makes the other 42
worse, so there is no systematic confusion. That one figure costs 0.011 of the
overall OKS (0.895 instead of 0.906). Not fixable by the guidelines: the rule
that sides belong to the person and not to the picture was in the guidelines
from the start and it is correct -- what was skipped was the colour check
before handing the figure in. In version 2 that check became a mandatory step.
The annotation was deliberately left uncorrected: fixing it after comparing
against the ground truth would be fitting to the ground truth.

**Ears -- 14 points, 13% of the flag disagreements.** The ground truth does
not annotate an ear hidden by the head or the hair at all; I placed it along
the line of the head with a "not visible" flag, because an ear can be located
precisely that way. The ground-truth rule is visible across all of val2017:
the ear is unannotated for 41% of people, against 3% for the shoulder. Fixable
by the guidelines, and the rule was adopted in the COCO reading: an ear that
cannot be seen is not annotated.

**The visible / not-visible boundary -- 58 points, 53% of the flag
disagreements.** 39 points were marked `occluded` by me where the ground truth
considers them visible, and 19 the other way round, 8 of those 19 being
ankles. Not fixable by the guidelines: I have no wording that would draw the
same boundary for two different annotators. Instead of promising a fix, the
report carries the kappa and the matrix. An indirect check that the boundary
is real rather than invented: on points the ground truth marked as not
visible, my offset was 11.0 px against 7.8 px on visible ones, PCK 0.864
against 0.962. Guessing at a hidden joint is objectively more expensive -- for
both sides.

**The disagreement that did not happen: the image border.** Of the 36 points I
placed where the ground truth places nothing, exactly one sits near the border
of the frame. The COCO rule -- do not invent a point beyond the frame, do not
pin it to the border, do not place it at all -- was measured and written into
the guidelines **before** annotating. The comparison with the tracking stage
is direct: there the MOT17 ground truth carries a box past the edge of the
image, my guidelines clipped at the edge, and that single rule produced 72 of
77 misses. The same question, the opposite answer; the difference is whether
the convention was stated up front or silently assumed.

## 5. What the metric does not see

The main section: the measured price of decisions that are usually made by
feel.

**OKS is blind to the visibility flag.** An annotation in which every point
sits exactly where the ground truth has it, but is marked "visible" without
discrimination, scores OKS 1.000 at a flag agreement of 0.900. That is a
computed number on these same 47 figures, not an estimate. My own run shows
the same thing in a milder form: OKS 0.895 and PCK 0.954 look respectable
while every sixth flag disagrees.

**Raw agreement without kappa is misleading.** For that same stand-in
annotation which always says "visible", kappa is exactly 0.000: an annotator
who never changes their answer carries no information about visibility. My
0.822 at kappa 0.345 means "noticeably better than chance, but far from
complete", and without the second number the first would read as a good
result. The same trick as with classes in the detection stage, where kappa
came out at 0.914.

**Matching by the metric hides the worst cases.** Matching by OKS instead of
by the box gave the same number of pairs on this data (43) but a different
partition, and OKS 0.899 against 0.895. The difference is small precisely
because there is only one swapped-sides figure here; on an annotation with a
dozen of them, those are the first to drop out of the count, and the metric
rises because they are gone.

**An acceptance threshold cannot be set in pixels.** On this data a systematic
offset of every point by 3 px gives OKS 0.966, by 8 px 0.817, by 12 px 0.684.
The same 8 px offset costs 0.705 on small people and 0.922 on large ones. An
acceptance threshold stated in pixels means different demands on the annotator
depending on which frames they happened to get.

**A convention is a property of the dataset, not of correctness.** Three
stages, one and the same question about the image border, three different
answers: MOT17 carries the box past the edge; COCO Keypoints does not place a
point beyond the edge at all -- zero out of 40,255; Total-Text (the OCR stage)
preserves the letter case of the sign, and a different rule there costs 0.702
CER. An annotator who brings a habit from a previous project accumulates
disagreements for no reason at all. Which is why the convention is stated when
a batch is handed over rather than assumed.

## 6. What changed in the guidelines

`annotation/GUIDELINES.md`: version 1 was written and committed before the
first annotated figure, version 2 after the analysis. Version 1 was left
untouched -- the history of the rules is part of the artefact.

| date | revision | volume |
|---|---|---|
| 2026-08-23 | an ear hidden by the head or the hair is not annotated (`v=0`) | 14 points, 13% of the flag disagreements |
| 2026-08-23 | the side check by skeleton colour is a mandatory step on every figure | 1 figure of 43, 0.011 of the overall OKS |
| 2026-08-23 | decided not to fix: the visible / not-visible boundary | 58 points, 53% of the disagreements; carried into the kappa and the matrix instead |

## Limitations

**47 figures is about 620 points, which leaves between 20 observations per
joint (ears) and 42 (shoulders).** The overall OKS is stable at that volume;
per-joint PCK is not: the difference between 0.90 and 0.95 over thirty points
means nothing. PCK is read here as an order of magnitude and as a pointer to
where to look, not as a measurement accurate to the second decimal.

**The COCO ground truth is careless in places.** Joints under loose clothing
are placed by eye there, exactly as they are by me; a disagreement on those
does not mean that I was the one who got it wrong. For the same reason neither
side is treated as "correct" in the visibility-flag disputes.

**This is a comparison against one ground truth, not between two annotators.**
It is therefore an upper bound on agreement: part of the disagreement is
explained by the convention of the dataset rather than by the quality of the
hand, and the two can only be separated where the convention has been measured
(the image border, the ears).

## Neighbouring stages

The same method on other annotation types, each with its own agreement metric:

- **Boxes** -- [detection-annotation-quality](https://github.com/daviddolya/detection-annotation-quality): 100 frames, kappa on classes 0.914, mean IoU 0.867.
- **Polygons** -- [polygon-annotation-agreement](https://github.com/daviddolya/polygon-annotation-agreement): 25 frames, mask IoU 0.840, Boundary IoU 0.676.
- **Tracks** -- [tracking-annotation-agreement](https://github.com/daviddolya/tracking-annotation-agreement): MOT17-09, IDF1 0.896, 2 identity switches, and an analysis of amodality beyond the image border.

Section 5 reads worse without them: boxes, polygons and tracks have no axis
like the one the annotation diverged on here -- there the metric sees
everything that was annotated.
