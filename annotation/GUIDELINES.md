# Skeleton annotation guidelines, 17 COCO keypoints

Version 1, 2026-08-23, written before annotating. Dataset: COCO val2017,
14 frames, annotated blind -- the ground truth is not opened until the batch
is handed in. Revisions made after the disagreement analysis are appended
below as a separate dated section; version 1 stays exactly as it was.

## 1. A point beyond the image border

Not placed at all, flag `v=0` (in CVAT: leave the point unannotated). I do not
invent a coordinate outside the image and do not pin the point to the border.

Rationale: that is how the ground truth is built. Of the 40,255 annotated
points in val2017, not a single one lies outside its image and 11 are pinned
to the border. This rule is **the opposite** of the one I applied to MOT17
boxes on the tracking stage, where the ground truth carries a box past the
edge of the frame and my clip-to-border rule produced 72 of my 77 misses.
A convention is a property of the dataset, not of correctness, and it gets
stated when a batch is handed over rather than silently assumed.

Practical consequence: on a person cropped by the bottom edge, ankles and
knees usually go unannotated. In the ground truth such people have on average
5.3 of 17 points missing, against 3.7 for people fully inside the frame.

## 2. A joint hidden by the body, by clothing or by another person

Placed with flag `v=1` (in CVAT: occluded) when the position of the joint can
be estimated to within roughly one sigma: the shoulder line is visible, the
direction of the leg is visible, the limb emerges from behind the obstacle.

Not placed at all (`v=0`) when the estimate would be a guess.

Practical boundary. A person standing with their back to the camera: shoulders,
hips and knees get placed. A person entirely behind another with only the head
sticking out: only what I can see gets placed. A hip under loose clothing gets
placed -- the line of the torso gives it to within a sigma, and the tolerance
there is 25 px.

Why not "if I cannot see it I do not place it": that rule has been measured
and it costs 62 points out of 620, dropping COCO-style OKS to 0.886 and kappa
on the flag to 0.474. Why not "place everything and always mark it visible":
then OKS stays at exactly 1.000 while kappa falls to zero -- the flag stops
carrying any information at all.

## 3. Left and right on a person facing away

Sides are always the sides of the PERSON, not of the picture. On a person
standing with their back to the camera the left shoulder appears on the left
of the image; on a person walking towards the camera it appears on the right.
The decision is taken once per figure, from the orientation of the torso, and
every point follows it, including the ones placed with flag `v=1`.

Check before handing the figure in -- by the colour of the skeleton in CVAT:
the left half is blue, the right half orange. Swapped sides do not look like
a mistake on screen but they wreck the metric: a full swap drops OKS to 0.077
with every coordinate still in place.

## 4. Who gets annotated at all

A person is annotated when both conditions hold: at least **eight** points are
visible or confidently estimable, AND the area of the figure is at least
**4000 px²**.

The area threshold: on a person smaller than 2000 px² even flawless hand work
(a 2 px tremor) caps OKS at 0.84, and below 1000 px² at 0.68. Below the
threshold the metric would be measuring the resolution of the picture rather
than my accuracy. The ground truth itself annotated none of the eleven people
below 1000 px² that appear in these frames.

The eight-point threshold matches the filter of the comparison script
(`--min-kp 8`): a figure with three points yields OKS over three points, which
is noise rather than a measurement.

## 5. Where I aim more carefully

Nose, eyes and ears are placed zoomed in: the one-sigma tolerance there is
6-8 px on a person of median size, and an 8 px error leaves 0.435 of the
contribution of the nose point. Wrists (15 px tolerance) get zoomed when
needed -- they are occluded more often. Shoulders, hips and ankles (19-26 px)
are placed at normal zoom: the same error costs 0.91-0.95 there, and the extra
minute does not pay for itself.

---

# Revisions after the disagreement analysis

Version 2, 2026-08-23. Based on the run over 14 frames: 43 pairs, OKS 0.895,
flag agreement 0.822 at kappa 0.345. Version 1 above is left untouched.

## Revision 1 (2026-08-23). An ear that cannot be seen is not annotated

Before (section 2): a hidden joint is placed with `v=1` when its position can
be estimated to within a sigma. An ear passed that test, so I placed it along
the line of the head.

Now: **the ear is an exception.** An ear hidden by the head, by hair or by a
hat is not annotated at all (`v=0`), however obvious its position.

What happened: 14 of the 110 flag disagreements are an ear I marked `v=1`
where the ground truth places no point at all. The ground-truth rule is
visible across all of val2017: the ear is unannotated for 41% of people,
against 3% for the shoulder. The revision covers 13% of all flag disagreements.

## Revision 2 (2026-08-23). The side check is mandatory before a figure is done

Before (section 3): the colour check was named, but as advice.

Now: **the colour check is a mandatory step on every figure**, before moving
on to the next one. The left half is blue, the right half orange; on a person
walking towards the camera the blue half is on the right of the screen.

What happened: one figure out of 43 (`000000551820.jpg`) was annotated with
the sides swapped -- OKS 0.404 against 0.880 once swapped back. That single
figure cost 0.011 of the overall OKS. The annotation was deliberately left
uncorrected: fixing it after comparing against the ground truth would be
fitting the annotation to the ground truth.

## What was decided NOT to fix

The visible / not-visible boundary on joints other than ears. 39 points were
marked `occluded` by me where the ground truth considers them visible, and 19
the other way round -- together 53% of the flag disagreements. I have no
wording that would remove this: the boundary is subjective for any annotator.
Instead of promising a fix, the report carries kappa 0.345 and the 3x3 matrix.

When a batch is handed over, this convention is stated explicitly, together
with the border rule from section 1.
