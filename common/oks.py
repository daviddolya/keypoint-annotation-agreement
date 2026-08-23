#!/usr/bin/env python3
"""OKS, PCK and visibility-flag agreement.

WHY KEYPOINTS NEED THEIR OWN METRIC. A box and a polygon have area, so they
have IoU. A point has none: the intersection of two points is always empty
and IoU is identically zero. That leaves distance -- but raw distance in
pixels is unusable for two reasons at once. Five pixels on a person filling
the frame and five pixels on a person a hundred pixels tall are different
errors. And five pixels on the tip of the nose and five pixels on the hip
are different too: where exactly the hip sits under clothing is something
two careful annotators agree on far worse than where the nose is.

OKS fixes both problems with one formula:

    OKS = sum over annotated points of exp( -d_i^2 / (2 * s^2 * k_i^2) )
          divided by the number of annotated points

    d_i  distance between the ground-truth point and mine, in pixels
    s^2  object area in the ground truth. This normalises out scale
    k_i  = 2 * sigma_i, a per-joint constant

SIGMAS ARE A MEASUREMENT, NOT A TUNING KNOB. The COCO authors had part of
the images re-annotated and measured the spread with which humans place
each joint. Shoulder came out at 0.079, hip at 0.107, nose at 0.026. On the
same person, an error on the nose is therefore roughly four times as
"expensive" as the same error on the hip -- simply because annotators tend
to agree about the nose and not about the hip.

THE PRACTICAL CONSEQUENCE. An offset of exactly d = s * k_i yields
exp(-1/2) = 0.6065, which makes the implementation checkable on paper.
And OKS is asymmetric: the scale s comes from the ground truth, so "OKS of
my annotation against the ground truth" and the reverse are different
numbers. Here the ground truth is always the first argument.

WHAT OKS DOES NOT SEE. It is computed from coordinates and says nothing
about the visibility flag: an annotation with perfect coordinates in which
every point is marked "visible" scores exactly 1.000, even though some of
the flags disagree with the ground truth. Flag agreement is therefore a
separate metric rather than something folded into OKS.

Further reading: cocodataset.org/#keypoints-eval (definition and the sigma
table); Ronchi & Perona, "Benchmarking and Error Diagnosis in Multi-Instance
Pose Estimation", arXiv:1707.05388 (a breakdown of what pose error is made of).

The Hungarian algorithm is carried over unchanged from
tracking-annotation-agreement/common/matching.py, where it is documented in
detail. No dependencies, scipy not required.
"""

import argparse
import math
from collections import Counter

from keypoints import (ABSENT, HIDDEN, K, LR_PAIRS, VISIBLE, COCO_KEYPOINTS,
                       Person, bbox_iou, by_image)

INF = float("inf")


def hungarian(cost: list[list[float]]) -> list[tuple[int, int]]:
    """Minimum-cost assignment in a square matrix.

    The Hungarian algorithm in Kuhn-Munkres form, O(n^3). Carried over from
    common/matching.py of the tracking project unchanged. What matters is
    not how it works inside but that it is not greedy: picking "best pair
    first, then the next best" can take a locally attractive pair and leave
    another person with nothing but bad options.
    """
    n, m = len(cost), len(cost[0])
    assert n == m, "the matrix must be square -- pad it with dummy rows"
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] > 0]


# --- stage 1: the metric on a single pair of people ------------------------

def oks(gt: Person, mine: Person, mode: str = "common") -> float | None:
    """OKS of a (ground-truth person, my person) pair. Scale from the GT.

    mode='common' -- averaged over points annotated by BOTH sides. Answers
        "how precisely do I place the points I do place". Disagreement about
        whether to place a point at all goes into flag agreement instead.
    mode='coco'   -- averaged over all points annotated in the ground truth;
        a point I did not place contributes zero. This is how COCOeval works,
        where the other side is a model and "did not find it" is an error.

    The two numbers side by side are worth more than either alone: if they
    diverge, the problem is not the coordinates but which points to annotate.
    """
    num, den = 0.0, 0
    for i, ((gx, gy, gv), (mx, my, mv)) in enumerate(zip(gt.points, mine.points)):
        if gv == ABSENT:
            continue
        if mode == "common" and mv == ABSENT:
            continue
        den += 1
        if mv == ABSENT:
            continue                       # coco mode: zero contribution
        d2 = (gx - mx) ** 2 + (gy - my) ** 2
        num += math.exp(-d2 / (2 * gt.area * K[i] ** 2))
    return num / den if den else None


def point_errors(gt: Person, mine: Person) -> list[tuple[int, float, float]]:
    """(joint index, distance in pixels, one-sigma tolerance in pixels)
    over points annotated by both sides. Raw material for PCK and analysis."""
    out = []
    s = math.sqrt(gt.area)
    for i, ((gx, gy, gv), (mx, my, mv)) in enumerate(zip(gt.points, mine.points)):
        if gv == ABSENT or mv == ABSENT:
            continue
        out.append((i, math.hypot(gx - mx, gy - my), s * K[i]))
    return out


def pck(pairs: list[tuple[Person, Person]], mult: float = 1.0
        ) -> tuple[dict[int, tuple[int, int]], float]:
    """Per-joint PCK: the share of points that landed inside the tolerance.

    The tolerance is not a fixed number of pixels but mult * s * k_i -- the
    same scale and the same per-joint sigma as in OKS. At mult=1.0 a "hit"
    means a contribution to OKS of at least exp(-1/2) = 0.6065, which makes
    the two numbers comparable. The threshold is stated in the report rather
    than tuned to the result.

    Returns {joint index: (hits, total)} and the overall share.
    """
    per: dict[int, list[int]] = {i: [0, 0] for i in range(17)}
    for g, m in pairs:
        for i, dist, sigma_px in point_errors(g, m):
            per[i][1] += 1
            if dist <= mult * sigma_px + 1e-9:
                per[i][0] += 1
    hit = sum(v[0] for v in per.values())
    tot = sum(v[1] for v in per.values())
    return {i: (v[0], v[1]) for i, v in per.items()}, (hit / tot if tot else 0.0)


# --- stage 2: flag agreement, separate from the coordinates ----------------

def flag_confusion(pairs: list[tuple[Person, Person]]) -> dict:
    """3x3 visibility-flag matrix plus raw agreement and Cohen's kappa.

    A slot counts if at least one side said something about it: a point
    neither side annotated is agreement about nothing, and including it
    inflates the metric (a quarter of all slots in the GT are of that kind).

    Kappa is computed the same way as for classes in the detection project:
    it subtracts the agreement that random flags with the same marginal
    frequencies would have produced.
    """
    m = [[0] * 3 for _ in range(3)]
    for g, my in pairs:
        for (_, _, gv), (_, _, mv) in zip(g.points, my.points):
            if gv == ABSENT and mv == ABSENT:
                continue
            m[gv][mv] += 1
    total = sum(sum(row) for row in m)
    if total == 0:
        return {"matrix": m, "total": 0, "agreement": 0.0, "kappa": 0.0}
    observed = sum(m[i][i] for i in range(3)) / total
    rows = [sum(r) / total for r in m]
    cols = [sum(m[i][j] for i in range(3)) / total for j in range(3)]
    expected = sum(rows[i] * cols[i] for i in range(3))
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return {"matrix": m, "total": total, "agreement": observed, "kappa": kappa}


# --- stage 3: deciding who is compared with whom ---------------------------

def match_people(gt: list[Person], mine: list[Person], mode: str = "bbox",
                 threshold: float = 0.3
                 ) -> tuple[list[tuple[Person, Person]], list[Person], list[Person]]:
    """Matches people within each frame. Hungarian algorithm.

    mode='bbox' (default) -- cost from the IoU of the box built FROM THE
        ANNOTATED POINTS, not from the bbox field in the file: the ground
        truth and a CVAT export fill that field differently, while the points
        are the same. The axis stays independent of the metric -- matched by
        one thing, measured by another.
    mode='oks'  -- cost from OKS, the way COCOeval does it. Measuring with
        the same quantity you matched on is circular: a person with swapped
        sides scores low OKS and risks being left unmatched, so the most
        interesting error disappears from the report instead of entering it.

    Returns the pairs, the unmatched ground-truth people and the unmatched
    people of mine.
    """
    pairs: list[tuple[Person, Person]] = []
    lost_gt: list[Person] = []
    lost_mine: list[Person] = []
    g_by, m_by = by_image(gt), by_image(mine)

    for image in sorted(set(g_by) | set(m_by)):
        gs, ms = g_by.get(image, []), m_by.get(image, [])
        if not gs or not ms:
            lost_gt.extend(gs)
            lost_mine.extend(ms)
            continue
        size = max(len(gs), len(ms))
        score = [[0.0] * size for _ in range(size)]
        for a, g in enumerate(gs):
            for b, m in enumerate(ms):
                if mode == "oks":
                    value = oks(g, m) or 0.0
                else:
                    value = bbox_iou(g.kp_bbox(), m.kp_bbox())
                score[a][b] = value
        cost = [[1.0 - score[a][b] for b in range(size)] for a in range(size)]
        taken_g, taken_m = set(), set()
        for a, b in hungarian(cost):
            if a < len(gs) and b < len(ms) and score[a][b] >= threshold:
                pairs.append((gs[a], ms[b]))
                taken_g.add(a)
                taken_m.add(b)
        lost_gt.extend(g for a, g in enumerate(gs) if a not in taken_g)
        lost_mine.extend(m for b, m in enumerate(ms) if b not in taken_m)
    return pairs, lost_gt, lost_mine


# --- stage 4: everything together ------------------------------------------

def evaluate(gt: list[Person], mine: list[Person], mode: str = "bbox",
             threshold: float = 0.3, pck_mult: float = 1.0) -> dict:
    pairs, lost_gt, lost_mine = match_people(gt, mine, mode, threshold)
    common = [oks(g, m) for g, m in pairs]
    coco = [oks(g, m, "coco") for g, m in pairs]
    common = [v for v in common if v is not None]
    coco = [v for v in coco if v is not None]
    per_joint, overall = pck(pairs, pck_mult)
    flags = flag_confusion(pairs)
    return {
        "matched": len(pairs),
        "gt_people": len(gt),
        "my_people": len(mine),
        "unmatched_gt": len(lost_gt),
        "unmatched_mine": len(lost_mine),
        "oks_common": sum(common) / len(common) if common else 0.0,
        "oks_coco": sum(coco) / len(coco) if coco else 0.0,
        "pck": overall,
        "pck_mult": pck_mult,
        "pck_per_joint": per_joint,
        "flags": flags,
        "pairs": pairs,
        "lost_gt": lost_gt,
        "lost_mine": lost_mine,
    }


# --- self-test cases --------------------------------------------------------

# A stick figure with known coordinates: area is exactly 10000, so s = 100
# and the one-sigma tolerance on the shoulder is 15.8 px. Nothing is
# downloaded or read from disk: the test must run on a bare machine.
TEMPLATE = [
    (100, 30), (108, 24), (92, 24), (118, 28), (82, 28),
    (135, 70), (65, 70), (150, 120), (50, 120), (160, 170), (40, 170),
    (125, 180), (75, 180), (130, 260), (70, 260), (133, 340), (67, 340),
]


def _person(flags: list[int], dx: float = 0.0, sigma_shift: float = 0.0,
            swap: bool = False, ident: int = 1) -> Person:
    pts = [(x + dx + sigma_shift * 100 * K[i], y, flags[i])
           for i, (x, y) in enumerate(TEMPLATE)]
    if swap:
        out = list(pts)
        for i, j in LR_PAIRS:
            out[i], out[j] = pts[j], pts[i]
        pts = out
    p = Person(image="synthetic.jpg", ident=ident, points=pts, area=10000.0,
               bbox=(40.0, 24.0, 120.0, 316.0))
    return p


def _selftest() -> int:
    """Four cases whose answers are known in advance."""
    flags = [VISIBLE] * 17
    flags[3] = flags[4] = HIDDEN     # ears annotated but not visible
    flags[9] = flags[10] = ABSENT    # wrists not annotated at all
    ref = _person(flags)

    cases = [
        ("annotation identical to the ground truth", _person(flags)),
        ("every point offset by exactly s*k_i", _person(flags, sigma_shift=1.0)),
        ("left and right swapped", _person(flags, swap=True)),
        ("same coordinates, every flag says visible",
         _person([ABSENT if f == ABSENT else VISIBLE for f in flags])),
    ]

    print(f"ground truth: area {ref.area:.0f}, s = {math.sqrt(ref.area):.0f} px, "
          f"annotated points {ref.labeled} of 17")
    print(f"one-sigma tolerance: nose {math.sqrt(ref.area) * K[0]:.1f} px, "
          f"shoulder {math.sqrt(ref.area) * K[5]:.1f} px, "
          f"hip {math.sqrt(ref.area) * K[11]:.1f} px")
    print()
    print("| case | OKS | PCK@1 sigma | flag agreement |")
    print("|---|---|---|---|")
    results = []
    for name, mine in cases:
        pairs = [(ref, mine)]
        _, overall = pck(pairs)
        f = flag_confusion(pairs)
        value = oks(ref, mine)
        results.append(value)
        print(f"| {name} | {value:.4f} | {overall:.3f} | {f['agreement']:.3f} |")

    print()
    ok = True
    if abs(results[0] - 1.0) > 1e-9:
        print("FAILED: an identical annotation must score exactly 1.0")
        ok = False
    expected = math.exp(-0.5)
    if abs(results[1] - expected) > 1e-9:
        print(f"FAILED: a one-sigma offset must give exp(-1/2) = {expected:.4f}")
        ok = False
    else:
        print(f"a one-sigma offset gave exp(-1/2) = {expected:.4f} -- "
              "this one is checkable on paper, without a computer")
    if not results[2] < 0.5:
        print("FAILED: swapped sides must wreck the metric")
        ok = False
    if abs(results[3] - 1.0) > 1e-9:
        print("FAILED: a flag-only difference must not change OKS")
        ok = False
    else:
        print("a flag-only difference did not move OKS at all -- "
              "which is exactly why flag agreement is measured separately")
    print("all four cases match" if ok else "there are discrepancies")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true",
                    help="run the four self-test cases")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
