#!/usr/bin/env python3
"""The price of one sentence in the guidelines: three conventions, one set
of coordinates.

Takes the ground truth and builds three "annotations" out of it whose POINT
COORDINATES MATCH THE GROUND TRUTH TO THE PIXEL. The only difference is the
rule by which the annotator decides whether to place a point and which flag
to give it:

  1. ground truth as is -- the upper bound, everything matches;
  2. "I do not place a point I cannot see" -- v=1 points are not annotated;
  3. "I place everything but do not distinguish the flag" -- every annotated
     point is marked visible.

The point of the third row: accuracy is perfect, OKS equals one, and yet an
eighth of the flags disagree. No coordinate-based computation will show it,
which is exactly why flag agreement is a metric of its own.

    python3 tools/visibility_baseline.py --ann data/coco/person_keypoints_val2017.json \
        --images data/subset/selection_keypoints.json
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from keypoints import ABSENT, VISIBLE, Person, load_coco_keypoints  # noqa: E402
from oks import flag_confusion, oks  # noqa: E402


def as_is(p: Person) -> Person:
    return p


def drop_hidden(p: Person) -> Person:
    """I do not place an invisible point: only what the eye sees remains."""
    return Person(image=p.image, ident=p.ident, area=p.area, bbox=p.bbox,
                  points=[(x, y, v) if v == VISIBLE else (0.0, 0.0, ABSENT)
                          for x, y, v in p.points])


def flag_all_visible(p: Person) -> Person:
    """I place everything and never think about visible-or-not."""
    return Person(image=p.image, ident=p.ident, area=p.area, bbox=p.bbox,
                  points=[(x, y, VISIBLE) if v > ABSENT else (0.0, 0.0, ABSENT)
                          for x, y, v in p.points])


VARIANTS = [
    ("ground truth as is", as_is),
    ("I do not place an invisible point", drop_hidden),
    ("I place everything, flag always visible", flag_all_visible),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ann", type=Path, required=True)
    ap.add_argument("--images", type=Path, default=None)
    ap.add_argument("--min-kp", type=int, default=8)
    ap.add_argument("--min-area", type=float, default=4000.0)
    args = ap.parse_args()

    images = None
    if args.images:
        images = set(json.loads(args.images.read_text(encoding="utf-8"))["files"])
    people, _ = load_coco_keypoints(args.ann, images=images,
                                    min_kp=args.min_kp, min_area=args.min_area)
    print(f"people {len(people)}; in all three variants the coordinates "
          "match the ground truth to the pixel")
    print()
    print("| convention | points placed | OKS over common points | "
          "OKS COCO-style | flag agreement | kappa |")
    print("|---|---|---|---|---|---|")
    for name, make in VARIANTS:
        mine = [make(p) for p in people]
        pairs = list(zip(people, mine))
        placed = sum(m.labeled for m in mine)
        common = [v for v in (oks(g, m) for g, m in pairs) if v is not None]
        coco = [v for v in (oks(g, m, "coco") for g, m in pairs) if v is not None]
        f = flag_confusion(pairs)
        print(f"| {name} | {placed} | {statistics.mean(common):.3f} | "
              f"{statistics.mean(coco):.3f} | {f['agreement']:.3f} | {f['kappa']:.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
