#!/usr/bin/env python3
"""A rehearsal of the pipeline before annotating: a stand-in "own" annotation.

Spending an hour and a half annotating and only then discovering that the
computation crashes or the rendering draws the wrong thing is a bad order of
operations. This script takes the ground truth and corrupts it in a known
way, impersonating a plausible annotator:

  * a shaky hand -- noise of roughly half a sigma on every point;
  * two people get their left and right sides swapped;
  * some invisible points are marked as visible;
  * for people cropped by the image border, some points are invented beyond
    the border -- i.e. the convention opposite to COCO is applied;
  * one person is not annotated at all, one extra person is.

The numbers this produces have nothing to do with real work. The only point
is to confirm that the pipeline runs and the pictures open.

    .venv/bin/python tools/dry_run.py --gt data/coco/person_keypoints_val2017.json \
        --selection data/subset/selection_keypoints.json --out reports/dry_run
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from keypoints import (ABSENT, HIDDEN, K, LR_PAIRS, VISIBLE, Person,  # noqa: E402
                       load_coco_keypoints, save_coco_keypoints)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("reports/dry_run"))
    ap.add_argument("--min-kp", type=int, default=8)
    ap.add_argument("--min-area", type=float, default=4000.0)
    ap.add_argument("--jitter", type=float, default=0.45,
                    help="noise in fractions of the joint sigma")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    images = set(json.loads(args.selection.read_text(encoding="utf-8"))["files"])
    gt, sizes = load_coco_keypoints(args.gt, images=images,
                                    min_kp=args.min_kp, min_area=args.min_area)
    rnd = random.Random(args.seed)

    mine: list[Person] = []
    for n, p in enumerate(gt):
        if n == 5:                                   # missed one person entirely
            continue
        s = math.sqrt(p.area)
        w, h = sizes[p.image]
        pts = []
        for i, (x, y, v) in enumerate(p.points):
            if v == ABSENT:
                if p.touches_edge(w, h) and i >= 13 and rnd.random() < 0.5:
                    # the convention opposite to COCO: invent beyond the border
                    pts.append((p.bbox[0] + p.bbox[2] / 2,
                                p.bbox[1] + p.bbox[3] + 20, HIDDEN))
                else:
                    pts.append((0.0, 0.0, ABSENT))
                continue
            jitter = args.jitter * s * K[i]
            nv = VISIBLE if (v == HIDDEN and rnd.random() < 0.35) else v
            pts.append((x + rnd.gauss(0, jitter), y + rnd.gauss(0, jitter), nv))
        if n in (2, 17):                             # swapped sides on two people
            swapped = list(pts)
            for i, j in LR_PAIRS:
                swapped[i], swapped[j] = pts[j], pts[i]
            pts = swapped
        mine.append(Person(image=p.image, ident=1000 + n, points=pts,
                           area=p.area, bbox=p.bbox))

    extra = gt[0]                                    # one extra person
    mine.append(Person(image=extra.image, ident=9001, area=extra.area,
                       bbox=extra.bbox,
                       points=[(x + 300, y, v) for x, y, v in extra.points]))

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "person_keypoints_fake.json"
    save_coco_keypoints(mine, sizes, dest)
    print("THIS IS A STAND-IN ANNOTATION, NOT A REAL ONE. Its numbers mean nothing.")
    print(f"ground-truth people {len(gt)}, stand-in people {len(mine)}")
    print(f"{dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
