#!/usr/bin/env python3
"""What the ground truth contains: flags, areas, the image border.

Without --images it runs over all of val2017 -- that is how the COCO
convention itself gets measured, and that is what is worth knowing before
annotating. With --images it is restricted to one's own subset: look at that
after annotating, before it is a hint.

    python3 tools/keypoint_stats.py --ann data/coco/person_keypoints_val2017.json
    python3 tools/keypoint_stats.py --ann ... --images data/subset/selection_keypoints.json
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from keypoints import ABSENT, COCO_KEYPOINTS, load_coco_keypoints  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ann", type=Path, required=True)
    ap.add_argument("--images", type=Path, default=None,
                    help="selection_keypoints.json -- restrict to that subset")
    ap.add_argument("--min-kp", type=int, default=8)
    ap.add_argument("--min-area", type=float, default=4000.0)
    args = ap.parse_args()

    images = None
    if args.images:
        images = set(json.loads(args.images.read_text(encoding="utf-8"))["files"])

    people, sizes = load_coco_keypoints(args.ann, images=images,
                                        min_kp=args.min_kp, min_area=args.min_area)
    if not people:
        raise SystemExit("no person passed the filters")

    frames = {p.image for p in people}
    print(f"frames {len(frames)}, people {len(people)} "
          f"(filter: >={args.min_kp} annotated points, area >={args.min_area:.0f})")

    areas = sorted(p.area for p in people)
    print(f"area: min {areas[0]:.0f}, median {statistics.median(areas):.0f}, "
          f"max {areas[-1]:.0f}")

    flags: Counter = Counter()
    per_joint: dict[int, Counter] = {i: Counter() for i in range(17)}
    for p in people:
        for i, (_, _, v) in enumerate(p.points):
            flags[v] += 1
            per_joint[i][v] += 1
    slots = sum(flags.values())
    print(f"keypoint slots {slots}: v=0 not annotated {flags[0]} ({flags[0] / slots:.0%}), "
          f"v=1 annotated but not visible {flags[1]} ({flags[1] / slots:.0%}), "
          f"v=2 visible {flags[2]} ({flags[2] / slots:.0%})")
    print(f"annotated points in total {flags[1] + flags[2]}")

    print()
    print("| joint | v=2 visible | v=1 not visible | v=0 not annotated |")
    print("|---|---|---|---|")
    order = sorted(range(17), key=lambda i: -per_joint[i][ABSENT])
    for i in order:
        c = per_joint[i]
        n = sum(c.values())
        print(f"| {COCO_KEYPOINTS[i]} | {c[2] / n:.0%} | {c[1] / n:.0%} | {c[0] / n:.0%} |")

    # The image border. The same question the tracking stage settled for
    # MOT17 boxes, and the answer here is the opposite one -- which is why it
    # is measured rather than assumed.
    outside = 0
    on_border = 0
    labeled = 0
    for p in people:
        w, h = sizes.get(p.image, (0, 0))
        if not w:
            continue
        for x, y, v in p.points:
            if v == ABSENT:
                continue
            labeled += 1
            if x < 0 or y < 0 or x > w or y > h:
                outside += 1
            elif min(x, y, w - x, h - y) <= 0.5:
                on_border += 1
    cut = [p for p in people if p.touches_edge(*sizes.get(p.image, (0, 0)))]
    whole = [p for p in people if not p.touches_edge(*sizes.get(p.image, (0, 0)))]
    print()
    print(f"image border: annotated points {labeled}, "
          f"of them outside the image {outside}, "
          f"pinned to the border (<=0.5 px) {on_border}")
    print(f"people whose box touches the image border: {len(cut)} of {len(people)}")

    # Where a point cut off by the border ends up: comparing the cropped
    # people with the fully visible ones answers that with a number.
    print()
    print("| people | count | points not annotated, mean | "
          "most often not annotated |")
    print("|---|---|---|---|")
    for name, group in (("cropped by the image border", cut), ("fully inside the frame", whole)):
        if not group:
            continue
        miss: Counter = Counter()
        for p in group:
            for i, (_, _, v) in enumerate(p.points):
                if v == ABSENT:
                    miss[i] += 1
        top = sorted(range(17), key=lambda i: -miss[i])[:3]
        names = ", ".join(f"{COCO_KEYPOINTS[i]} {miss[i] / len(group):.0%}" for i in top)
        print(f"| {name} | {len(group)} | "
              f"{sum(miss.values()) / len(group):.1f} of 17 | {names} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
