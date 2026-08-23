#!/usr/bin/env python3
"""Checks the export numerically before any metric is computed.

Three things break silently and produce not an error but a wrong number:

  1. POINT ORDER. If the sublabel order in the skeleton config drifts away
     from COCO, the metric computes without a single warning -- it simply
     compares a left shoulder against a right elbow. The symptom: OKS near
     zero on skeletons that look alike.
  2. VISIBILITY FLAG. If occluded and outside were never pressed, every
     point in the file is v=2 and flag agreement becomes meaningless. The
     histogram catches it: a v=2 share near 100% does not mean "everything
     was visible", it means "the flag was never set".
  3. FRAME COMPOSITION. An export can come from the wrong task, or miss
     frames. Checked against the selection manifest.

    python3 tools/check_export.py --mine annotation/person_keypoints_default.json \
        --selection data/subset/selection_keypoints.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from keypoints import COCO_KEYPOINTS, load_coco_keypoints  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mine", type=Path, required=True)
    ap.add_argument("--selection", type=Path, default=None)
    args = ap.parse_args()

    raw = json.loads(args.mine.read_text(encoding="utf-8"))
    problems = 0

    cats = [c for c in raw.get("categories", []) if c.get("keypoints")]
    if not cats:
        print("NO category with a keypoints field -- this is not a COCO Keypoints export")
        return 1
    names = cats[0]["keypoints"]
    print(f"category \"{cats[0].get('name')}\", keypoints in it {len(names)}")
    if names == COCO_KEYPOINTS:
        print("point order matches COCO -- safe to compute")
    else:
        problems += 1
        print("POINT ORDER DIFFERS FROM COCO. Do not compute, the metric would be garbage.")
        for i, (mine, ref) in enumerate(zip(names, COCO_KEYPOINTS)):
            if mine != ref:
                print(f"  position {i + 1}: got \"{mine}\", expected \"{ref}\"")
        if len(names) != len(COCO_KEYPOINTS):
            print(f"  and {len(names)} keypoints instead of {len(COCO_KEYPOINTS)}")

    people, sizes = load_coco_keypoints(args.mine)
    frames = {p.image for p in people}
    print(f"frames in the export {len(raw.get('images', []))}, "
          f"skeletons {len(people)}, frames with skeletons {len(frames)}")
    if not people:
        print("ZERO SKELETONS. Either the task is not marked completed, or the "
              "export came from an empty task.")
        return 1

    flags: Counter = Counter()
    zero_coord = 0
    for p in people:
        for x, y, v in p.points:
            flags[v] += 1
            if v > 0 and x == 0 and y == 0:
                zero_coord += 1
    slots = sum(flags.values())
    print(f"keypoint slots {slots}: v=0 {flags[0]} ({flags[0] / slots:.0%}), "
          f"v=1 {flags[1]} ({flags[1] / slots:.0%}), "
          f"v=2 {flags[2]} ({flags[2] / slots:.0%})")
    if flags[1] == 0 and flags[0] == 0:
        problems += 1
        print("EVERY POINT IS v=2. The occluded and outside flags were never set, "
              "so there is nothing to measure flag agreement on.")
    if zero_coord:
        problems += 1
        print(f"{zero_coord} points sit at (0,0) with v>0 -- that is what a point "
              "left behind in the template position looks like.")

    has_bbox = sum(1 for p in people if p.bbox[2] > 0 and p.bbox[3] > 0)
    print(f"with a non-empty bbox {has_bbox} of {len(people)} "
          "(matching uses the keypoint box anyway, this is informational)")

    if args.selection:
        want = set(json.loads(args.selection.read_text(encoding="utf-8"))["files"])
        missing = sorted(want - frames)
        extra = sorted(frames - want)
        if missing:
            problems += 1
            print(f"NO SKELETONS on {len(missing)} selected frames: "
                  f"{', '.join(missing[:5])}{' …' if len(missing) > 5 else ''}")
        if extra:
            print(f"extra frames outside the selection: {len(extra)}")
        if not missing and not extra:
            print(f"frame composition matches the selection manifest ({len(want)})")

    print()
    print("check passed, safe to compute" if problems == 0
          else f"problems: {problems}. Fix them first, compute after.")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
