#!/usr/bin/env python3
"""Selects the frames to annotate and downloads them.

A frame is taken if it holds three or four people, each with enough
annotated keypoints in the ground truth and none of them microscopic.
The rules, and why they are what they are:

  1. EXACTLY 3-4 PEOPLE. A frame with a single person does not exercise
     matching: there is nothing to match against, any figure of mine pairs
     up with the only ground-truth one. A frame with eight is half an hour
     of work and a crowd the ground truth itself annotates every other time.
     Three or four give both matching and people occluding each other, i.e.
     genuinely disputable cases.
  2. MINIMUM ANNOTATED POINTS. A person with three annotated points in the
     ground truth is not a skeleton and there is nothing to compare.
  3. LOWER AREA THRESHOLD. OKS normalises by scale: on a small person the
     same pixel error costs three times as much. Annotating tiny people
     measures the resolution of the picture, not the accuracy of the hand.
  4. OUTSIDE THE DETECTION SUBSET, when its selection.json is passed in:
     the same photographs annotated a third time read as repetition rather
     than as range.

How many people are in each frame and where their points sit is NOT
printed: the annotation is done blind, otherwise there is no agreement to
measure. --stats shows the distribution, and it is meant to be run after
the annotation, not before.

    python3 select_people.py --ann data/coco/person_keypoints_val2017.json \
        --out data/subset --count 14 \
        --exclude ../detection-annotation-quality/data/subset/selection.json
"""

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path


def fetch(url: str, dest: Path, attempts: int = 4) -> int:
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, dest.open("wb") as f:
                f.write(r.read())
            return dest.stat().st_size
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if dest.exists():
                dest.unlink()
            if attempt == attempts:
                raise
            print(f"  {dest.name}: attempt {attempt} failed ({e})")
            time.sleep(2 * attempt)
    raise RuntimeError("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ann", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=14)
    ap.add_argument("--min-people", type=int, default=3)
    ap.add_argument("--max-people", type=int, default=4)
    ap.add_argument("--min-kp", type=int, default=8)
    ap.add_argument("--min-area", type=float, default=4000.0)
    ap.add_argument("--exclude", type=Path, default=None,
                    help="selection.json of another stage: skip those frames")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--stats", action="store_true",
                    help="ground-truth distribution -- look at it AFTER annotating")
    args = ap.parse_args()

    data = json.loads(args.ann.read_text(encoding="utf-8"))
    images = {img["id"]: img for img in data["images"]}

    skip: set[str] = set()
    if args.exclude and args.exclude.exists():
        skip = set(json.loads(args.exclude.read_text(encoding="utf-8"))["files"])

    per_image: dict[int, list] = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        if ann.get("num_keypoints", 0) < args.min_kp or ann["area"] < args.min_area:
            continue
        per_image.setdefault(ann["image_id"], []).append(ann)

    pool = [i for i, anns in per_image.items()
            if args.min_people <= len(anns) <= args.max_people
            and images[i]["file_name"] not in skip]
    if len(pool) < args.count:
        raise SystemExit(f"only {len(pool)} candidates, {args.count} requested")

    rnd = random.Random(args.seed)
    rnd.shuffle(pool)
    picked = sorted(pool[:args.count], key=lambda i: images[i]["file_name"])

    frames = args.out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    total = 0
    for i in picked:
        img = images[i]
        dest = frames / img["file_name"]
        if dest.exists():
            total += dest.stat().st_size
            continue
        total += fetch(img["coco_url"], dest)

    manifest = {
        "source": "COCO val2017, person_keypoints",
        "task": "skeleton annotation, 17 COCO keypoints",
        "filters": {"people_per_frame": [args.min_people, args.max_people],
                    "min_keypoints": args.min_kp, "min_area": args.min_area,
                    "excluded_subset": str(args.exclude) if skip else None},
        "seed": args.seed,
        "count": len(picked),
        "files": [images[i]["file_name"] for i in picked],
    }
    (args.out / "selection_keypoints.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"candidates {len(pool)}, frames selected {len(picked)}, {total / 1e6:.2f} MB")
    print(f"frames: {frames}")
    print(f"manifest: {args.out / 'selection_keypoints.json'}")

    if args.stats:
        people = [a for i in picked for a in per_image[i]]
        flags: Counter = Counter()
        for a in people:
            for j in range(17):
                flags[a["keypoints"][3 * j + 2]] += 1
        slots = sum(flags.values())
        areas = sorted(a["area"] for a in people)
        print()
        print(f"[stats] people {len(people)}, "
              f"{len(people) / len(picked):.1f} per frame")
        print(f"[stats] area: min {areas[0]:.0f}, "
              f"median {areas[len(areas) // 2]:.0f}, max {areas[-1]:.0f}")
        print(f"[stats] keypoint slots {slots}: v=0 {flags[0]}, "
              f"v=1 {flags[1]}, v=2 {flags[2]}; annotated {flags[1] + flags[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
