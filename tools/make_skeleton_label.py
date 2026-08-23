#!/usr/bin/env python3
"""A ready-made COCO-17 skeleton label config for CVAT.

Assembling a seventeen-point skeleton by hand in the label configurator is
the most tedious part of the setup and pure configuration at that: the point
order is easy to get wrong, and the mistake is silent -- the metric will
compute happily and return garbage. So the config is generated and pasted in
as a whole.

What is inside:
  * 17 sublabels with the canonical COCO names in the canonical order;
  * the 19 COCO skeleton edges;
  * node coordinates taken from a working spec in the CVAT repository
    (serverless/pytorch/mmpose/hrnet32/nuclio/function.yaml), which uses the
    same point order;
  * colour: the left half blue, the right half orange, the nose green.
    Swapped sides are the most common skeleton mistake, and this makes them
    visible at a glance.

    python3 tools/make_skeleton_label.py --out cvat_coco17_skeleton.json

The contents of that file are then pasted into the Raw tab of the label editor.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from keypoints import COCO_KEYPOINTS, SKELETON  # noqa: E402

# Nodes in the configurator coordinate system (0..100), 1-based indexing,
# order identical to COCO_KEYPOINTS.
NODES = [
    (48.876, 9.485),    # nose
    (51.229, 7.637),    # left_eye
    (47.195, 7.637),    # right_eye
    (54.254, 7.805),    # left_ear
    (44.170, 7.805),    # right_ear
    (60.809, 19.905),   # left_shoulder
    (37.784, 20.410),   # right_shoulder
    (63.834, 34.023),   # left_elbow
    (35.935, 34.359),   # right_elbow
    (66.859, 47.132),   # left_wrist
    (33.918, 47.468),   # right_wrist
    (57.111, 49.653),   # left_hip
    (44.002, 50.158),   # right_hip
    (58.120, 71.166),   # left_knee
    (44.338, 70.662),   # right_knee
    (57.784, 87.973),   # left_ankle
    (46.187, 92.511),   # right_ankle
]

LEFT = "#1f77b4"
RIGHT = "#ff7f0e"
CENTER = "#2ca02c"


def color_of(name: str) -> str:
    if name.startswith("left_"):
        return LEFT
    if name.startswith("right_"):
        return RIGHT
    return CENTER


def build(label_name: str = "person") -> list[dict]:
    edges = []
    for a, b in SKELETON:
        (x1, y1), (x2, y2) = NODES[a], NODES[b]
        edges.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" '
            f'stroke-width="0.5" data-type="edge" '
            f'data-node-from="{a + 1}" data-node-to="{b + 1}"></line>'
        )
    nodes = []
    for i, name in enumerate(COCO_KEYPOINTS):
        cx, cy = NODES[i]
        nodes.append(
            f'<circle r="1.5" stroke="black" fill="{color_of(name)}" '
            f'cx="{cx}" cy="{cy}" stroke-width="0.1" '
            f'data-type="element node" data-element-id="{i + 1}" '
            f'data-node-id="{i + 1}" data-label-name="{name}"></circle>'
        )
    return [{
        "name": label_name,
        "type": "skeleton",
        "attributes": [],
        "svg": "".join(edges + nodes),
        "sublabels": [
            {"name": name, "type": "points", "color": color_of(name), "attributes": []}
            for name in COCO_KEYPOINTS
        ],
    }]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("cvat_coco17_skeleton.json"))
    ap.add_argument("--label", default="person")
    args = ap.parse_args()

    spec = build(args.label)
    args.out.write_text(json.dumps(spec, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    label = spec[0]
    print(f"{args.out}: label \"{label['name']}\", "
          f"sublabels {len(label['sublabels'])}, edges {len(SKELETON)}")
    print("point order (the same order the export uses -- do not change it):")
    for i, name in enumerate(COCO_KEYPOINTS):
        print(f"  {i + 1:2d}. {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
