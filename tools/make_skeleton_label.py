#!/usr/bin/env python3
"""Готовый конфиг скелета COCO-17 для CVAT (P4d, шаг 4).

Собирать скелет из семнадцати точек мышью в конфигураторе — самая муторная
часть этапа и при этом чистая настройка: никакой учебной сложности в ней нет,
а ошибиться в порядке точек легко, и ошибка тихая — метрика посчитается
и выдаст мусор. Поэтому конфиг генерируется и вставляется целиком.

Что внутри:
  * 17 подметок с каноническими именами COCO в каноническом порядке;
  * 19 рёбер скелета COCO;
  * координаты узлов взяты из рабочей спецификации в репозитории CVAT
    (serverless/pytorch/mmpose/hrnet32/nuclio/function.yaml) — там тот же
    порядок точек;
  * цвет: левая половина синяя, правая оранжевая, нос зелёный. Путаница
    сторон — самая частая ошибка на скелете, и глазом она видна сразу.

    python3 tools/make_skeleton_label.py --out cvat_coco17_skeleton.json

Дальше содержимое файла целиком вставляется во вкладку Raw редактора меток.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from keypoints import COCO_KEYPOINTS, SKELETON  # noqa: E402

# Узлы в системе координат конфигуратора (0..100), 1-based индексация,
# порядок совпадает с COCO_KEYPOINTS.
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
    print(f"{args.out}: метка «{label['name']}», "
          f"подметок {len(label['sublabels'])}, рёбер {len(SKELETON)}")
    print("порядок точек (он же порядок в экспорте, менять нельзя):")
    for i, name in enumerate(COCO_KEYPOINTS):
        print(f"  {i + 1:2d}. {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
