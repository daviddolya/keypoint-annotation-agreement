#!/usr/bin/env python3
"""Репетиция конвейера до разметки: подставная «своя» разметка (P4d, шаг 5).

Полтора часа разметки, а потом выясняется, что расчёт падает или отрисовка
рисует не то, — плохой порядок. Скрипт берёт эталон и портит его известным
образом, изображая правдоподобного разметчика:

  * дрожь руки — шум примерно в половину сигмы на каждой точке;
  * двум людям перепутаны стороны;
  * части невидимых точек проставлен флаг «видна»;
  * у людей, обрезанных краем кадра, часть точек досочинена за границу —
    то есть применена конвенция, противоположная COCO;
  * один человек не размечен вовсе, один лишний размечен.

Числа, которые получатся, к твоей работе отношения не имеют. Смысл один:
убедиться, что конвейер работает и картинки открываются.

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
                    help="шум в долях сигмы сустава")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    images = set(json.loads(args.selection.read_text(encoding="utf-8"))["files"])
    gt, sizes = load_coco_keypoints(args.gt, images=images,
                                    min_kp=args.min_kp, min_area=args.min_area)
    rnd = random.Random(args.seed)

    mine: list[Person] = []
    for n, p in enumerate(gt):
        if n == 5:                                   # одного не заметил вовсе
            continue
        s = math.sqrt(p.area)
        w, h = sizes[p.image]
        pts = []
        for i, (x, y, v) in enumerate(p.points):
            if v == ABSENT:
                if p.touches_edge(w, h) and i >= 13 and rnd.random() < 0.5:
                    # конвенция, противоположная COCO: досочиняю за краем кадра
                    pts.append((p.bbox[0] + p.bbox[2] / 2,
                                p.bbox[1] + p.bbox[3] + 20, HIDDEN))
                else:
                    pts.append((0.0, 0.0, ABSENT))
                continue
            jitter = args.jitter * s * K[i]
            nv = VISIBLE if (v == HIDDEN and rnd.random() < 0.35) else v
            pts.append((x + rnd.gauss(0, jitter), y + rnd.gauss(0, jitter), nv))
        if n in (2, 17):                             # двоим перепутал стороны
            swapped = list(pts)
            for i, j in LR_PAIRS:
                swapped[i], swapped[j] = pts[j], pts[i]
            pts = swapped
        mine.append(Person(image=p.image, ident=1000 + n, points=pts,
                           area=p.area, bbox=p.bbox))

    extra = gt[0]                                    # один лишний
    mine.append(Person(image=extra.image, ident=9001, area=extra.area,
                       bbox=extra.bbox,
                       points=[(x + 300, y, v) for x, y, v in extra.points]))

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "person_keypoints_fake.json"
    save_coco_keypoints(mine, sizes, dest)
    print("ЭТО ПОДСТАВНАЯ РАЗМЕТКА, А НЕ ТВОЯ. Числа по ней ничего не значат.")
    print(f"эталонных людей {len(gt)}, подставных {len(mine)}")
    print(f"{dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
