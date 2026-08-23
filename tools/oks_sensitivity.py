#!/usr/bin/env python3
"""Цена пикселя: во что превращается промах в N px (P4d, шаг 2).

Берёт эталон и портит его известным образом — сдвигает КАЖДУЮ точку на
одно и то же число пикселей. Аккуратность при этом всюду одинаковая,
и всё, что видно в таблице, — это работа нормировки OKS.

Три среза одного и того же замера:
  * сдвиг -> OKS в среднем;
  * тот же сдвиг по группам площади: мелкий человек наказывается сильнее;
  * тот же сдвиг по суставам: нос дороже бедра.

    python3 tools/oks_sensitivity.py --ann data/coco/person_keypoints_val2017.json \
        --images data/subset/selection_keypoints.json
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from keypoints import COCO_KEYPOINTS, K, Person, load_coco_keypoints  # noqa: E402
from oks import oks  # noqa: E402

SHIFTS = (2, 3, 5, 8, 12, 20)
JOINTS_SHOWN = (0, 3, 5, 9, 11, 15)


def shifted(p: Person, dx: float) -> Person:
    return Person(image=p.image, ident=p.ident, area=p.area, bbox=p.bbox,
                  points=[(x + dx, y, v) for x, y, v in p.points])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ann", type=Path, required=True)
    ap.add_argument("--images", type=Path, default=None)
    ap.add_argument("--min-kp", type=int, default=8)
    ap.add_argument("--min-area", type=float, default=4000.0)
    ap.add_argument("--probe", type=float, default=8.0,
                    help="сдвиг, на котором делаются разбивки")
    args = ap.parse_args()

    images = None
    if args.images:
        images = set(json.loads(args.images.read_text(encoding="utf-8"))["files"])
    people, _ = load_coco_keypoints(args.ann, images=images,
                                    min_kp=args.min_kp, min_area=args.min_area)
    print(f"людей {len(people)}")

    print()
    print("| сдвиг всех точек, px | средний OKS | медиана | доля людей OKS < 0.5 |")
    print("|---|---|---|---|")
    for px in SHIFTS:
        vals = [oks(p, shifted(p, px)) for p in people]
        vals = [v for v in vals if v is not None]
        low = sum(1 for v in vals if v < 0.5) / len(vals)
        print(f"| {px} | {statistics.mean(vals):.3f} | "
              f"{statistics.median(vals):.3f} | {low:.0%} |")

    areas = sorted(p.area for p in people)
    q1, q3 = areas[len(areas) // 4], areas[3 * len(areas) // 4]
    groups: dict[str, list[float]] = {"мелкие": [], "средние": [], "крупные": []}
    for p in people:
        v = oks(p, shifted(p, args.probe))
        if v is None:
            continue
        name = "мелкие" if p.area <= q1 else ("средние" if p.area <= q3 else "крупные")
        groups[name].append(v)
    print()
    print(f"тот же сдвиг {args.probe:.0f} px по группам площади "
          f"(границы квартилей: {q1:.0f} и {q3:.0f} px²)")
    print("| группа | людей | средний OKS |")
    print("|---|---|---|")
    for name in ("мелкие", "средние", "крупные"):
        print(f"| {name} | {len(groups[name])} | {statistics.mean(groups[name]):.3f} |")

    median_area = statistics.median(areas)
    s = math.sqrt(median_area)
    print()
    print(f"тот же сдвиг {args.probe:.0f} px по суставам, "
          f"человек медианной площади ({median_area:.0f} px²)")
    print("| сустав | сигма | допуск 1σ, px | вклад точки в OKS |")
    print("|---|---|---|---|")
    for i in JOINTS_SHOWN:
        contrib = math.exp(-(args.probe ** 2) / (2 * median_area * K[i] ** 2))
        print(f"| {COCO_KEYPOINTS[i]} | {K[i] / 2:.3f} | {s * K[i]:.1f} | {contrib:.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
