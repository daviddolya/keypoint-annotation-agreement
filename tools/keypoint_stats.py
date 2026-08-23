#!/usr/bin/env python3
"""Что лежит в эталоне: флаги, площади, край кадра (P4d, шаги 2-3).

Без --images считает по всему val2017 — так меряется конвенция COCO вообще,
и это то, что нужно знать до разметки. С --images ограничивается своим
подмножеством: смотреть после разметки, до неё это подсказка.

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
                    help="selection_keypoints.json — ограничить своим набором")
    ap.add_argument("--min-kp", type=int, default=8)
    ap.add_argument("--min-area", type=float, default=4000.0)
    args = ap.parse_args()

    images = None
    if args.images:
        images = set(json.loads(args.images.read_text(encoding="utf-8"))["files"])

    people, sizes = load_coco_keypoints(args.ann, images=images,
                                        min_kp=args.min_kp, min_area=args.min_area)
    if not people:
        raise SystemExit("под фильтры не попал ни один человек")

    frames = {p.image for p in people}
    print(f"кадров {len(frames)}, людей {len(people)} "
          f"(фильтр: размечено ≥{args.min_kp} точек, площадь ≥{args.min_area:.0f})")

    areas = sorted(p.area for p in people)
    print(f"площадь: мин {areas[0]:.0f}, медиана {statistics.median(areas):.0f}, "
          f"макс {areas[-1]:.0f}")

    flags: Counter = Counter()
    per_joint: dict[int, Counter] = {i: Counter() for i in range(17)}
    for p in people:
        for i, (_, _, v) in enumerate(p.points):
            flags[v] += 1
            per_joint[i][v] += 1
    slots = sum(flags.values())
    print(f"слотов точек {slots}: v=0 не размечена {flags[0]} ({flags[0] / slots:.0%}), "
          f"v=1 размечена и не видна {flags[1]} ({flags[1] / slots:.0%}), "
          f"v=2 видна {flags[2]} ({flags[2] / slots:.0%})")
    print(f"размечено точек всего {flags[1] + flags[2]}")

    print()
    print("| сустав | v=2 видна | v=1 не видна | v=0 не размечена |")
    print("|---|---|---|---|")
    order = sorted(range(17), key=lambda i: -per_joint[i][ABSENT])
    for i in order:
        c = per_joint[i]
        n = sum(c.values())
        print(f"| {COCO_KEYPOINTS[i]} | {c[2] / n:.0%} | {c[1] / n:.0%} | {c[0] / n:.0%} |")

    # Край кадра. Тот же вопрос, что решался на A3 для боксов MOT17,
    # и ответ здесь противоположный — поэтому меряется, а не предполагается.
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
    print(f"край кадра: размеченных точек {labeled}, "
          f"из них за пределами изображения {outside}, "
          f"прижато к границе (≤0.5 px) {on_border}")
    print(f"людей, чья рамка упирается в край кадра: {len(cut)} из {len(people)}")

    # Куда девается точка, которой не видно из-за края кадра: сравнение
    # обрезанных и целиком попавших в кадр отвечает на это числом.
    print()
    print("| люди | сколько их | не размечено точек в среднем | "
          "чаще всего не размечены |")
    print("|---|---|---|---|")
    for name, group in (("обрезаны краем кадра", cut), ("целиком в кадре", whole)):
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
              f"{sum(miss.values()) / len(group):.1f} из 17 | {names} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
