#!/usr/bin/env python3
"""Цена одной фразы в инструкции: три конвенции на одних координатах (P4d, шаг 3).

Берёт эталон и делает из него три «разметки», у которых КООРДИНАТЫ ТОЧЕК
СОВПАДАЮТ С ЭТАЛОНОМ ДО ПИКСЕЛЯ. Отличается только правило, по которому
разметчик решает, ставить точку и какой флаг ей дать:

  1. эталон как есть — верхняя граница, всё совпадает;
  2. «невидимую точку не ставлю» — точки с v=1 не размечаются вовсе;
  3. «ставлю всё, но флаг не различаю» — все размеченные точки помечены
     как видимые.

Смысл третьей строки: аккуратность идеальна, OKS равен единице, и при этом
восьмая часть флагов расходится. Ни один расчёт по координатам этого
не покажет — отсюда и отдельная метрика.

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
    """Невидимую точку не ставлю: остаётся только то, что видно глазом."""
    return Person(image=p.image, ident=p.ident, area=p.area, bbox=p.bbox,
                  points=[(x, y, v) if v == VISIBLE else (0.0, 0.0, ABSENT)
                          for x, y, v in p.points])


def flag_all_visible(p: Person) -> Person:
    """Ставлю всё, но про «видна или нет» не думаю."""
    return Person(image=p.image, ident=p.ident, area=p.area, bbox=p.bbox,
                  points=[(x, y, VISIBLE) if v > ABSENT else (0.0, 0.0, ABSENT)
                          for x, y, v in p.points])


VARIANTS = [
    ("эталон как есть", as_is),
    ("невидимую точку не ставлю", drop_hidden),
    ("ставлю всё, флаг всегда «видна»", flag_all_visible),
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
    print(f"людей {len(people)}; координаты во всех трёх вариантах "
          "совпадают с эталоном до пикселя")
    print()
    print("| конвенция | точек поставлено | OKS по общим точкам | "
          "OKS COCO-style | согласие по флагу | каппа |")
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
