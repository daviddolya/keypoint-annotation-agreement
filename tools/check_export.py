#!/usr/bin/env python3
"""Проверка экспорта числом до того, как считать метрики (P4d, шаг 4).

Три вещи ломаются молча и дают не ошибку, а неправильное число:

  1. ПОРЯДОК ТОЧЕК. Если в конфиге скелета порядок подметок разошёлся
     с COCO, метрика посчитается без единого предупреждения — просто
     сравнит левое плечо с правым локтем. Симптом на выходе: OKS около
     нуля при визуально похожих скелетах.
  2. ФЛАГ ВИДИМОСТИ. Если ни разу не нажаты occluded и outside, в файле
     все точки будут v=2, и согласие по флагу окажется бессмысленным.
     Проверяется гистограммой: доля v=2 около 100% — это не «всё видно»,
     а «флаг не проставлялся».
  3. СОСТАВ КАДРОВ. Экспорт может уехать не с той задачи или не со всеми
     кадрами. Сверяется с манифестом отбора.

    python3 tools/check_export.py --mine annotation/my_labels/person_keypoints_default.json \
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
        print("НЕТ категории с полем keypoints — экспорт сделан не в COCO Keypoints")
        return 1
    names = cats[0]["keypoints"]
    print(f"категория «{cats[0].get('name')}», точек в ней {len(names)}")
    if names == COCO_KEYPOINTS:
        print("порядок точек совпадает с COCO — можно считать")
    else:
        problems += 1
        print("ПОРЯДОК ТОЧЕК РАЗОШЁЛСЯ С COCO. Считать нельзя, метрика будет мусором.")
        for i, (mine, ref) in enumerate(zip(names, COCO_KEYPOINTS)):
            if mine != ref:
                print(f"  позиция {i + 1}: у тебя «{mine}», ожидается «{ref}»")
        if len(names) != len(COCO_KEYPOINTS):
            print(f"  и точек {len(names)} вместо {len(COCO_KEYPOINTS)}")

    people, sizes = load_coco_keypoints(args.mine)
    frames = {p.image for p in people}
    print(f"кадров в экспорте {len(raw.get('images', []))}, "
          f"скелетов {len(people)}, кадров со скелетами {len(frames)}")
    if not people:
        print("СКЕЛЕТОВ НОЛЬ. Задача не переведена в completed либо экспорт "
              "сделан из пустой задачи.")
        return 1

    flags: Counter = Counter()
    zero_coord = 0
    for p in people:
        for x, y, v in p.points:
            flags[v] += 1
            if v > 0 and x == 0 and y == 0:
                zero_coord += 1
    slots = sum(flags.values())
    print(f"слотов точек {slots}: v=0 {flags[0]} ({flags[0] / slots:.0%}), "
          f"v=1 {flags[1]} ({flags[1] / slots:.0%}), "
          f"v=2 {flags[2]} ({flags[2] / slots:.0%})")
    if flags[1] == 0 and flags[0] == 0:
        problems += 1
        print("ВСЕ ТОЧКИ v=2. Флаги occluded и outside ни разу не проставлены — "
              "согласие по флагу считать не на чем.")
    if zero_coord:
        problems += 1
        print(f"{zero_coord} точек имеют координату (0,0) при v>0 — "
              "так выглядит точка, которую забыли перетащить из шаблона.")

    has_bbox = sum(1 for p in people if p.bbox[2] > 0 and p.bbox[3] > 0)
    print(f"со непустым bbox {has_bbox} из {len(people)} "
          "(сопоставление всё равно идёт по рамке из точек, это справочно)")

    if args.selection:
        want = set(json.loads(args.selection.read_text(encoding="utf-8"))["files"])
        missing = sorted(want - frames)
        extra = sorted(frames - want)
        if missing:
            problems += 1
            print(f"НЕТ СКЕЛЕТОВ на {len(missing)} кадрах отбора: "
                  f"{', '.join(missing[:5])}{' …' if len(missing) > 5 else ''}")
        if extra:
            print(f"лишние кадры вне отбора: {len(extra)}")
        if not missing and not extra:
            print(f"состав кадров совпадает с манифестом отбора ({len(want)})")

    print()
    print("проверка пройдена, можно считать" if problems == 0
          else f"проблем: {problems}. Сначала чинить, потом считать.")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
