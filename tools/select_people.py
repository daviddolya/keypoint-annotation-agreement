#!/usr/bin/env python3
"""Отбор кадров под разметку скелетов и их загрузка (P4d, шаг 0).

Кадр берётся, если в нём три-четыре человека, у каждого в эталоне размечено
достаточно точек и он не микроскопический. Правила и почему именно они:

  1. РОВНО 3-4 ЧЕЛОВЕКА. Кадр с одним человеком не проверяет сопоставление:
     сопоставлять не с чем, любая моя фигура встанет в пару с единственной
     эталонной. Кадр с восемью — это полчаса работы и толпа, где эталон сам
     размечает через раз. Три-четыре дают и сопоставление, и перекрытия
     людей друг другом, то есть настоящие спорные случаи.
  2. МИНИМУМ РАЗМЕЧЕННЫХ ТОЧЕК. Человек, у которого в эталоне размечено
     три точки, — это не скелет, и сравнивать с ним нечего.
  3. НИЖНИЙ ПОРОГ ПЛОЩАДИ. OKS нормируется на масштаб: на мелком человеке
     тот же промах в пикселях стоит втрое дороже. Размечать мелочь значит
     мерить не свою аккуратность, а разрешение картинки (см. шаг 2).
  4. ВНЕ СОТНИ P2, если передан её selection.json: одни и те же снимки,
     размеченные третий раз, читаются как повтор, а не как широта.

Сколько людей в каждом кадре и где у них точки — НЕ печатается: разметка
идёт вслепую, иначе согласие считать не на чем. Распределение показывает
--stats, и запускать его следует после разметки, а не до.

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
            print(f"  {dest.name}: попытка {attempt} сорвалась ({e})")
            time.sleep(2 * attempt)
    raise RuntimeError("недостижимо")


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
                    help="selection.json другого этапа: эти кадры не брать")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--stats", action="store_true",
                    help="эталонное распределение — смотреть ПОСЛЕ разметки")
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
        raise SystemExit(f"кандидатов всего {len(pool)}, просили {args.count}")

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
        "task": "разметка скелетов, 17 точек COCO",
        "filters": {"people_per_frame": [args.min_people, args.max_people],
                    "min_keypoints": args.min_kp, "min_area": args.min_area,
                    "excluded_subset": str(args.exclude) if skip else None},
        "seed": args.seed,
        "count": len(picked),
        "files": [images[i]["file_name"] for i in picked],
    }
    (args.out / "selection_keypoints.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"кандидатов {len(pool)}, отобрано кадров {len(picked)}, {total / 1e6:.2f} МБ")
    print(f"кадры: {frames}")
    print(f"манифест: {args.out / 'selection_keypoints.json'}")

    if args.stats:
        people = [a for i in picked for a in per_image[i]]
        flags: Counter = Counter()
        for a in people:
            for j in range(17):
                flags[a["keypoints"][3 * j + 2]] += 1
        slots = sum(flags.values())
        areas = sorted(a["area"] for a in people)
        print()
        print(f"[stats] людей {len(people)}, "
              f"по {len(people) / len(picked):.1f} на кадр")
        print(f"[stats] площадь: мин {areas[0]:.0f}, "
              f"медиана {areas[len(areas) // 2]:.0f}, макс {areas[-1]:.0f}")
        print(f"[stats] слотов точек {slots}: v=0 {flags[0]}, "
              f"v=1 {flags[1]}, v=2 {flags[2]}; размечено {flags[1] + flags[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
