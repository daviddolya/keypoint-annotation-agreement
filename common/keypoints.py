#!/usr/bin/env python3
"""Формат COCO Keypoints: эталон и экспорт CVAT читаются одним загрузчиком (P4d).

Одна аннотация человека — это 17 точек в жёстко заданном порядке, плоским
списком из 51 числа: x1, y1, v1, x2, y2, v2, ... Порядок нельзя менять и
нельзя выбирать свой: индекс 5 — это left_shoulder и ни что иное, а если
в твоей разметке на этом месте окажется правое плечо, метрика посчитается
без единой ошибки и выдаст мусор.

Флаг v принимает три значения и означает не «качество точки», а её статус:

    v=0  точка не размечена вовсе. Координаты в файле стоят нулевые
         и в расчёте не участвуют.
    v=1  точка размечена, но не видна: закрыта телом, другим человеком,
         предметом. Координаты осмысленные — это догадка разметчика о том,
         где сустав находится.
    v=2  точка размечена и видна.

Экспорт CVAT (формат COCO Keypoints 1.0) кладёт файл с тем же именем,
что у эталона — annotations/person_keypoints_<subset>.json — и с той же
семантикой флага: outside -> 0, occluded -> 1, ничего -> 2
(cvat/apps/dataset_manager/bindings.py). Поэтому загрузчик здесь один.

Зависимостей нет.
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Сигмы COCO: разброс, с которым люди размечают этот сустав повторно.
# Это не настройка, а замер — см. докстроку oks.py.
SIGMAS = [
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
]
K = [2 * s for s in SIGMAS]

# Рёбра скелета COCO, 19 штук, индексы точек с нуля.
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
]

# Пары «левое — правое». Нужны для контрольного случая с перестановкой сторон
# и для разбора расхождений: путаница сторон — самая частая ошибка на скелете.
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

ABSENT, HIDDEN, VISIBLE = 0, 1, 2


@dataclass
class Person:
    """Один размеченный человек: 17 точек плюс масштаб."""

    image: str                              # имя файла кадра
    ident: int                              # id аннотации в исходном файле
    points: list[tuple[float, float, int]]  # ровно 17 штук, (x, y, v)
    area: float                             # площадь сегментации — масштаб для OKS
    bbox: tuple[float, float, float, float]  # x, y, w, h

    @property
    def labeled(self) -> int:
        """Сколько точек размечено (v > 0). То же, что num_keypoints в COCO."""
        return sum(1 for _, _, v in self.points if v > ABSENT)

    @property
    def visible(self) -> int:
        return sum(1 for _, _, v in self.points if v == VISIBLE)

    def kp_bbox(self) -> tuple[float, float, float, float]:
        """Рамка по размеченным точкам. Нужна, когда bbox экспорта не доверяем."""
        xs = [x for x, _, v in self.points if v > ABSENT]
        ys = [y for _, y, v in self.points if v > ABSENT]
        if not xs:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    def touches_edge(self, width: float, height: float, margin: float = 1.0) -> bool:
        x, y, w, h = self.bbox
        return x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin


def unflatten(flat: list[float]) -> list[tuple[float, float, int]]:
    """51 число -> 17 троек. Единственное место, где знание о раскладке живёт."""
    if len(flat) != 51:
        raise ValueError(f"ожидалось 51 число, пришло {len(flat)}")
    return [(float(flat[3 * i]), float(flat[3 * i + 1]), int(flat[3 * i + 2]))
            for i in range(17)]


def flatten(points: list[tuple[float, float, int]]) -> list[float]:
    out: list[float] = []
    for x, y, v in points:
        out.extend([x, y, v] if v > ABSENT else [0, 0, 0])
    return out


def load_coco_keypoints(path: str | Path, images: set[str] | None = None,
                        min_kp: int = 0, min_area: float = 0.0
                        ) -> tuple[list[Person], dict[str, tuple[int, int]]]:
    """Читает и эталон, и экспорт CVAT.

    images   — оставить только эти имена файлов (None: все).
    min_kp   — минимум размеченных точек у человека.
    min_area — минимум площади. У экспорта CVAT поля area может не быть,
               тогда берётся площадь bbox: масштаб для OKS всё равно
               считается по эталону, а здесь это только фильтр.

    Возвращает список людей и размеры кадров {имя: (ширина, высота)}.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    name_by_id = {img["id"]: img["file_name"] for img in data["images"]}
    sizes = {img["file_name"]: (img["width"], img["height"]) for img in data["images"]}

    people: list[Person] = []
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        if "keypoints" not in ann:
            continue
        name = name_by_id.get(ann["image_id"])
        if name is None or (images is not None and name not in images):
            continue
        bbox = tuple(float(c) for c in ann.get("bbox", (0, 0, 0, 0)))
        area = float(ann.get("area") or bbox[2] * bbox[3])
        person = Person(image=name, ident=int(ann["id"]),
                        points=unflatten(ann["keypoints"]), area=area, bbox=bbox)
        if person.labeled < min_kp or person.area < min_area:
            continue
        people.append(person)
    return people, sizes


def save_coco_keypoints(people: list[Person], sizes: dict[str, tuple[int, int]],
                        path: str | Path) -> None:
    """Пишет файл в том же формате. Нужен, чтобы собрать подставную разметку
    для репетиции расчёта до того, как появится настоящая."""
    names = sorted({p.image for p in people})
    ids = {name: i + 1 for i, name in enumerate(names)}
    doc = {
        "info": {"description": "P4d"},
        "licenses": [],
        "images": [{"id": ids[n], "file_name": n,
                    "width": sizes.get(n, (0, 0))[0], "height": sizes.get(n, (0, 0))[1]}
                   for n in names],
        "annotations": [{
            "id": p.ident, "image_id": ids[p.image], "category_id": 1, "iscrowd": 0,
            "keypoints": flatten(p.points), "num_keypoints": p.labeled,
            "bbox": list(p.bbox), "area": p.area,
        } for p in people],
        "categories": [{
            "id": 1, "name": "person", "supercategory": "person",
            "keypoints": COCO_KEYPOINTS,
            "skeleton": [[a + 1, b + 1] for a, b in SKELETON],
        }],
    }
    Path(path).write_text(json.dumps(doc), encoding="utf-8")


def by_image(people: list[Person]) -> dict[str, list[Person]]:
    out: dict[str, list[Person]] = {}
    for p in people:
        out.setdefault(p.image, []).append(p)
    return out


def bbox_iou(a: tuple[float, float, float, float],
             b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (aw * ah + bw * bh - inter)


def flag_histogram(people: list[Person]) -> Counter:
    c: Counter = Counter()
    for p in people:
        for _, _, v in p.points:
            c[v] += 1
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description="Что лежит в файле COCO Keypoints")
    ap.add_argument("path", type=Path)
    ap.add_argument("--min-kp", type=int, default=0)
    ap.add_argument("--min-area", type=float, default=0.0)
    args = ap.parse_args()

    people, sizes = load_coco_keypoints(args.path, min_kp=args.min_kp,
                                        min_area=args.min_area)
    hist = flag_histogram(people)
    total = sum(hist.values())
    print(f"кадров {len({p.image for p in people})}, людей {len(people)}")
    print(f"слотов точек {total}: "
          f"v=0 {hist[0]} ({hist[0] / total:.0%}), "
          f"v=1 {hist[1]} ({hist[1] / total:.0%}), "
          f"v=2 {hist[2]} ({hist[2] / total:.0%})")
    print(f"размечено точек {hist[1] + hist[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
