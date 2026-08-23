#!/usr/bin/env python3
"""OKS, PCK и согласие по флагу видимости (P4d).

ЗАЧЕМ ОТДЕЛЬНАЯ МЕТРИКА. У бокса и полигона есть площадь, поэтому есть IoU.
У точки площади нет: пересечение двух точек пусто всегда, IoU тождественно
равен нулю. Мерить приходится расстоянием — но голое расстояние в пикселях
непригодно по двум причинам сразу. Пять пикселей на человеке во весь кадр
и пять пикселей на человеке ростом в сто пикселей — разные ошибки. И пять
пикселей на кончике носа и пять пикселей на бедре — тоже разные: где именно
находится бедро под одеждой, два аккуратных разметчика согласны куда хуже,
чем где находится нос.

OKS чинит обе проблемы одной формулой:

    OKS = сумма по размеченным точкам exp( -d_i^2 / (2 * s^2 * k_i^2) )
          делить на число размеченных точек

    d_i  расстояние между точкой эталона и моей точкой, в пикселях
    s^2  площадь объекта в эталоне (area). Нормировка на масштаб
    k_i  = 2 * sigma_i, посуставная константа

СИГМЫ — ЭТО ЗАМЕР, А НЕ НАСТРОЙКА. Авторы COCO дали часть кадров на повторную
разметку и посчитали, с каким разбросом люди ставят каждый сустав. Плечо
вышло 0.079, бедро 0.107, нос 0.026. То есть на одном и том же человеке
промах по носу в четыре раза «дороже» такого же промаха по бедру — просто
потому, что по носу разметчики обычно согласны, а по бедру нет.

ЧТО ОТСЮДА СЛЕДУЕТ ПРАКТИЧЕСКИ. Сдвиг ровно на d = s * k_i даёт
exp(-1/2) = 0.6065 — этим удобно проверять реализацию без компьютера.
И OKS асимметрична: масштаб s берётся из эталона, поэтому «OKS моей
разметки относительно эталона» и наоборот — разные числа. Здесь всегда
эталон в первом аргументе.

ЧЕГО OKS НЕ ВИДИТ. Она считается по точкам и молчит про флаг видимости:
разметка с идеальными координатами, где каждая точка помечена «видна»,
даёт ровно 1.000, хотя часть флагов расходится с эталоном. Поэтому
согласие по флагу считается отдельной метрикой, а не приписывается к OKS.

Глубже: cocodataset.org/#keypoints-eval (определение и таблица сигм);
Ronchi & Perona, "Benchmarking and Error Diagnosis in Multi-Instance Pose
Estimation", arXiv:1707.05388 (разбор того, из чего складывается ошибка).

Венгерский алгоритм перенесён из tracking-annotation-agreement/common/matching.py
(этап A3, P4c) — там же его подробное описание. Зависимостей нет, scipy не нужен.
"""

import argparse
import math
from collections import Counter

from keypoints import (ABSENT, HIDDEN, K, LR_PAIRS, VISIBLE, COCO_KEYPOINTS,
                       Person, bbox_iou, by_image)

INF = float("inf")


def hungarian(cost: list[list[float]]) -> list[tuple[int, int]]:
    """Минимальное по стоимости паросочетание в квадратной матрице.

    Венгерский алгоритм в форме Кюна — Манкреса, O(n^3). Перенесён из
    common/matching.py проекта A3 без изменений. Важно не то, как он устроен
    внутри, а то, что он не жадный: жадный выбор «сначала лучшая пара, потом
    следующая лучшая» может забрать пару, выгодную локально, и оставить
    другому человеку только плохие варианты.
    """
    n, m = len(cost), len(cost[0])
    assert n == m, "матрица должна быть квадратной — дополняй фиктивными строками"
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] > 0]


# --- стадия 1: метрика на одной паре людей ---------------------------------

def oks(gt: Person, mine: Person, mode: str = "common") -> float | None:
    """OKS пары «эталонный человек — мой человек». Масштаб берётся из эталона.

    mode='common' — усреднение по точкам, размеченным ОБЕИМИ сторонами.
        Отвечает на вопрос «насколько точно я ставлю точки там, где ставлю».
        Расхождение в самом факте разметки уходит в согласие по флагу.
    mode='coco'   — усреднение по всем точкам, размеченным в эталоне;
        не поставленная мной точка даёт нулевой вклад. Так считает COCOeval,
        где вторая сторона — модель, и «не нашла» это ошибка.

    Два числа рядом полезнее одного: если они сильно разошлись, значит
    расхождение не в координатах, а в том, какие точки вообще размечать.
    """
    num, den = 0.0, 0
    for i, ((gx, gy, gv), (mx, my, mv)) in enumerate(zip(gt.points, mine.points)):
        if gv == ABSENT:
            continue
        if mode == "common" and mv == ABSENT:
            continue
        den += 1
        if mv == ABSENT:
            continue                       # coco-режим: вклад нулевой
        d2 = (gx - mx) ** 2 + (gy - my) ** 2
        num += math.exp(-d2 / (2 * gt.area * K[i] ** 2))
    return num / den if den else None


def point_errors(gt: Person, mine: Person) -> list[tuple[int, float, float]]:
    """(индекс сустава, расстояние в пикселях, порог 1 сигмы в пикселях)
    по точкам, размеченным обеими сторонами. Сырьё для PCK и для разбора."""
    out = []
    s = math.sqrt(gt.area)
    for i, ((gx, gy, gv), (mx, my, mv)) in enumerate(zip(gt.points, mine.points)):
        if gv == ABSENT or mv == ABSENT:
            continue
        out.append((i, math.hypot(gx - mx, gy - my), s * K[i]))
    return out


def pck(pairs: list[tuple[Person, Person]], mult: float = 1.0
        ) -> tuple[dict[int, tuple[int, int]], float]:
    """PCK по суставам: доля точек, попавших в допуск.

    Допуск здесь — не фиксированное число пикселей, а mult * s * k_i, то есть
    тот же масштаб и та же посуставная сигма, что в OKS. При mult=1.0 «попал»
    означает вклад в OKS не ниже exp(-1/2) = 0.6065, и одно число сравнимо
    с другим. Порог называется в отчёте, а не подбирается под результат.

    Возвращает {индекс сустава: (попаданий, всего)} и общую долю.
    """
    per: dict[int, list[int]] = {i: [0, 0] for i in range(17)}
    for g, m in pairs:
        for i, dist, sigma_px in point_errors(g, m):
            per[i][1] += 1
            if dist <= mult * sigma_px + 1e-9:
                per[i][0] += 1
    hit = sum(v[0] for v in per.values())
    tot = sum(v[1] for v in per.values())
    return {i: (v[0], v[1]) for i, v in per.items()}, (hit / tot if tot else 0.0)


# --- стадия 2: согласие по флагу, отдельно от координат ---------------------

def flag_confusion(pairs: list[tuple[Person, Person]]) -> dict:
    """Матрица 3x3 по флагу видимости плюс доля совпадений и каппа Коэна.

    Слот учитывается, если хотя бы одна сторона что-то про него сказала:
    точка, которую обе стороны не разметили, — это согласие ни о чём,
    и включать её значит завышать метрику (в эталоне таких четверть).

    Каппа считается как в P2 по классам: она вычитает то согласие, которое
    получилось бы при случайном проставлении флагов с теми же частотами.
    """
    m = [[0] * 3 for _ in range(3)]
    for g, my in pairs:
        for (_, _, gv), (_, _, mv) in zip(g.points, my.points):
            if gv == ABSENT and mv == ABSENT:
                continue
            m[gv][mv] += 1
    total = sum(sum(row) for row in m)
    if total == 0:
        return {"matrix": m, "total": 0, "agreement": 0.0, "kappa": 0.0}
    observed = sum(m[i][i] for i in range(3)) / total
    rows = [sum(r) / total for r in m]
    cols = [sum(m[i][j] for i in range(3)) / total for j in range(3)]
    expected = sum(rows[i] * cols[i] for i in range(3))
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return {"matrix": m, "total": total, "agreement": observed, "kappa": kappa}


# --- стадия 3: кого с кем сравнивать ---------------------------------------

def match_people(gt: list[Person], mine: list[Person], mode: str = "bbox",
                 threshold: float = 0.3
                 ) -> tuple[list[tuple[Person, Person]], list[Person], list[Person]]:
    """Сопоставление людей внутри каждого кадра. Венгерский алгоритм.

    mode='bbox' (по умолчанию) — стоимость по IoU рамки, построенной
        ПО РАЗМЕЧЕННЫМ ТОЧКАМ, а не по полю bbox из файла: у эталона и
        у экспорта CVAT это поле заполняется по-разному, а точки одни и те же.
        Ось независима от самой метрики — сопоставили по одному, меряем другим.
    mode='oks'  — стоимость по OKS, как в COCOeval. Считать метрику тем же,
        чем сопоставляли, — замкнутый круг: человек с перепутанными сторонами
        получает низкий OKS и рискует остаться без пары, то есть самая
        интересная ошибка исчезнет из отчёта вместо того, чтобы попасть в него.

    Возвращает пары, несопоставленных из эталона и несопоставленных своих.
    """
    pairs: list[tuple[Person, Person]] = []
    lost_gt: list[Person] = []
    lost_mine: list[Person] = []
    g_by, m_by = by_image(gt), by_image(mine)

    for image in sorted(set(g_by) | set(m_by)):
        gs, ms = g_by.get(image, []), m_by.get(image, [])
        if not gs or not ms:
            lost_gt.extend(gs)
            lost_mine.extend(ms)
            continue
        size = max(len(gs), len(ms))
        score = [[0.0] * size for _ in range(size)]
        for a, g in enumerate(gs):
            for b, m in enumerate(ms):
                if mode == "oks":
                    value = oks(g, m) or 0.0
                else:
                    value = bbox_iou(g.kp_bbox(), m.kp_bbox())
                score[a][b] = value
        cost = [[1.0 - score[a][b] for b in range(size)] for a in range(size)]
        taken_g, taken_m = set(), set()
        for a, b in hungarian(cost):
            if a < len(gs) and b < len(ms) and score[a][b] >= threshold:
                pairs.append((gs[a], ms[b]))
                taken_g.add(a)
                taken_m.add(b)
        lost_gt.extend(g for a, g in enumerate(gs) if a not in taken_g)
        lost_mine.extend(m for b, m in enumerate(ms) if b not in taken_m)
    return pairs, lost_gt, lost_mine


# --- стадия 4: всё вместе ---------------------------------------------------

def evaluate(gt: list[Person], mine: list[Person], mode: str = "bbox",
             threshold: float = 0.3, pck_mult: float = 1.0) -> dict:
    pairs, lost_gt, lost_mine = match_people(gt, mine, mode, threshold)
    common = [oks(g, m) for g, m in pairs]
    coco = [oks(g, m, "coco") for g, m in pairs]
    common = [v for v in common if v is not None]
    coco = [v for v in coco if v is not None]
    per_joint, overall = pck(pairs, pck_mult)
    flags = flag_confusion(pairs)
    return {
        "matched": len(pairs),
        "gt_people": len(gt),
        "my_people": len(mine),
        "unmatched_gt": len(lost_gt),
        "unmatched_mine": len(lost_mine),
        "oks_common": sum(common) / len(common) if common else 0.0,
        "oks_coco": sum(coco) / len(coco) if coco else 0.0,
        "pck": overall,
        "pck_mult": pck_mult,
        "pck_per_joint": per_joint,
        "flags": flags,
        "pairs": pairs,
        "lost_gt": lost_gt,
        "lost_mine": lost_mine,
    }


# --- контрольные случаи -----------------------------------------------------

# Фигурка человека с известными координатами: площадь ровно 10000, значит
# s = 100 и порог одной сигмы по плечу — 15.8 px. Ничего не качается и
# не читается с диска: тест обязан идти на пустой машине.
TEMPLATE = [
    (100, 30), (108, 24), (92, 24), (118, 28), (82, 28),
    (135, 70), (65, 70), (150, 120), (50, 120), (160, 170), (40, 170),
    (125, 180), (75, 180), (130, 260), (70, 260), (133, 340), (67, 340),
]


def _person(flags: list[int], dx: float = 0.0, sigma_shift: float = 0.0,
            swap: bool = False, ident: int = 1) -> Person:
    pts = [(x + dx + sigma_shift * 100 * K[i], y, flags[i])
           for i, (x, y) in enumerate(TEMPLATE)]
    if swap:
        out = list(pts)
        for i, j in LR_PAIRS:
            out[i], out[j] = pts[j], pts[i]
        pts = out
    p = Person(image="synthetic.jpg", ident=ident, points=pts, area=10000.0,
               bbox=(40.0, 24.0, 120.0, 316.0))
    return p


def _selftest() -> int:
    """Четыре случая с ответом, известным заранее."""
    flags = [VISIBLE] * 17
    flags[3] = flags[4] = HIDDEN     # уши размечены, но не видны
    flags[9] = flags[10] = ABSENT    # запястья не размечены вовсе
    ref = _person(flags)

    cases = [
        ("разметка совпадает с эталоном", _person(flags)),
        ("каждая точка сдвинута ровно на s*k_i", _person(flags, sigma_shift=1.0)),
        ("лево и право переставлены местами", _person(flags, swap=True)),
        ("координаты те же, флаг везде «видна»",
         _person([ABSENT if f == ABSENT else VISIBLE for f in flags])),
    ]

    print(f"эталон: площадь {ref.area:.0f}, s = {math.sqrt(ref.area):.0f} px, "
          f"размечено точек {ref.labeled} из 17")
    print(f"порог одной сигмы: нос {math.sqrt(ref.area) * K[0]:.1f} px, "
          f"плечо {math.sqrt(ref.area) * K[5]:.1f} px, "
          f"бедро {math.sqrt(ref.area) * K[11]:.1f} px")
    print()
    print("| случай | OKS | PCK@1σ | согласие по флагу |")
    print("|---|---|---|---|")
    results = []
    for name, mine in cases:
        pairs = [(ref, mine)]
        _, overall = pck(pairs)
        f = flag_confusion(pairs)
        value = oks(ref, mine)
        results.append(value)
        print(f"| {name} | {value:.4f} | {overall:.3f} | {f['agreement']:.3f} |")

    print()
    ok = True
    if abs(results[0] - 1.0) > 1e-9:
        print("ПРОВАЛ: совпадающая разметка обязана давать ровно 1.0")
        ok = False
    expected = math.exp(-0.5)
    if abs(results[1] - expected) > 1e-9:
        print(f"ПРОВАЛ: сдвиг на одну сигму обязан давать exp(-1/2) = {expected:.4f}")
        ok = False
    else:
        print(f"сдвиг на одну сигму дал exp(-1/2) = {expected:.4f} — "
              "это проверяется на бумаге, без компьютера")
    if not results[2] < 0.5:
        print("ПРОВАЛ: перестановка сторон обязана рушить метрику")
        ok = False
    if abs(results[3] - 1.0) > 1e-9:
        print("ПРОВАЛ: разница только во флаге не должна менять OKS")
        ok = False
    else:
        print("разница только во флаге не изменила OKS вовсе — "
              "именно поэтому согласие по флагу считается отдельно")
    print("все четыре случая сошлись" if ok else "есть расхождения")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true",
                    help="прогнать четыре контрольных случая")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
