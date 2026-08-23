#!/usr/bin/env python3
"""Согласие по скелетам: OKS, PCK по суставам, флаг видимости (P4d, шаг 5).

Главный код этапа. Конвейер из четырёх стадий, каждая отвечает на свой
вопрос, и путать их нельзя:

  1. КОГО С КЕМ СРАВНИВАТЬ. Внутри кадра моих людей надо сопоставить
     с эталонными. Венгерский алгоритм по IoU рамки, построенной по
     размеченным точкам. Ось независима от метрики: сопоставили по одному,
     меряем другим.
  2. НАСКОЛЬКО ТОЧНО СТОЯТ ТОЧКИ. OKS на каждой паре, два режима рядом —
     по общим точкам и COCO-style. Расхождение между ними означает, что
     проблема не в координатах, а в том, какие точки вообще размечать.
  3. ГДЕ ИМЕННО ПРОМАХ. PCK по каждому из 17 суставов: доля точек,
     попавших в допуск одной сигмы. Здесь видно, что запястья и лодыжки
     сходятся хуже плеч, и это материал для инструкции.
  4. ЧТО МЕТРИКА ПО КООРДИНАТАМ НЕ ВИДИТ. Согласие по флагу видимости
     отдельным числом плюс матрица 3x3 и каппа Коэна.

Плюс разбор остатка: мой человек без пары — это либо кто-то, кого фильтр
эталона отбросил (мелкий, размечен тремя точками), либо разметка того,
чего в эталоне нет вовсе. Разные вещи, и в отчёте их не смешивают.

    python3 annotation/keypoint_agreement.py \
        --gt data/coco/person_keypoints_val2017.json \
        --mine annotation/my_labels/person_keypoints_default.json \
        --selection data/subset/selection_keypoints.json \
        --out reports/keypoint_metrics.json
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from keypoints import COCO_KEYPOINTS, Person, bbox_iou, by_image, load_coco_keypoints  # noqa: E402
from oks import evaluate, oks, point_errors  # noqa: E402


def classify_extra(lost_mine: list[Person], all_gt: list[Person],
                   kept_ids: set[int], threshold: float = 0.3
                   ) -> tuple[list[Person], list[Person]]:
    """Мои люди без пары: кто из них соответствует отброшенному фильтром
    эталонному человеку, а кто не соответствует никому."""
    pool = by_image([p for p in all_gt if p.ident not in kept_ids])
    filtered_out, invented = [], []
    for m in lost_mine:
        best = max((bbox_iou(g.kp_bbox(), m.kp_bbox()) for g in pool.get(m.image, [])),
                   default=0.0)
        (filtered_out if best >= threshold else invented).append(m)
    return filtered_out, invented


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--mine", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("reports/keypoint_metrics.json"))
    ap.add_argument("--min-kp", type=int, default=8)
    ap.add_argument("--min-area", type=float, default=4000.0)
    ap.add_argument("--match", choices=("bbox", "oks"), default="bbox")
    ap.add_argument("--match-threshold", type=float, default=0.3)
    ap.add_argument("--pck-mult", type=float, default=1.0)
    args = ap.parse_args()

    images = set(json.loads(args.selection.read_text(encoding="utf-8"))["files"])
    gt, sizes = load_coco_keypoints(args.gt, images=images,
                                    min_kp=args.min_kp, min_area=args.min_area)
    all_gt, _ = load_coco_keypoints(args.gt, images=images)
    mine, _ = load_coco_keypoints(args.mine)

    res = evaluate(gt, mine, args.match, args.match_threshold, args.pck_mult)
    pairs = res["pairs"]

    print(f"кадров {len(images)}; эталонных людей {len(gt)}, своих {len(mine)}")
    print(f"сопоставление по {'рамке из точек' if args.match == 'bbox' else 'OKS'}, "
          f"порог {args.match_threshold}: пар {res['matched']}, "
          f"эталонных без пары {res['unmatched_gt']}, своих без пары {res['unmatched_mine']}")

    kept = {p.ident for p in gt}
    filtered_out, invented = classify_extra(res["lost_mine"], all_gt, kept)
    if res["lost_mine"]:
        print(f"  из своих без пары: {len(filtered_out)} — человек, которого "
              f"отбросил фильтр эталона; {len(invented)} — не соответствует "
              "ничему в эталоне")

    # сверка второго способа сопоставления: совпало ли разбиение
    other = "oks" if args.match == "bbox" else "bbox"
    alt = evaluate(gt, mine, other, args.match_threshold, args.pck_mult)
    same = {(g.ident, m.ident) for g, m in pairs} == {
        (g.ident, m.ident) for g, m in alt["pairs"]}
    if same:
        print(f"  сопоставление по «{other}» дало то же разбиение")
    else:
        print(f"  сопоставление по «{other}» дало ДРУГОЕ разбиение: "
              f"пар {alt['matched']} вместо {res['matched']}, "
              f"OKS {alt['oks_common']:.3f} вместо {res['oks_common']:.3f}. "
              "Разницу назвать в отчёте: если пар стало меньше, а OKS вырос — "
              "из счёта выпали именно худшие пары.")

    if not pairs:
        raise SystemExit("ни одной пары: проверь экспорт через tools/check_export.py")

    per_pair = [(g, m, oks(g, m)) for g, m in pairs]
    values = [v for _, _, v in per_pair if v is not None]
    coco_values = [v for v in (oks(g, m, "coco") for g, m in pairs) if v is not None]
    print()
    print(f"OKS по общим точкам: {statistics.mean(values):.3f} "
          f"(медиана {statistics.median(values):.3f}, "
          f"минимум {min(values):.3f})")
    print(f"OKS COCO-style (не поставленная точка — промах): "
          f"{statistics.mean(coco_values):.3f}")
    print(f"PCK@{args.pck_mult:g}σ по всем точкам: {res['pck']:.3f}")

    print()
    print("| сустав | PCK | точек | средний промах, px |")
    print("|---|---|---|---|")
    dist: dict[int, list[float]] = {i: [] for i in range(17)}
    for g, m in pairs:
        for i, d, _ in point_errors(g, m):
            dist[i].append(d)
    order = sorted(range(17), key=lambda i: (res["pck_per_joint"][i][0]
                                             / res["pck_per_joint"][i][1]
                                             if res["pck_per_joint"][i][1] else 1.0))
    for i in order:
        hit, tot = res["pck_per_joint"][i]
        if not tot:
            print(f"| {COCO_KEYPOINTS[i]} | — | 0 | — |")
            continue
        print(f"| {COCO_KEYPOINTS[i]} | {hit / tot:.2f} | {tot} | "
              f"{statistics.mean(dist[i]):.1f} |")

    f = res["flags"]
    print()
    print(f"согласие по флагу видимости: {f['agreement']:.3f}, "
          f"каппа Коэна {f['kappa']:.3f} (слотов {f['total']})")
    print("| эталон \\ моё | v=0 | v=1 | v=2 |")
    print("|---|---|---|---|")
    for i in range(3):
        print(f"| v={i} | {f['matrix'][i][0]} | {f['matrix'][i][1]} | {f['matrix'][i][2]} |")

    areas = sorted(g.area for g, _ in pairs)
    q1, q3 = areas[len(areas) // 4], areas[3 * len(areas) // 4]
    groups: dict[str, list[float]] = {"мелкие": [], "средние": [], "крупные": []}
    for g, _, v in per_pair:
        if v is None:
            continue
        groups["мелкие" if g.area <= q1 else
               ("средние" if g.area <= q3 else "крупные")].append(v)
    print()
    print("| размер человека | пар | OKS |")
    print("|---|---|---|")
    for name in ("мелкие", "средние", "крупные"):
        if groups[name]:
            print(f"| {name} | {len(groups[name])} | "
                  f"{statistics.mean(groups[name]):.3f} |")

    worst = sorted([p for p in per_pair if p[2] is not None], key=lambda t: t[2])[:5]
    print()
    print("худшие пары:")
    for g, m, v in worst:
        print(f"  {g.image} эталон#{g.ident} ↔ моё#{m.ident}: OKS {v:.3f}, "
              f"размечено {g.labeled}/{m.labeled}")

    doc = {
        "frames": len(images),
        "gt_people": len(gt), "my_people": len(mine),
        "matched": res["matched"],
        "unmatched_gt": res["unmatched_gt"], "unmatched_mine": res["unmatched_mine"],
        "unmatched_mine_filtered_out": len(filtered_out),
        "unmatched_mine_invented": len(invented),
        "match_mode": args.match, "match_threshold": args.match_threshold,
        "match_modes_agree": same,
        "oks_common": statistics.mean(values),
        "oks_common_median": statistics.median(values),
        "oks_coco": statistics.mean(coco_values),
        "pck": res["pck"], "pck_mult": args.pck_mult,
        "pck_per_joint": {COCO_KEYPOINTS[i]: {
            "hit": res["pck_per_joint"][i][0], "total": res["pck_per_joint"][i][1],
            "mean_px": (statistics.mean(dist[i]) if dist[i] else None),
        } for i in range(17)},
        "flag_agreement": f["agreement"], "flag_kappa": f["kappa"],
        "flag_matrix": f["matrix"], "flag_slots": f["total"],
        "oks_by_size": {k: {"pairs": len(v), "oks": statistics.mean(v)}
                        for k, v in groups.items() if v},
        "worst_pairs": [{"image": g.image, "gt_id": g.ident, "my_id": m.ident,
                         "oks": v, "gt_labeled": g.labeled, "my_labeled": m.labeled}
                        for g, m, v in worst],
        "pairs": [{"image": g.image, "gt_id": g.ident, "my_id": m.ident,
                   "oks": v, "area": g.area} for g, m, v in per_pair],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"метрики: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
