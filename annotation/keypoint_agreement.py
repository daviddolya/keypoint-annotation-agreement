#!/usr/bin/env python3
"""Skeleton agreement: OKS, per-joint PCK, visibility flag.

The main script of the project. A pipeline of four stages, each answering
its own question, and they must not be conflated:

  1. WHO IS COMPARED WITH WHOM. Within a frame, my people have to be matched
     to the ground-truth ones. Hungarian algorithm over the IoU of the box
     built from the annotated points. The axis is independent of the metric:
     matched by one thing, measured by another.
  2. HOW PRECISELY THE POINTS SIT. OKS on every pair, two modes side by side
     -- over common points and COCO-style. A gap between them means the
     problem is not the coordinates but which points to annotate at all.
  3. WHERE THE ERROR IS. PCK for each of the 17 joints: the share of points
     inside a one-sigma tolerance. This is where wrists and ankles turn out
     to agree worse than shoulders, which is material for the guidelines.
  4. WHAT A COORDINATE METRIC CANNOT SEE. Visibility-flag agreement as a
     separate number, plus a 3x3 matrix and Cohen's kappa.

Plus the leftovers: a person of mine without a match is either someone the
ground-truth filter dropped (too small, annotated with three points) or an
annotation of something the ground truth does not contain at all. Different
things, and a report must not merge them.

    python3 annotation/keypoint_agreement.py \
        --gt data/coco/person_keypoints_val2017.json \
        --mine annotation/person_keypoints_default.json \
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
    """My unmatched people: which of them correspond to a ground-truth person
    dropped by the filter, and which correspond to nobody at all."""
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

    print(f"frames {len(images)}; ground-truth people {len(gt)}, mine {len(mine)}")
    print(f"matching by {'keypoint box' if args.match == 'bbox' else 'OKS'}, "
          f"threshold {args.match_threshold}: pairs {res['matched']}, "
          f"unmatched in GT {res['unmatched_gt']}, unmatched of mine {res['unmatched_mine']}")

    kept = {p.ident for p in gt}
    filtered_out, invented = classify_extra(res["lost_mine"], all_gt, kept)
    if res["lost_mine"]:
        print(f"  of my unmatched: {len(filtered_out)} match a person the "
              f"ground-truth filter dropped; {len(invented)} match nothing "
              "in the ground truth")

    # cross-check the other matching mode: did the partition come out the same
    other = "oks" if args.match == "bbox" else "bbox"
    alt = evaluate(gt, mine, other, args.match_threshold, args.pck_mult)
    same = {(g.ident, m.ident) for g, m in pairs} == {
        (g.ident, m.ident) for g, m in alt["pairs"]}
    if same:
        print(f"  matching by \"{other}\" produced the same partition")
    else:
        print(f"  matching by \"{other}\" produced a DIFFERENT partition: "
              f"pairs {alt['matched']} instead of {res['matched']}, "
              f"OKS {alt['oks_common']:.3f} instead of {res['oks_common']:.3f}. "
              "State the difference in the report: if there are fewer pairs and "
              "OKS went up, the worst pairs are what dropped out of the count.")

    if not pairs:
        raise SystemExit("no pairs at all: check the export with tools/check_export.py")

    per_pair = [(g, m, oks(g, m)) for g, m in pairs]
    values = [v for _, _, v in per_pair if v is not None]
    coco_values = [v for v in (oks(g, m, "coco") for g, m in pairs) if v is not None]
    print()
    print(f"OKS over common points: {statistics.mean(values):.3f} "
          f"(median {statistics.median(values):.3f}, "
          f"minimum {min(values):.3f})")
    print(f"OKS COCO-style (a point I did not place counts as a miss): "
          f"{statistics.mean(coco_values):.3f}")
    print(f"PCK@{args.pck_mult:g} sigma over all points: {res['pck']:.3f}")

    print()
    print("| joint | PCK | points | mean offset, px |")
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
    print(f"visibility-flag agreement: {f['agreement']:.3f}, "
          f"Cohen's kappa {f['kappa']:.3f} (slots {f['total']})")
    print("| GT \\ mine | v=0 | v=1 | v=2 |")
    print("|---|---|---|---|")
    for i in range(3):
        print(f"| v={i} | {f['matrix'][i][0]} | {f['matrix'][i][1]} | {f['matrix'][i][2]} |")

    areas = sorted(g.area for g, _ in pairs)
    q1, q3 = areas[len(areas) // 4], areas[3 * len(areas) // 4]
    groups: dict[str, list[float]] = {"small": [], "medium": [], "large": []}
    for g, _, v in per_pair:
        if v is None:
            continue
        groups["small" if g.area <= q1 else
               ("medium" if g.area <= q3 else "large")].append(v)
    print()
    print("| person size | pairs | OKS |")
    print("|---|---|---|")
    for name in ("small", "medium", "large"):
        if groups[name]:
            print(f"| {name} | {len(groups[name])} | "
                  f"{statistics.mean(groups[name]):.3f} |")

    worst = sorted([p for p in per_pair if p[2] is not None], key=lambda t: t[2])[:5]
    print()
    print("worst pairs:")
    for g, m, v in worst:
        print(f"  {g.image} GT#{g.ident} <-> mine#{m.ident}: OKS {v:.3f}, "
              f"annotated {g.labeled}/{m.labeled}")

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
    print(f"metrics: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
