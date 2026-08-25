#!/usr/bin/env python3
"""Renders skeleton disagreements.

The number "OKS 0.30" says nothing until you can see what actually went
wrong: an error spread over all points, one arm flying off, or swapped
sides. Three outputs, each answering its own question.

    pairs   a crop around the person carrying both skeletons. Blue is the
            ground truth, orange is mine. A point is filled when marked
            visible and hollow when marked not visible. The caption is the
            OKS of the pair
    joints  PCK bars for each of the 17 joints: where exactly the error is
    flags   bars of visibility-flag disagreement per joint: where the two
            sides diverged not on the coordinate but on whether the point
            is visible at all

    .venv/bin/python tools/render_skeletons.py \
        --gt data/coco/person_keypoints_val2017.json \
        --mine annotation/my_labels/person_keypoints_default.json \
        --selection data/subset/selection_keypoints.json \
        --images data/subset/frames --out reports/review
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from keypoints import (ABSENT, COCO_KEYPOINTS, SKELETON, VISIBLE,  # noqa: E402
                       Person, load_coco_keypoints)
from oks import evaluate, oks  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chrome import (MINE_COLOR as MY_COLOR, REF_COLOR,  # noqa: E402
                    load_font, with_chrome)

BAD = (220, 38, 38)
OK = (34, 160, 90)


def draw_person(d: ImageDraw.ImageDraw, p: Person, color, scale: float,
                ox: float, oy: float, radius: int = 4) -> None:
    def at(i):
        x, y, v = p.points[i]
        return ((x - ox) * scale, (y - oy) * scale, v)

    for a, b in SKELETON:
        xa, ya, va = at(a)
        xb, yb, vb = at(b)
        if va == ABSENT or vb == ABSENT:
            continue
        d.line([xa, ya, xb, yb], fill=color, width=2)
    for i in range(17):
        x, y, v = at(i)
        if v == ABSENT:
            continue
        box = [x - radius, y - radius, x + radius, y + radius]
        if v == VISIBLE:
            d.ellipse(box, fill=color, outline=color)
        else:
            d.ellipse(box, fill=None, outline=color, width=2)


def render_pair(image_path: Path, gt: Person, mine: Person, value: float,
                out: Path, font, target: int = 520) -> None:
    img = Image.open(image_path).convert("RGB")
    xs, ys = [], []
    for p in (gt, mine):
        x, y, w, h = p.kp_bbox()
        xs += [x, x + w]
        ys += [y, y + h]
    pad = 0.18 * max(max(xs) - min(xs), max(ys) - min(ys), 40)
    x0 = max(0, min(xs) - pad)
    y0 = max(0, min(ys) - pad)
    x1 = min(img.width, max(xs) + pad)
    y1 = min(img.height, max(ys) + pad)
    crop = img.crop((int(x0), int(y0), int(x1), int(y1)))
    scale = target / max(crop.width, crop.height)
    crop = crop.resize((max(1, int(crop.width * scale)),
                        max(1, int(crop.height * scale))))

    facts = f"OKS {value:.3f} · hollow = marked not visible"
    footer = f"{image_path.name} · reference #{gt.ident}"
    canvas, d, top, _ = with_chrome(crop, facts=facts, footer=footer, font=font)
    draw_person(d, gt, REF_COLOR, scale, x0, y0 - top / scale)
    draw_person(d, mine, MY_COLOR, scale, x0, y0 - top / scale)
    canvas.save(out, quality=92)


def render_joints(per_joint: dict, out: Path, font, mult: float) -> None:
    rows = [(COCO_KEYPOINTS[i], per_joint[i][0], per_joint[i][1]) for i in range(17)]
    rows = [(n, h, t) for n, h, t in rows if t]
    rows.sort(key=lambda r: r[1] / r[2])
    w, row_h, left = 640, 26, 150
    img = Image.new("RGB", (w, row_h * len(rows) + 44), (255, 255, 255))
    d = ImageDraw.Draw(img)
    title = f"PCK@{mult:g} sigma by joint"
    d.text((10, 10), title, fill=(20, 20, 20), font=font)
    bar_w = w - left - 90
    for k, (name, hit, tot) in enumerate(rows):
        y = 40 + k * row_h
        share = hit / tot
        d.text((10, y + 4), name, fill=(40, 40, 40), font=font)
        d.rectangle([left, y + 4, left + bar_w, y + 18], fill=(235, 235, 235))
        d.rectangle([left, y + 4, left + bar_w * share, y + 18],
                    fill=OK if share >= 0.8 else BAD)
        d.text((left + bar_w + 8, y + 4), f"{share:.2f}  n={tot}",
               fill=(60, 60, 60), font=font)
    img.save(out)


def render_flags(pairs, out: Path, font) -> None:
    """Share of flag disagreements per joint.

    A slot counts if at least one side annotated it: a slot both sides
    skipped is agreement about nothing.
    """
    diff = [0] * 17
    total = [0] * 17
    for g, m in pairs:
        for i, ((_, _, gv), (_, _, mv)) in enumerate(zip(g.points, m.points)):
            if gv == ABSENT and mv == ABSENT:
                continue
            total[i] += 1
            if gv != mv:
                diff[i] += 1
    rows = [(COCO_KEYPOINTS[i], diff[i], total[i]) for i in range(17) if total[i]]
    rows.sort(key=lambda r: -r[1] / r[2])

    w, row_h, left = 640, 26, 150
    img = Image.new("RGB", (w, row_h * len(rows) + 44), (255, 255, 255))
    d = ImageDraw.Draw(img)
    title = "visibility flag disagreement, share of slots"
    d.text((10, 10), title, fill=(20, 20, 20), font=font)
    bar_w = w - left - 90
    for k, (name, bad, tot) in enumerate(rows):
        y = 40 + k * row_h
        share = bad / tot
        d.text((10, y + 4), name, fill=(40, 40, 40), font=font)
        d.rectangle([left, y + 4, left + bar_w, y + 18], fill=(235, 235, 235))
        d.rectangle([left, y + 4, left + bar_w * share, y + 18],
                    fill=BAD if share >= 0.1 else OK)
        d.text((left + bar_w + 8, y + 4), f"{share:.2f}  n={tot}",
               fill=(60, 60, 60), font=font)
    img.save(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--mine", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--worst", type=int, default=8)
    ap.add_argument("--min-kp", type=int, default=8)
    ap.add_argument("--min-area", type=float, default=4000.0)
    ap.add_argument("--pck-mult", type=float, default=1.0)
    args = ap.parse_args()

    images = set(json.loads(args.selection.read_text(encoding="utf-8"))["files"])
    gt, _ = load_coco_keypoints(args.gt, images=images,
                                min_kp=args.min_kp, min_area=args.min_area)
    mine, _ = load_coco_keypoints(args.mine)
    res = evaluate(gt, mine, "bbox", 0.3, args.pck_mult)

    args.out.mkdir(parents=True, exist_ok=True)
    font = load_font(15)

    scored = [(g, m, oks(g, m)) for g, m in res["pairs"]]
    scored = sorted([t for t in scored if t[2] is not None], key=lambda t: t[2])
    made = []
    for k, (g, m, value) in enumerate(scored[:args.worst], 1):
        name = f"{k:02d}_{Path(g.image).stem}_gt{g.ident}.jpg"
        render_pair(args.images / g.image, g, m, value, args.out / name, font)
        made.append({"file": name, "image": g.image, "gt_id": g.ident,
                     "my_id": m.ident, "oks": value})
    render_joints(res["pck_per_joint"], args.out / "pck_by_joint.png",
                  font, args.pck_mult)
    render_flags(res["pairs"], args.out / "flag_by_joint.png", font)

    (args.out / "pairs_manifest.json").write_text(
        json.dumps(made, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"pairs rendered {len(made)}, plus pck_by_joint.png and flag_by_joint.png")
    print(f"directory: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
