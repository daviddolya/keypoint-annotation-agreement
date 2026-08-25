#!/usr/bin/env python3
"""Builds the README from the metrics and the worst-pair review.

The trick is carried over from the polygon and tracking projects and works
the same way: the numbers are generated from reports/keypoint_metrics.json,
while hand-written commentary survives. Text between <!-- note:key --> and
<!-- /note --> is read back from the existing README and carried into the
new one, so rebuilding after a re-annotation never destroys it.

Empty note blocks are never emitted: a marker appears only where there is
something to say, so the published README carries no placeholders.

The unit of a section here is a PAIR "ground-truth person - my person":
with skeletons what matters is not the frame but the individual figure and
what went wrong inside it.

    .venv/bin/python tools/build_readme.py
"""

import argparse
import json
import re
from pathlib import Path

NOTE_RE = re.compile(r"<!-- note:(?P<key>[^\s>]+) -->\n(?P<body>.*?)\n<!-- /note -->",
                     re.DOTALL)
# Empty stubs left by earlier runs; treated as "no note at all".
PLACEHOLDERS = {"> **What happened here:** _to be written_", "", "_to be written_"}


def existing_notes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    found = {m.group("key"): m.group("body").strip()
             for m in NOTE_RE.finditer(path.read_text(encoding="utf-8"))}
    return {k: v for k, v in found.items() if v not in PLACEHOLDERS}


def note_block(key: str, notes: dict[str, str]) -> list[str]:
    """A marker block, but only when there is real text to preserve."""
    body = notes.get(key)
    return [f"<!-- note:{key} -->", body, "<!-- /note -->", ""] if body else []


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics", type=Path, default=Path("reports/keypoint_metrics.json"))
    p.add_argument("--readme", type=Path, default=Path("README.md"))
    p.add_argument("--review", default="reports/review")
    p.add_argument("--repo", default="keypoint-annotation-agreement")
    a = p.parse_args()

    m = json.loads(a.metrics.read_text(encoding="utf-8"))
    notes = existing_notes(a.readme)
    review = Path(a.review)
    manifest = review / "pairs_manifest.json"
    pairs = (json.loads(manifest.read_text(encoding="utf-8"))
             if manifest.exists() else [])

    out = [f"# {a.repo}", "",
           "Annotation agreement for human skeletons: 17 COCO keypoints per person.",
           f"{m['frames']} val2017 frames were annotated by hand, blind to the",
           "ground truth. Three different things are measured: coordinate accuracy",
           "(OKS), accuracy per individual joint (PCK), and agreement on the",
           "visibility flag -- the last of which no coordinate metric can see.",
           "Stage A4 of an annotation-quality portfolio.", ""]
    out += note_block("intro", notes)
    out += ["## Result", "", "| | |", "|---|---|",
            f"| frames | {m['frames']} |",
            f"| people, mine / ground truth | {m['my_people']} / {m['gt_people']} |",
            f"| matched pairs | {m['matched']} |",
            f"| **OKS over common points** | **{m['oks_common']:.3f}** |",
            f"| OKS COCO-style | {m['oks_coco']:.3f} |",
            f"| PCK@{m['pck_mult']:g} sigma | {m['pck']:.3f} |",
            f"| **visibility-flag agreement** | **{m['flag_agreement']:.3f}** |",
            f"| Cohen's kappa on the flag | {m['flag_kappa']:.3f} |", "",
            f"People are matched by the box built from the annotated keypoints, "
            f"IoU threshold {m['match_threshold']}: that axis stays independent of",
            "the metric, otherwise a person with swapped left and right sides finds",
            "no match and the worst case simply vanishes from the report.", ""]

    if m.get("unmatched_gt") or m.get("unmatched_mine"):
        out += [f"Unmatched: {m['unmatched_gt']} in the ground truth, "
                f"{m['unmatched_mine']} of mine ({m['unmatched_mine_filtered_out']} of "
                "them match a person the ground-truth filter dropped, "
                f"{m['unmatched_mine_invented']} match nothing in the ground truth).", ""]

    out += ["## What a coordinate metric cannot see", "",
            "The visibility flag is a separate axis of the annotation, and OKS does",
            "not touch it at all: an annotation with perfect coordinates in which",
            "every point is marked visible scores exactly 1.000. Flag agreement is",
            "therefore computed on its own, with kappa next to it: raw agreement",
            "looks high by itself whenever one flag value dominates the rest.", "",
            "| GT \\ mine | v=0 not annotated | v=1 not visible | v=2 visible |",
            "|---|---|---|---|"]
    labels = ["v=0 not annotated", "v=1 not visible", "v=2 visible"]
    for i in range(3):
        row = m["flag_matrix"][i]
        out.append(f"| {labels[i]} | {row[0]} | {row[1]} | {row[2]} |")
    out.append("")
    if (review / "flag_by_joint.png").exists():
        out += [f"![flag disagreement by joint]({a.review}/flag_by_joint.png)", ""]

    out += ["## Where the hand misses", "",
            "The PCK tolerance is one COCO sigma for that joint at that scale, i.e.",
            "a contribution to OKS of no less than exp(-1/2) = 0.607. The threshold",
            "was fixed in advance and never tuned to the result.", ""]
    if (review / "pck_by_joint.png").exists():
        out += [f"![PCK by joint]({a.review}/pck_by_joint.png)", ""]
    worst = sorted(m["pck_per_joint"].items(),
                   key=lambda kv: (kv[1]["hit"] / kv[1]["total"]
                                   if kv[1]["total"] else 1.0))[:5]
    out += ["| worst joints | PCK | points | mean offset, px |", "|---|---|---|---|"]
    for name, v in worst:
        if not v["total"]:
            continue
        out.append(f"| {name} | {v['hit'] / v['total']:.2f} | {v['total']} | "
                   f"{v['mean_px']:.1f} |")
    out.append("")

    if m.get("oks_by_size"):
        out += ["| person size | pairs | OKS |", "|---|---|---|"]
        for name in ("small", "medium", "large"):
            b = m["oks_by_size"].get(name)
            if b:
                out.append(f"| {name} | {b['pairs']} | {b['oks']:.3f} |")
        out.append("")

    if pairs:
        out += ["## The worst pairs", "",
                "Every picture carries its own legend: a blue swatch for the reference, an orange one for mine, the numbers of the case beside them and the frame name underneath.",
                "A point is filled when it",
                "is marked visible and hollow when it is marked not visible.", ""]
        for item in pairs:
            key = f"{Path(item['image']).stem}_{item['gt_id']}"
            out += [f"### {item['image']} - ground truth #{item['gt_id']}", "",
                    f"OKS {item['oks']:.3f}", "",
                    f"![{key}]({a.review}/{item['file']})", ""]
            out += note_block(key, notes)

    out += ["## Reproduce", "",
            "Python 3.10+ and Pillow; Pillow is needed for the rendering only, all",
            "metrics run on the standard library (the Hungarian algorithm included).",
            "", "```bash",
            "python3 -m venv .venv && .venv/bin/pip install -r requirements.txt",
            "",
            "# the four self-test cases with answers known in advance",
            ".venv/bin/python common/oks.py --selftest",
            "",
            "# ground truth (10 MB, not stored in the repository)",
            ".venv/bin/python tools/fetch_keypoints.py --out data/coco",
            "",
            "# sanity-check the export before computing anything",
            ".venv/bin/python tools/check_export.py \\",
            "    --mine annotation/person_keypoints_default.json \\",
            "    --selection data/subset/selection_keypoints.json",
            "",
            "# the numbers in this README",
            ".venv/bin/python annotation/keypoint_agreement.py \\",
            "    --gt data/coco/person_keypoints_val2017.json \\",
            "    --mine annotation/person_keypoints_default.json \\",
            "    --selection data/subset/selection_keypoints.json \\",
            "    --out reports/keypoint_metrics.json",
            "",
            "# the pictures above, then this README",
            ".venv/bin/python tools/render_skeletons.py \\",
            "    --gt data/coco/person_keypoints_val2017.json \\",
            "    --mine annotation/person_keypoints_default.json \\",
            "    --selection data/subset/selection_keypoints.json \\",
            "    --images data/subset/frames --out reports/review",
            f".venv/bin/python tools/build_readme.py --repo {a.repo}",
            "```", "",
            "The 14 frames and the selection manifest are committed, so the numbers",
            "can be reproduced without rebuilding the subset.", ""]

    out += ["## What else is here", "",
            "- Annotation guidelines and the disputed-case decisions -- "
            "[annotation/GUIDELINES.md](annotation/GUIDELINES.md)",
            "- Full report -- [reports/keypoint_report.md](reports/keypoint_report.md)",
            "- Code I did not write myself, and what I owe an explanation for -- "
            "[DEBT.md](DEBT.md)", "",
            "## The other stages of this portfolio", "",
            "| stage | type | headline numbers |", "|---|---|---|",
            "| P2 | [boxes](https://github.com/daviddolya/detection-annotation-agreement) "
            "| kappa 0.914, mean IoU 0.867 |",
            "| A2 | [polygons and masks](https://github.com/daviddolya/polygon-annotation-agreement) "
            "| mask IoU 0.840, Boundary IoU 0.676 |",
            "| A3 | [tracks on video](https://github.com/daviddolya/tracking-annotation-agreement) "
            "| IDF1 0.896, 2 ID switches |",
            "| A4 | skeletons -- **this repository** "
            f"| OKS {m['oks_common']:.3f}, flag agreement {m['flag_agreement']:.3f} |",
            "| A5 | [scene text](https://github.com/daviddolya/ocr-annotation-agreement) "
            "| mask IoU 0.784, CER 0.223 |", "",
            "This README is generated by `tools/build_readme.py` from",
            "`reports/keypoint_metrics.json`; edit the report, not this file.", ""]

    a.readme.write_text("\n".join(out), encoding="utf-8")
    print(f"{a.readme}: pair sections {len(pairs)}, notes preserved {len(notes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
