#!/usr/bin/env python3
"""Сборка README из метрик и разбора пар (P4d, шаг 6).

Приём перенесён из A2 и A3 и работает так же: числа генерируются из
reports/keypoint_metrics.json, а твои комментарии сохраняются. Текст между
маркерами <!-- note:ключ --> и <!-- /note --> вычитывается из существующего
README и переносится в новый, поэтому пересборка после переразметки
ничего не затирает.

Единица раздела здесь — ПАРА «эталонный человек — мой человек»: у скелета
интересен не кадр целиком, а конкретная фигура и то, что в ней разошлось.

    .venv/bin/python tools/build_readme.py
"""

import argparse
import json
import re
from pathlib import Path

PLACEHOLDER = "> **Что здесь произошло:** _заполнить_"
NOTE_RE = re.compile(r"<!-- note:(?P<key>[^\s>]+) -->\n(?P<body>.*?)\n<!-- /note -->",
                     re.DOTALL)


def existing_notes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {m.group("key"): m.group("body").strip()
            for m in NOTE_RE.finditer(path.read_text(encoding="utf-8"))}


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
           "Согласованность разметки скелетов: 17 точек COCO на человека.",
           f"{m['frames']} кадров val2017 размечены вручную вслепую от эталона.",
           "Считаются три разные вещи: точность координат (OKS), точность",
           "по каждому суставу отдельно (PCK) и согласие по флагу видимости —",
           "последнее не видно ни одной метрике по координатам.",
           "Этап A4 портфолио по контролю качества разметки.", "",
           "<!-- note:intro -->", notes.get("intro", PLACEHOLDER), "<!-- /note -->", "",
           "## Результат", "", "| | |", "|---|---|",
           f"| кадров | {m['frames']} |",
           f"| людей своих / эталонных | {m['my_people']} / {m['gt_people']} |",
           f"| сопоставлено пар | {m['matched']} |",
           f"| **OKS по общим точкам** | **{m['oks_common']:.3f}** |",
           f"| OKS COCO-style | {m['oks_coco']:.3f} |",
           f"| PCK@{m['pck_mult']:g}σ | {m['pck']:.3f} |",
           f"| **согласие по флагу видимости** | **{m['flag_agreement']:.3f}** |",
           f"| каппа Коэна по флагу | {m['flag_kappa']:.3f} |", "",
           f"Сопоставление людей — по рамке из размеченных точек, порог IoU "
           f"{m['match_threshold']}: ось независима от метрики, иначе человек",
           "с перепутанными сторонами не находит пары и худший случай просто",
           "исчезает из отчёта.", ""]

    if m.get("unmatched_gt") or m.get("unmatched_mine"):
        out += [f"Без пары: эталонных {m['unmatched_gt']}, своих "
                f"{m['unmatched_mine']} (из них {m['unmatched_mine_filtered_out']} — "
                "человек, которого отбросил фильтр эталона, "
                f"{m['unmatched_mine_invented']} — не соответствует эталону ни в чём).", ""]

    out += ["## Что метрика по координатам не видит", "",
            "Флаг видимости у точки — отдельная ось разметки, и OKS её не",
            "затрагивает вовсе: разметка с идеальными координатами, где каждая",
            "точка помечена видимой, даёт OKS ровно 1.000. Поэтому согласие по",
            "флагу считается отдельно, а рядом стоит каппа: доля совпадений",
            "высока сама по себе, если один из флагов встречается чаще прочих.", "",
            "| эталон \\ моё | v=0 не размечена | v=1 не видна | v=2 видна |",
            "|---|---|---|---|"]
    labels = ["v=0 не размечена", "v=1 не видна", "v=2 видна"]
    for i in range(3):
        row = m["flag_matrix"][i]
        out.append(f"| {labels[i]} | {row[0]} | {row[1]} | {row[2]} |")
    out.append("")
    if (review / "flag_by_joint.png").exists():
        out += [f"![расхождение по флагу]({a.review}/flag_by_joint.png)", ""]

    out += ["## Где промахивается рука", "",
            f"Допуск PCK — одна сигма COCO для этого сустава на этом масштабе,",
            "то есть вклад точки в OKS не ниже exp(-1/2) = 0.607. Порог назван",
            "заранее и под результат не подбирался.", ""]
    if (review / "pck_by_joint.png").exists():
        out += [f"![PCK по суставам]({a.review}/pck_by_joint.png)", ""]
    worst = sorted(m["pck_per_joint"].items(),
                   key=lambda kv: (kv[1]["hit"] / kv[1]["total"]
                                   if kv[1]["total"] else 1.0))[:5]
    out += ["| худшие суставы | PCK | точек | средний промах, px |", "|---|---|---|---|"]
    for name, v in worst:
        if not v["total"]:
            continue
        out.append(f"| {name} | {v['hit'] / v['total']:.2f} | {v['total']} | "
                   f"{v['mean_px']:.1f} |")
    out.append("")

    if m.get("oks_by_size"):
        out += ["| размер человека | пар | OKS |", "|---|---|---|"]
        for name in ("мелкие", "средние", "крупные"):
            b = m["oks_by_size"].get(name)
            if b:
                out.append(f"| {name} | {b['pairs']} | {b['oks']:.3f} |")
        out.append("")

    if pairs:
        out += ["## Разбор худших пар", "",
                "Синий — эталон, оранжевый — моё. Точка закрашена, если помечена",
                "видимой, и пустая, если помечена невидимой.", ""]
        for item in pairs:
            key = f"{Path(item['image']).stem}_{item['gt_id']}"
            out += [f"### {item['image']} · эталон #{item['gt_id']}", "",
                    f"OKS {item['oks']:.3f}", "",
                    f"![{key}]({a.review}/{item['file']})", "",
                    f"<!-- note:{key} -->", notes.get(key, PLACEHOLDER),
                    "<!-- /note -->", ""]

    out += ["## Что дальше", "",
            "- Инструкция и решения по спорным случаям — "
            "[annotation/GUIDELINES.md](annotation/GUIDELINES.md)",
            "- Полный отчёт — [reports/keypoint_report.md](reports/keypoint_report.md)",
            "- Долг по написанному не мной коду — [DEBT.md](DEBT.md)", "",
            "README пересобирается `tools/build_readme.py`; комментарии между маркерами",
            "`<!-- note:… -->` и `<!-- /note -->` при пересборке сохраняются.", ""]

    a.readme.write_text("\n".join(out), encoding="utf-8")
    kept = sum(1 for v in notes.values() if v != PLACEHOLDER)
    print(f"{a.readme}: разделов по парам {len(pairs)}, "
          f"сохранено комментариев {kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
