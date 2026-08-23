#!/usr/bin/env python3
"""Скачивает эталон COCO Keypoints — person_keypoints_val2017.json (P4d, шаг 0).

Файла нет в том, что распаковано для P2: там лежит только instances_val2017.json.
Официальный источник отдаёт его лишь внутри архива annotations_trainval2017.zip
на 241 МБ, поэтому берём зеркало HuggingFace, раздающее файлы поштучно:
10 МБ вместо 241.

Сеть рвётся: на A3 загрузка обрывалась на середине с Network is unreachable.
Поэтому повторные попытки с паузой, а уже скачанный файл не перекачивается.

    python3 fetch_keypoints.py --out data/coco
"""

import argparse
import time
import urllib.error
import urllib.request
from pathlib import Path

MIRROR = ("https://huggingface.co/datasets/merve/coco/resolve/main/annotations/"
          "person_keypoints_val2017.json")
EXPECTED_BYTES = 10_020_657


def download(url: str, dest: Path, attempts: int = 4) -> int:
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, dest.open("wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            return dest.stat().st_size
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if dest.exists():
                dest.unlink()
            if attempt == attempts:
                raise
            print(f"  попытка {attempt} сорвалась ({e}), повтор через {2 * attempt} с")
            time.sleep(2 * attempt)
    raise RuntimeError("недостижимо")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("data/coco"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "person_keypoints_val2017.json"

    if dest.exists() and dest.stat().st_size == EXPECTED_BYTES:
        print(f"{dest} уже на месте, {dest.stat().st_size} б")
        return 0

    print(f"качаю {dest.name} с зеркала HuggingFace")
    size = download(MIRROR, dest)
    print(f"{dest} {size} б")
    if size != EXPECTED_BYTES:
        print(f"размер отличается от ожидаемого ({EXPECTED_BYTES} б) — "
              "зеркало могло обновиться, проверь файл прежде чем считать")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
