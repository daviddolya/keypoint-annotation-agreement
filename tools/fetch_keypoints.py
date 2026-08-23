#!/usr/bin/env python3
"""Downloads the ground truth: person_keypoints_val2017.json.

The official source only ships it inside annotations_trainval2017.zip, which
is 241 MB, so this pulls from a HuggingFace mirror that serves the files one
by one: 10 MB instead of 241.

Networks drop: on an earlier stage the download died halfway through with
"Network is unreachable". Hence the retries with a pause, and an already
downloaded file is never fetched again.

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
            print(f"  attempt {attempt} failed ({e}), retrying in {2 * attempt} s")
            time.sleep(2 * attempt)
    raise RuntimeError("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("data/coco"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "person_keypoints_val2017.json"

    if dest.exists() and dest.stat().st_size == EXPECTED_BYTES:
        print(f"{dest} already in place, {dest.stat().st_size} bytes")
        return 0

    print(f"downloading {dest.name} from the HuggingFace mirror")
    size = download(MIRROR, dest)
    print(f"{dest} {size} bytes")
    if size != EXPECTED_BYTES:
        print(f"size differs from the expected {EXPECTED_BYTES} bytes -- the "
              "mirror may have been updated, check the file before computing")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
