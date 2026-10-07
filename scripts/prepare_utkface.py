"""Prepare UTKFace for Chameleon.

Reproduces PreProcessor/utkface_tools/*.ipynb in one pass:
  * parse `<age>_<gender>_<race>_<date>.jpg[.chip.jpg]`, drop race 4 ("others") and malformed names
  * bucket age into the 9 age groups of PreProcessor/utkface_tools/config.json
  * center-crop to a square, resize, save as PNG (the image-edit APIs require square PNGs)
  * write the dataset table ImageAnalyzer reads: filename,age_group,gender,race,is_generated

Output layout (under --data-dir, i.e. CHAMELEON_DATA_DIR):
  resources/<name>/<age_group>_<gender>_<race>_<date>.png
  datasets/<name>.csv

Usage:
  python3 scripts/prepare_utkface.py --src ~/Downloads/UTKFace --data-dir ./data

To reproduce the paper's exact 18,978-image set (the authors applied only their size filter
to the in-the-wild images, and their list is the ground truth):
  docker run --rm --platform linux/amd64 --entrypoint cat merfanian/fairness-lens:0.2.1 data/images.csv > paper_images.csv
  python3 scripts/prepare_utkface.py --src ~/Downloads/UTKFace --data-dir ./data \
      --only-in paper_images.csv --max-aspect 100 --max-side 100000
"""
import argparse
import csv
import json
import os
from pathlib import Path

from PIL import Image

AGE_GROUPS = json.loads((Path(__file__).resolve().parent.parent /
                         "PreProcessor/utkface_tools/config.json").read_text())["age_groups"]


def age_to_group(age: int):
    for g in AGE_GROUPS:
        if g["start_age"] <= age < g["end_age"]:
            return g["id"]
    return None


def parse(name: str):
    parts = name.split(".")[0].split("_")
    if len(parts) != 4 or not all(p.isdigit() for p in parts):
        return None
    age, gender, race, date = parts
    if race == "4":
        return None
    group = age_to_group(int(age))
    if group is None:
        return None
    return group, int(gender), int(race), date


def square(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((size, size), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory with raw UTKFace .jpg files (searched recursively)")
    ap.add_argument("--data-dir", default="data", help="CHAMELEON_DATA_DIR")
    ap.add_argument("--name", default="utkface", help="dataset name; must match config.json and contain no '_'")
    ap.add_argument("--size", type=int, default=512, help="square side in px")
    ap.add_argument("--max-aspect", type=float, default=1.5,
                    help="skip images whose long/short side ratio exceeds this (paper: 1.5)")
    ap.add_argument("--min-side", type=int, default=256,
                    help="skip images with a side below this (authors' preprocess_images.ipynb: 256)")
    ap.add_argument("--max-side", type=int, default=1280,
                    help="skip images with a side above this (authors' preprocess_images.ipynb: 1280)")
    ap.add_argument("--only-in", help="CSV with a `filename` column; keep only those images. The authors' list is "
                                      "/app/data/images.csv in merfanian/fairness-lens:0.2.1 (18,978 rows)")
    args = ap.parse_args()
    assert "_" not in args.name, "ImageAnalyzer derives the parent dataset from the id prefix before '_'"

    out_img = Path(args.data_dir) / "resources" / args.name
    out_img.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.data_dir) / "datasets" / f"{args.name}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    paths = sorted((os.path.join(d, f), f) for d, _, fs in os.walk(args.src) for f in fs
                   if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith("."))
    keep = None
    if args.only_in:
        with open(args.only_in) as fh:
            keep = {r["filename"] for r in csv.DictReader(fh)}
    rows, seen, skipped = [], set(), 0
    for path, f in paths:
        parsed = parse(f)
        if parsed is None:
            skipped += 1
            continue
        group, gender, race, date = parsed
        filename = f"{group}_{gender}_{race}_{date}.png"
        if filename in seen or (keep is not None and filename not in keep):
            skipped += 1
            continue
        img = Image.open(path)
        w, h = img.size
        if min(w, h) < args.min_side or max(w, h) > args.max_side or max(w, h) / min(w, h) > args.max_aspect:
            skipped += 1
            continue
        square(img.convert("RGB"), args.size).save(out_img / filename)
        seen.add(filename)
        rows.append((filename, group, gender, race, False))

    with open(out_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["filename", "age_group", "gender", "race", "is_generated"])
        wr.writerows(rows)
    print(f"wrote {len(rows)} images to {out_img} and {out_csv} (skipped {skipped})")


if __name__ == "__main__":
    main()
