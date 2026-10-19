"""
Find and remove train/val/test leakage in the Chest X-Ray Pneumonia
Balanced dataset.

Balanced datasets are often built by augmenting images (flips, brightness
changes, re-saves). If copies of one X-ray end up in different splits, the
model is tested on images it has already seen and scores look perfect.

What this script does
---------------------
1. Computes a perceptual hash (16x16 difference hash) for every image.
2. Compares splits in priority order  test > val > train.
   - any val image that matches a test image is flagged
   - any train image that matches a test or val image is flagged
   Matching also checks the horizontally-flipped version of each image.
3. DRY RUN by default: only prints a report and writes leakage_report.csv.
   With --apply, flagged files are MOVED (not deleted) into a quarantine
   folder, so you can restore them if needed.

Usage
-----
    python fix_data_leakage.py --root "D:\\Vivek"            # report only
    python fix_data_leakage.py --root "D:\\Vivek" --apply    # move leaks

Options
-------
    --threshold N   max Hamming distance (out of 256 bits) to count as a
                    duplicate. Default 10. Lower = stricter (fewer matches).
    --quarantine P  where flagged files are moved (default: <root>_leakage_removed)

Limits: catches exact copies, re-saves, brightness/contrast tweaks and
horizontal flips. It will NOT catch heavy rotations or crops. If scores are
still suspiciously perfect after cleaning, check for those too.
"""

import argparse
import csv
import os
import shutil

import numpy as np
from PIL import Image, ImageOps

SPLITS_PRIORITY = ["test", "val", "train"]   # earlier = protected
CLASS_FOLDERS = ["NORMAL", "PNEUMONIA"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
HASH_SIZE = 16                                # 16x16 -> 256-bit hash


def list_images(split_dir):
    """Return [(path, class_folder_name)] for one split (case-insensitive classes)."""
    items = []
    if not os.path.isdir(split_dir):
        return items

    lookup = {n.lower(): n for n in os.listdir(split_dir)
              if os.path.isdir(os.path.join(split_dir, n))}

    for cls in CLASS_FOLDERS:
        actual = lookup.get(cls.lower())
        if actual is None:
            continue
        folder = os.path.join(split_dir, actual)
        for name in sorted(os.listdir(folder)):
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                items.append((os.path.join(folder, name), cls))
    return items


def dhash_bits(img):
    """16x16 difference hash as a flat uint8 array of 256 bits."""
    small = img.convert("L").resize((HASH_SIZE + 1, HASH_SIZE), Image.LANCZOS)
    arr = np.asarray(small, dtype=np.int16)
    return (arr[:, 1:] > arr[:, :-1]).astype(np.uint8).flatten()


def hash_image(path):
    """Return (hash of image, hash of horizontally flipped image)."""
    with Image.open(path) as img:
        img = ImageOps.autocontrast(img.convert("L"))   # robust to brightness/contrast edits
        return dhash_bits(img), dhash_bits(ImageOps.mirror(img))


def hash_split(items, label):
    hashes, flipped = [], []
    for i, (path, _) in enumerate(items, 1):
        try:
            h, hf = hash_image(path)
        except Exception as exc:
            print(f"  [skip] could not read {path}: {exc}")
            # impossible bit values -> an unreadable file never matches anything
            h = np.full(HASH_SIZE * HASH_SIZE, 2, dtype=np.uint8)
            hf = np.full(HASH_SIZE * HASH_SIZE, 3, dtype=np.uint8)
        hashes.append(h)
        flipped.append(hf)
        if i % 500 == 0:
            print(f"  hashed {i}/{len(items)} {label} images")
    if not hashes:
        empty = np.zeros((0, HASH_SIZE * HASH_SIZE), dtype=np.uint8)
        return empty, empty
    return np.stack(hashes), np.stack(flipped)


def find_matches(low_bits, high_items, high_bits, high_flipped, threshold):
    """
    Return {low_index: (high_path, distance)} for every image in the lower
    priority split that matches an image in the higher priority split.
    """
    matches = {}
    if len(low_bits) == 0:
        return matches

    for j, (high_path, _) in enumerate(high_items):
        for query in (high_bits[j], high_flipped[j]):
            dist = (low_bits != query).sum(axis=1)
            for i in np.where(dist <= threshold)[0]:
                i = int(i)
                if i not in matches or dist[i] < matches[i][1]:
                    matches[i] = (high_path, int(dist[i]))
    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="folder containing train/ val/ test/")
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--apply", action="store_true", help="move flagged files (default: report only)")
    parser.add_argument("--quarantine", default=None)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    quarantine = args.quarantine or (root.rstrip("\\/") + "_leakage_removed")

    items, bits, flipped = {}, {}, {}
    for split in SPLITS_PRIORITY:
        items[split] = list_images(os.path.join(root, split))
        print(f"{split}: {len(items[split])} images - hashing...")
        bits[split], flipped[split] = hash_split(items[split], split)

    # flagged[split] = {index: (matched_path, distance)}
    flagged = {s: {} for s in SPLITS_PRIORITY}

    for li, low in enumerate(SPLITS_PRIORITY):
        for high in SPLITS_PRIORITY[:li]:
            found = find_matches(bits[low], items[high], bits[high], flipped[high], args.threshold)
            for idx, hit in found.items():
                if idx not in flagged[low] or hit[1] < flagged[low][idx][1]:
                    flagged[low][idx] = hit

    # ---------------- report ----------------
    report_path = os.path.join(os.getcwd(), "leakage_report.csv")
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class", "file", "matches_file", "hamming_distance"])
        for split in SPLITS_PRIORITY:
            for idx, (match_path, dist) in sorted(flagged[split].items()):
                path, cls = items[split][idx]
                writer.writerow([split, cls, path, match_path, dist])

    print("\n" + "=" * 56)
    print("LEAKAGE REPORT")
    print("=" * 56)
    total = 0
    for split in SPLITS_PRIORITY:
        n = len(flagged[split])
        total += n
        pct = 100 * n / max(1, len(items[split]))
        print(f"{split:<6}: {n:>5} of {len(items[split]):>5} images duplicate a higher-priority split ({pct:.1f}%)")
    print(f"Report written to: {report_path}")

    if total == 0:
        print("\nNo cross-split duplicates found at this threshold.")
        print("If results are still ~100%, the test set may simply be too small/easy,")
        print("or leakage involves rotations/crops/same-patient images.")
        return

    if not args.apply:
        print("\nDRY RUN - nothing moved. Re-run with --apply to move these files to:")
        print(f"  {quarantine}")
        return

    # ---------------- apply ----------------
    moved = 0
    for split in SPLITS_PRIORITY:
        for idx in flagged[split]:
            path, cls = items[split][idx]
            dest_dir = os.path.join(quarantine, split, cls)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, os.path.basename(path))
            if os.path.exists(dest):
                base, ext = os.path.splitext(os.path.basename(path))
                dest = os.path.join(dest_dir, f"{base}_{idx}{ext}")
            shutil.move(path, dest)
            moved += 1

    print(f"\nMoved {moved} files to {quarantine}")
    print("\nImages remaining per split:")
    for split in SPLITS_PRIORITY:
        remaining = list_images(os.path.join(root, split))
        n_norm = sum(1 for _, c in remaining if c == "NORMAL")
        n_pneu = sum(1 for _, c in remaining if c == "PNEUMONIA")
        print(f"  {split:<6}: {len(remaining):>5}  (NORMAL={n_norm}, PNEUMONIA={n_pneu})")
    print("\nNow re-run train_fedprox.py on the cleaned folders.")


if __name__ == "__main__":
    main()
