"""
Move more images into test/ WITHOUT creating train/test leakage.

Why not just move random files?
    Balanced datasets often contain augmented copies of the same X-ray
    (flips, brightness changes, re-saves). If you move one copy into test/
    and its siblings stay in train/, you recreate the leakage problem.

What this script does
    1. Hashes every image in the source split (default: train/).
    2. Groups near-duplicate images together, separately per class
       (flipped copies are matched too).
    3. Randomly picks WHOLE groups until it has about --per-class images
       of each class, so siblings always travel together.
    4. Checks the other split (val/ by default) for copies of the moved
       images and quarantines them, so nothing leaks into test.
    5. DRY RUN by default. With --apply it moves the files into
       test/<CLASS>/ (nothing is deleted; leaked val copies go to a
       quarantine folder).

Usage
    python expand_test_set.py --root "D:\\Vivek" --per-class 200            # plan only
    python expand_test_set.py --root "D:\\Vivek" --per-class 200 --apply    # do it

Keep this file in the same folder as fix_data_leakage.py (it reuses it).
After --apply, run fix_data_leakage.py once more (report mode) as a
final check, then re-run train_fedprox.py. No changes to train_fedprox.py
are needed: it reads whatever is in train/ val/ test/.
"""

import argparse
import csv
import os
import random
import shutil

import numpy as np

from fix_data_leakage import CLASS_FOLDERS, find_matches, hash_split, list_images


def group_near_duplicates(bits, flipped, threshold):
    """Union-find grouping of near-duplicates. Returns a list of index lists."""
    n = len(bits)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n - 1):
        rest = bits[i + 1:]
        dist = np.minimum((rest != bits[i]).sum(axis=1), (rest != flipped[i]).sum(axis=1))
        for j in np.where(dist <= threshold)[0]:
            union(i, i + 1 + int(j))

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def pick_groups(groups, target, rng):
    """Pick whole groups (random order) until ~target images are selected."""
    groups = groups[:]
    rng.shuffle(groups)
    slack = max(5, int(0.1 * target))
    chosen, count = [], 0
    for g in groups:
        if count >= target:
            break
        if count > 0 and count + len(g) > target + slack:
            continue                      # would overshoot too much; try a smaller group
        chosen.extend(g)
        count += len(g)
    return chosen


def count_by_class(items):
    return {c: sum(1 for _, k in items if k == c) for c in CLASS_FOLDERS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="folder containing train/ val/ test/")
    parser.add_argument("--per-class", type=int, default=200, help="images per class to ADD to test/")
    parser.add_argument("--source", choices=["train", "val"], default="train",
                        help="split to take the new test images from")
    parser.add_argument("--threshold", type=int, default=10, help="near-duplicate Hamming distance (of 256)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply", action="store_true", help="actually move files (default: plan only)")
    parser.add_argument("--quarantine", default=None)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    quarantine = args.quarantine or (root.rstrip("\\/") + "_leakage_removed")
    other = "val" if args.source == "train" else "train"
    rng = random.Random(args.seed)

    src_items = list_images(os.path.join(root, args.source))
    other_items = list_images(os.path.join(root, other))
    test_items = list_images(os.path.join(root, "test"))

    if not src_items:
        raise SystemExit(f"No images found in {os.path.join(root, args.source)}")

    print(f"{args.source}: {len(src_items)} {count_by_class(src_items)}")
    print(f"{other}: {len(other_items)} {count_by_class(other_items)}")
    print(f"test: {len(test_items)} {count_by_class(test_items)}")
    print(f"\nHashing {args.source}/ ...")
    src_bits, src_flip = hash_split(src_items, args.source)

    # ---------------- choose whole near-duplicate groups per class ----------------
    selected = []
    for cls in CLASS_FOLDERS:
        idx = [i for i, (_, c) in enumerate(src_items) if c == cls]
        if not idx:
            continue
        print(f"Grouping {len(idx)} {cls} images...")
        local = group_near_duplicates(src_bits[idx], src_flip[idx], args.threshold)
        groups = [[idx[k] for k in g] for g in local]
        picked = pick_groups(groups, args.per_class, rng)
        selected.extend(picked)
        n_dup = len(idx) - len(groups)
        print(f"  {cls}: {len(groups)} unique groups ({n_dup} near-duplicate copies); "
              f"selected {len(picked)} images (target {args.per_class})")

    selected = sorted(set(selected))
    selected_items = [src_items[i] for i in selected]
    sel_bits, sel_flip = src_bits[selected], src_flip[selected]

    # Safety: the source split must keep enough images
    left = len(src_items) - len(selected)
    left_by_class = {c: count_by_class(src_items)[c] - count_by_class(selected_items)[c] for c in CLASS_FOLDERS}
    if args.source == "train" and any(v < 100 for v in left_by_class.values()):
        print(f"\nWARNING: train/ would keep only {left_by_class} images per class. Lower --per-class.")

    # ---------------- copies of moved images in the other split ----------------
    print(f"\nChecking {other}/ for copies of the images being moved...")
    other_bits, _ = hash_split(other_items, other)
    leaked = find_matches(other_bits, selected_items, sel_bits, sel_flip, args.threshold)
    print(f"  {len(leaked)} {other}/ images duplicate a moved image -> will be quarantined")

    # ---------------- plan report ----------------
    plan_path = os.path.join(os.getcwd(), "test_expansion_plan.csv")
    with open(plan_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["action", "class", "file", "note"])
        for path, cls in selected_items:
            w.writerow([f"move {args.source} -> test", cls, path, ""])
        for i, (match, dist) in sorted(leaked.items()):
            path, cls = other_items[i]
            w.writerow([f"quarantine from {other}", cls, path, f"copy of {match} (distance {dist})"])

    new_counts = count_by_class(test_items)
    for _, c in selected_items:
        new_counts[c] += 1
    print("\n" + "=" * 56)
    print("PLAN")
    print("=" * 56)
    print(f"Move {len(selected_items)} images {args.source}/ -> test/  {count_by_class(selected_items)}")
    print(f"New test size: {len(test_items) + len(selected_items)}  {new_counts}")
    print(f"Plan written to: {plan_path}")

    if not args.apply:
        print("\nDRY RUN - nothing moved. Re-run with --apply to execute.")
        return

    # ---------------- apply ----------------
    for path, cls in selected_items:
        dest_dir = os.path.join(root, "test", cls)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(path))
        if os.path.exists(dest):
            dest = os.path.join(dest_dir, f"from_{args.source}_{os.path.basename(path)}")
        shutil.move(path, dest)

    for i in leaked:
        path, cls = other_items[i]
        dest_dir = os.path.join(quarantine, other, cls)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(path))
        if os.path.exists(dest):
            base, ext = os.path.splitext(os.path.basename(path))
            dest = os.path.join(dest_dir, f"{base}_{i}{ext}")
        shutil.move(path, dest)

    print("\nDone. Images per split now:")
    for split in ("train", "val", "test"):
        items = list_images(os.path.join(root, split))
        print(f"  {split:<6}: {len(items):>5}  {count_by_class(items)}")
    print("\nNext: run fix_data_leakage.py (report mode) as a final check, then retrain.")


if __name__ == "__main__":
    main()
