"""Prepare user-owned split files from class folders or an existing JSON manifest."""

import argparse
import json
import random
from pathlib import Path

from .finegrained import DATASETS, read_manifest, validate_class_count

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def split_class_folders(image_root, seed=0):
    """Split each class 50%/20%/30% using sorted paths and one seeded generator."""
    classes = sorted(p for p in image_root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not classes:
        raise ValueError("No class folders found; use --source-split for flat image layouts")
    rng = random.Random(seed)
    splits = {name: [] for name in ("train", "val", "test")}
    for label, directory in enumerate(classes):
        files = sorted(
            p.relative_to(image_root).as_posix()
            for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        n_train, n_val = round(0.5 * len(files)), round(0.2 * len(files))
        if min(n_train, n_val, len(files) - n_train - n_val) < 1:
            raise ValueError(f"Not enough images to split class {directory.name}")
        rng.shuffle(files)
        for split, paths in (
            ("train", files[:n_train]),
            ("val", files[n_train : n_train + n_val]),
            ("test", files[n_train + n_val :]),
        ):
            splits[split].extend([[path, label, directory.name] for path in paths])
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Dataset parent directory")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--output-dir", default="splits")
    parser.add_argument(
        "--seed", type=int, default=0, help="Fixed split seed, independent of training seeds"
    )
    parser.add_argument(
        "--source-split", type=Path, help="Existing Zhou-style JSON or JSON.GZ manifest"
    )
    args = parser.parse_args()
    _, directory, subdir = DATASETS[args.dataset]
    image_root = Path(args.root).expanduser().resolve() / directory / subdir
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    destination = Path(args.output_dir).expanduser().resolve() / f"{args.dataset}.json"
    if destination.exists() or destination.with_suffix(".json.gz").exists():
        raise FileExistsError(f"Split already exists: {destination}; choose another --output-dir")
    if args.source_split:
        splits = read_manifest(args.source_split.expanduser())
    else:
        splits = split_class_folders(image_root, args.seed)
    validate_class_count(splits, args.dataset)
    for rows in splits.values():
        for relative, _, _ in rows:
            if not (image_root / relative).is_file():
                raise FileNotFoundError(image_root / relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects an existing split even if another job races us.
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(splits, handle, ensure_ascii=False, indent=2)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
