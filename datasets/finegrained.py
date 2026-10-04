"""Read fixed Zhou-style JSON splits; never re-split data or load old pickle caches."""

import gzip
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path, PurePosixPath

from dassl.data.datasets import DATASET_REGISTRY, DatasetBase, Datum

# CLI name: (Dassl name, dataset directory, image directory relative to dataset)
DATASETS = {
    "oxford_pets": ("OxfordPets", "oxford_pets", "images"),
    "oxford_flowers": ("OxfordFlowers", "oxford_flowers", "jpg"),
    "stanford_cars": ("StanfordCars", "stanford_cars", "."),
    "fgvc_aircraft": ("FGVCAircraft", "fgvc_aircraft", "images"),
    "food101": ("Food101", "food101", "images"),
    "food172": ("Food172", "food172", "images"),
    "food200": ("Food200", "food200", "images"),
    "food500": ("Food500", "food500", "images"),
    "dog_breed": ("DogBreed", "dog_breed", "images"),
    "stanford_dogs": ("StanfordDogs", "stanford_dogs", "images"),
    "cub200": ("Cub200", "cub200", "images"),
    "fruit92": ("Fruit92", "fruit92", "images"),
    "veg200": ("Veg200", "veg200", "images"),
    "webcar": ("WebCar", "webcar", "images"),
}

EXPECTED_CLASSES = dict(
    zip(DATASETS, (37, 102, 196, 100, 101, 172, 200, 500, 120, 120, 200, 92, 200, 196))
)


def validate_class_count(splits, key):
    count = len({row[1] for row in splits["train"]})
    expected = EXPECTED_CLASSES[key]
    if count != expected:
        raise ValueError(f"{key} requires {expected} classes; the supplied split contains {count}")


def read_manifest(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        splits = json.load(handle)
    names = {}
    seen = set()
    for split in ("train", "val", "test"):
        if not splits.get(split):
            raise ValueError(f"Missing or empty {split} split in {path}")
        labels = set()
        for relative, label, classname in splits[split]:
            relative_path = PurePosixPath(relative)
            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
                or "\\" in relative
                or ":" in relative
            ):
                raise ValueError(f"Split paths must be safe relative POSIX paths: {relative}")
            if relative in seen:
                raise ValueError(f"Duplicate image or train/val/test overlap: {relative}")
            seen.add(relative)
            if (
                not isinstance(label, int)
                or label < 0
                or not isinstance(classname, str)
                or not classname
            ):
                raise ValueError(f"Invalid label or class name: {relative}")
            if label in names and names[label] != classname:
                raise ValueError(f"Inconsistent class name for label {label}")
            names[label] = classname
            labels.add(label)
        if split == "train":
            training_labels = labels
        elif labels != training_labels:
            raise ValueError("All-class few-shot splits must contain the same classes")
    if set(names) != set(range(len(names))):
        raise ValueError("Labels must be contiguous, zero-based integers")
    return splits


def few_shot(rows, shots, rng):
    """Use Dassl's insertion-order class grouping and Python random.sample rule."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[1]].append(row)
    sampled = []
    for label, items in grouped.items():
        if len(items) < shots:
            raise ValueError(
                f"Class {label} has {len(items)} images, fewer than requested {shots} shots"
            )
        sampled.extend(rng.sample(items, shots))
    return sampled


@DATASET_REGISTRY.register()
class FineGrainedDataset(DatasetBase):
    def __init__(self, cfg):
        key = cfg.DATASET.KEY
        _, folder, image_folder = DATASETS[key]
        root = Path(cfg.DATASET.ROOT).expanduser().resolve() / folder / image_folder
        if not root.is_dir():
            raise FileNotFoundError(f"Missing image directory: {root}; see docs/DATASETS.md")
        path = Path(cfg.DATASET.SPLIT_DIR).expanduser() / f"{key}.json.gz"
        if not path.exists():
            path = path.with_suffix("")
        if not path.exists():
            raise FileNotFoundError(f"Missing fixed split: {path}; see docs/DATASETS.md")
        splits = read_manifest(path)
        validate_class_count(splits, key)
        rng = random.Random(cfg.SEED)
        selected = {
            "train": few_shot(splits["train"], cfg.DATASET.NUM_SHOTS, rng),
            "val": few_shot(splits["val"], min(cfg.DATASET.NUM_SHOTS, 4), rng),
            "test": splits["test"],
        }
        # Log portable sample choices for auditing, without touching dataset files.
        output = Path(cfg.OUTPUT_DIR)
        output.mkdir(parents=True, exist_ok=True)
        sampling = {
            "dataset": key,
            "seed": cfg.SEED,
            "shots": cfg.DATASET.NUM_SHOTS,
            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "train": selected["train"],
            "val": selected["val"],
        }
        (output / "fewshot_selection.json").write_text(
            json.dumps(sampling, indent=2), encoding="utf-8"
        )

        def convert(rows):
            items = []
            for relative, label, name in rows:
                image = root / relative
                if not image.is_file():
                    raise FileNotFoundError(f"Image from fixed split not found: {image}")
                items.append(Datum(impath=str(image), label=label, classname=name))
            return items

        super().__init__(
            train_x=convert(selected["train"]),
            val=convert(selected["val"]),
            test=convert(selected["test"]),
        )
