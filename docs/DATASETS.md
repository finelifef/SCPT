# Dataset preparation

Download datasets from their original providers. Set `DATA_ROOT` to the parent
directory containing the folders below. No images or split files are shipped.

| Dataset key | Image root relative to DATA_ROOT | Classes |
|---|---|---:|
| `oxford_pets` | `oxford_pets/images/` | 37 |
| `oxford_flowers` | `oxford_flowers/jpg/` | 102 |
| `stanford_cars` | `stanford_cars/` | 196 |
| `fgvc_aircraft` | `fgvc_aircraft/images/` | 100 |
| `food101` | `food101/images/` | 101 |
| `food172` | `food172/images/` | 172 |
| `food200` | `food200/images/` | 200 |
| `food500` | `food500/images/` | 500 |
| `dog_breed` | `dog_breed/images/` | 120 |
| `stanford_dogs` | `stanford_dogs/images/` | 120 |
| `cub200` | `cub200/images/` | 200 |
| `fruit92` | `fruit92/images/` | 92 |
| `veg200` | `veg200/images/` | 200 |
| `webcar` | `webcar/images/` | 196 |

## Prepare from class folders

For datasets stored as `<image_root>/<class_name>/<image_file>`, run:

```bash
python -m datasets.prepare --root /path/to/datasets --dataset food172 \
  --output-dir splits --seed 0
```

The utility sorts classes and filenames, then shuffles each class using a fixed
seed. It assigns 50% of each class to training, 20% to validation, and the remainder
to testing. The generated file is `splits/food172.json`. Class-folder names are used
as textual class names; use meaningful category names rather than numeric IDs.
The preparation seed is independent of the few-shot training seeds.

## Prepare from existing annotations

For flat image layouts or datasets with designated annotations/splits, provide
a JSON manifest rather than inferring classes from directories. This applies,
for example, to the standard Oxford Pets, Oxford Flowers, FGVC Aircraft and
Stanford Cars layouts. CoOp's standard dataset preparation instructions are
[available upstream](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).

```bash
python -m datasets.prepare --root /path/to/datasets --dataset oxford_pets \
  --source-split /path/to/split_zhou_OxfordPets.json --output-dir splits
```

The manifest must have three lists of `[relative_image_path, label, class_name]`:

```json
{
  "train": [["class_a/train_image.jpg", 0, "class_a"]],
  "val": [["class_a/val_image.jpg", 0, "class_a"]],
  "test": [["class_a/test_image.jpg", 0, "class_a"]]
}
```

This abbreviated example shows the schema; supply all classes and images.
Paths are relative to the image root in the table, use `/` separators, and must
not be absolute paths. Labels must be contiguous integers starting at zero,
and each label must map to the same class name across all splits. All three
splits must contain the expected number of classes, with no shared image paths.
The preparation utility validates the class count and file existence.

Existing output split files are not overwritten. Use another `--output-dir`
for a different split. The training loader also accepts `<dataset_key>.json.gz`.
Set `SPLIT_DIR` when using a split directory other than the repository's `splits/`.

## Few-shot sampling

Training draws K images per class from the training split and `min(K,4)` per class
from the validation split, controlled by the run seed. The full test split is
used for evaluation. Ensure each class has enough examples. Checkpoint selection
uses validation accuracy; test samples are never used for model selection.
