# SCPT

**Structured-Condensed Prompt Tuning in Vision-Language Models for Fine-grained Image Recognition**

Xinda Liu, Qinyu Zhang, Weiqing Min, Guohua Geng, Shuqiang Jiang

[Paper](https://arxiv.org/abs/2607.06185) · [Dataset preparation](docs/DATASETS.md)

SCPT combines Semantic Relation Encoding (SRE) and Semantic Condensation Loss
(ScLoss) for few-shot adaptation of CLIP. This repository contains the SCPT
implementation and one few-shot experiment script. Datasets, pretrained model
weights, trained checkpoints, and experiment results are not included.

## Installation

Use Python 3.11 and a PyTorch build compatible with your GPU. For CUDA 12.8:

```bash
conda create -n scpt python=3.11 -y
conda activate scpt
python -m pip install "setuptools<81" wheel
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps "git+https://github.com/KaiyangZhou/Dassl.pytorch.git@c61a1b570ac6333bd50fb5ae06aea59002fb20bb"
```

A separate `clip` package or custom CUDA SVD extension is not required.

## Data preparation

Download the datasets separately and follow [docs/DATASETS.md](docs/DATASETS.md)
to prepare your image directories and split files. Food172 uses 172 classes.

For datasets organized into class folders:

```bash
python -m datasets.prepare --root /path/to/datasets --dataset food172 --output-dir splits
```

For existing JSON splits, use `--source-split /path/to/split.json` as described in
the dataset guide. Prepared files remain local and are ignored by Git.

## Few-shot training

```bash
DATA_ROOT=/path/to/datasets CUDA_VISIBLE_DEVICES=0 bash scripts/fewshot.sh food172
```

The default configuration uses ViT-B/16, 16 shots per class, seeds 1/2/3, 50 epochs,
and batch size 32. Configuration values are in `configs/scpt.yaml`.

```bash
DATA_ROOT=/path/to/datasets SPLIT_DIR=/path/to/splits OUTPUT_ROOT=/path/to/output \
SHOTS=16 SEEDS="1 2 3" CUDA_VISIBLE_DEVICES=0 \
bash scripts/fewshot.sh food172 DATALOADER.NUM_WORKERS 4
```

Set `PYTHON` to select another interpreter. Paths are configurable; the script
can run from any working directory. Use a new output directory for each run.

The official CLIP ViT-B/16 weights are downloaded at runtime into
`CLIP_CACHE_DIR` (default: `~/.cache/clip`). For offline use, obtain the official
weights separately and set `CLIP_CHECKPOINT=/path/to/ViT-B-16.pt`.

## Evaluation

Evaluate a checkpoint produced by your own training run:

```bash
python train.py --root /path/to/datasets --dataset food172 --shots 16 --seed 1 \
  --eval-only --model-dir output/food172/16shot/seed1 \
  --output-dir output/eval/food172/seed1
```

The best validation checkpoint is selected by default. Use `--load-epoch 50`
to evaluate the final-epoch checkpoint instead.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Citation

```bibtex
@article{liu2026scpt,
  title={Structured-Condensed Prompt Tuning in Vision-Language Models for Fine-grained Image Recognition},
  author={Liu, Xinda and Zhang, Qinyu and Min, Weiqing and Geng, Guohua and Jiang, Shuqiang},
  journal={arXiv preprint arXiv:2607.06185},
  year={2026}
}
```

Built on [TCP](https://github.com/htyao89/Textual-based_Class-aware_prompt_tuning),
[CoOp](https://github.com/KaiyangZhou/CoOp),
[Dassl](https://github.com/KaiyangZhou/Dassl.pytorch), and
[CLIP](https://github.com/openai/CLIP). See [NOTICE.md](NOTICE.md) and [LICENSE](LICENSE).
