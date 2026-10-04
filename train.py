"""Single entry point for SCPT all-class few-shot training and evaluation."""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Directory containing the dataset folders")
    parser.add_argument(
        "--dataset", default="oxford_pets", help="Dataset key; see docs/DATASETS.md"
    )
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-file", default=str(ROOT / "configs/scpt.yaml"))
    parser.add_argument("--split-dir", default=str(ROOT / "splits"))
    parser.add_argument(
        "--clip-checkpoint", default="", help="Optional official ViT-B-16.pt for offline use"
    )
    parser.add_argument("--cache-dir", default=os.environ.get("CLIP_CACHE_DIR", "~/.cache/clip"))
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model-dir", default="")
    parser.add_argument(
        "--load-epoch", type=int, default=None, help="Default: best validation checkpoint"
    )
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="Optional KEY VALUE configuration overrides"
    )
    return parser.parse_args()


def setup_cfg(args):
    from dassl.config import get_cfg_default
    from yacs.config import CfgNode as CN

    from datasets import DATASETS

    if args.dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {args.dataset}; choose from {', '.join(DATASETS)}")
    if args.shots < 1 or args.seed < 0:
        raise ValueError("--shots must be positive and --seed must be nonnegative")
    cfg = get_cfg_default()
    for name in list(cfg.TRAINER):
        if name != "NAME":
            del cfg.TRAINER[name]
    cfg.TRAINER.SCPT = CN(dict(N_CTX=4, EXTRA_BITS=3, TAU=0.3, LOSS_WEIGHT=8.5, PREC="fp16"))
    cfg.DATASET.KEY = args.dataset
    cfg.DATASET.SPLIT_DIR = str(Path(args.split_dir).expanduser().resolve())
    cfg.MODEL.CLIP_CHECKPOINT = args.clip_checkpoint
    cfg.MODEL.CACHE_DIR = args.cache_dir
    cfg.OPTIM.ADAM_EPS = 0.001
    cfg.merge_from_file(args.config_file)
    cfg.DATASET.NAME = "FineGrainedDataset"
    cfg.DATASET.ROOT = str(Path(args.root).expanduser().resolve())
    cfg.DATASET.NUM_SHOTS = args.shots
    cfg.OUTPUT_DIR = str(Path(args.output_dir).expanduser().resolve())
    cfg.SEED = args.seed
    cfg.TRAINER.NAME = "SCPT"
    cfg.merge_from_list(args.opts)
    if cfg.TRAINER.NAME != "SCPT" or cfg.DATASET.NAME != "FineGrainedDataset":
        raise ValueError("Only SCPT all-class few-shot experiments are supported")
    if cfg.DATASET.KEY != args.dataset or cfg.DATASET.NUM_SHOTS < 1 or cfg.SEED < 0:
        raise ValueError("Use --dataset, a positive shot count, and a nonnegative seed")
    if cfg.TRAINER.SCPT.TAU < 0 or cfg.TRAINER.SCPT.LOSS_WEIGHT <= 0:
        raise ValueError("TAU must be nonnegative and LOSS_WEIGHT positive")
    if cfg.OPTIM.MAX_EPOCH < 1 or cfg.DATALOADER.TRAIN_X.BATCH_SIZE < 1:
        raise ValueError("Epoch count and training batch size must be positive")
    cfg.freeze()
    return cfg


def main(args):
    import torch
    from dassl.engine import build_trainer
    from dassl.utils import collect_env_info, set_random_seed, setup_logger

    import datasets  # noqa: F401 (registry)
    import trainers.scpt  # noqa: F401 (registry)

    cfg = setup_cfg(args)
    output = Path(cfg.OUTPUT_DIR)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}; use a new directory")
    if args.eval_only and not args.model_dir:
        raise ValueError("--eval-only requires --model-dir")
    setup_logger(cfg.OUTPUT_DIR)
    set_random_seed(cfg.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(cfg)
    (output / "config.yaml").write_text(cfg.dump(), encoding="utf-8")
    print(collect_env_info())
    trainer = build_trainer(cfg)
    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
    else:
        trainer.train()


if __name__ == "__main__":
    main(parse_args())
