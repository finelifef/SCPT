"""Explicit optimizers and a current-PyTorch-compatible warmup/cosine schedule."""

import math

import torch


def build_optimization(parameters, cfg):
    """Match the source environment without requiring private Dassl patches."""
    if cfg.NAME == "adam":
        optimizer = torch.optim.Adam(
            parameters,
            lr=cfg.LR,
            weight_decay=cfg.WEIGHT_DECAY,
            betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2),
            eps=cfg.ADAM_EPS,
        )
    elif cfg.NAME == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=cfg.LR,
            momentum=cfg.MOMENTUM,
            weight_decay=cfg.WEIGHT_DECAY,
            dampening=cfg.SGD_DAMPNING,
            nesterov=cfg.SGD_NESTEROV,
        )
    else:
        raise ValueError("Supported optimizers: adam and sgd")
    if cfg.LR <= 0 or cfg.ADAM_EPS <= 0:
        raise ValueError("Learning rate and Adam epsilon must be positive")
    if cfg.LR_SCHEDULER != "cosine" or cfg.WARMUP_TYPE != "constant" or not cfg.WARMUP_RECOUNT:
        raise ValueError("SCPT uses cosine decay with constant warmup and WARMUP_RECOUNT=True")
    if cfg.WARMUP_EPOCH < 0 or cfg.WARMUP_EPOCH > cfg.MAX_EPOCH:
        raise ValueError("Warmup must be between zero and MAX_EPOCH")

    def factor(epoch):
        if epoch < cfg.WARMUP_EPOCH:
            return cfg.WARMUP_CONS_LR / cfg.LR
        # Source Dassl's successor has T_max=MAX_EPOCH (warmup counted separately).
        progress = (epoch - cfg.WARMUP_EPOCH) / cfg.MAX_EPOCH
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
