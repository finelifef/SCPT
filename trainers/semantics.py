"""Semantic relation encoding and the source implementation's ScLoss target."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def relation_codes(features, extra_bits=3):
    """Encode pairwise cosine relations as signed Gaussian-projection bits.

    Sampling on CPU in fp16 follows the original implementation. The global
    torch seed controls the projection; the resulting token buffers are saved
    with the prompt learner so evaluation never relies on re-sampling them.
    """
    count = len(features)
    if count < 2:
        raise ValueError("SCPT requires at least two classes")
    bits = math.ceil(math.log2(count)) + extra_bits
    similarities = F.cosine_similarity(features[:, None], features[None, :], dim=-1)
    planes = torch.randn(bits, count, dtype=torch.float16).to(features)
    codes = (similarities @ planes.T > 0).int()
    return ["".join(map(str, row)) for row in codes.cpu().tolist()]


def sample_rank(singular_values, threshold, rng=None):
    """Original count-based rank rule, with the all-above-threshold edge fixed.

    Above threshold: always keep. Below: exp(-15*(tau-s)^2 - .05*n_below).
    Reconstruct the first K components, where K counts selected components.
    """
    values = np.asarray(singular_values)
    below = values < threshold
    probabilities = np.ones_like(values)
    probabilities[below] = np.exp(-15.0 * (threshold - values[below]) ** 2)
    probabilities[below] *= np.exp(-0.05 * below.sum())
    # Draw only below threshold, preserving the original normal-case RNG usage.
    generator = np.random if rng is None else rng
    selected = generator.random(int(below.sum())) < probabilities[below]
    return max(1, int((~below).sum() + selected.sum()))


class SemanticCondensation(nn.Module):
    """Cache the fixed CPU SVD; resample its truncation rank every training step."""

    def __init__(self, features, threshold):
        super().__init__()
        normalized = features / features.norm(dim=-1, keepdim=True)
        u, s, vh = torch.linalg.svd(normalized.detach().cpu().float(), full_matrices=False)
        # These are derived from the checkpointed teacher, not trainable weights.
        self.register_buffer("u", u, persistent=False)
        self.register_buffer("s", s, persistent=False)
        self.register_buffer("vh", vh, persistent=False)
        self.values = s.numpy().copy()
        self.threshold = threshold

    def forward(self, dtype):
        rank = sample_rank(self.values, self.threshold)
        return ((self.u[:, :rank] * self.s[:rank]) @ self.vh[:rank]).to(dtype=dtype)
