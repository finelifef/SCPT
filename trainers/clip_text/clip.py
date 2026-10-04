"""Minimal CLIP checkpoint loader and tokenizer (derived from OpenAI CLIP)."""

import hashlib
import os
import urllib.request
from pathlib import Path

import torch

from .model import build_model
from .simple_tokenizer import SimpleTokenizer

MODEL_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
)
_tokenizer = SimpleTokenizer()


def tokenize(texts, context_length=77):
    """Tokenize without silently truncating class descriptions or SRE codes."""
    if isinstance(texts, str):
        texts = [texts]
    result = torch.zeros(len(texts), context_length, dtype=torch.long)
    for i, text in enumerate(texts):
        tokens = [_tokenizer.encoder["<|startoftext|>"]]
        tokens += _tokenizer.encode(text)
        tokens += [_tokenizer.encoder["<|endoftext|>"]]
        if len(tokens) > context_length:
            raise ValueError(f"Prompt exceeds CLIP's {context_length}-token limit: {text}")
        result[i, : len(tokens)] = torch.tensor(tokens)
    return result


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_clip(checkpoint="", cache_dir="~/.cache/clip"):
    """Load the official ViT-B/16 checkpoint, verifying its SHA-256 first."""
    expected = MODEL_URL.split("/")[-2]
    path = (
        Path(checkpoint).expanduser()
        if checkpoint
        else Path(cache_dir).expanduser() / "ViT-B-16.pt"
    )
    if not path.exists():
        if checkpoint:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".download")
        print(f"Downloading official CLIP weights to {path}")
        urllib.request.urlretrieve(MODEL_URL, temporary)
        if sha256(temporary) != expected:
            raise RuntimeError(f"CLIP checksum mismatch: {temporary}")
        os.replace(temporary, path)
    if sha256(path) != expected:
        raise RuntimeError(f"Not the official ViT-B/16 checkpoint (SHA-256 mismatch): {path}")
    archive = torch.jit.load(str(path), map_location="cpu").eval()
    return build_model(archive.state_dict())
