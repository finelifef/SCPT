"""CPU unit tests; no datasets or downloaded model weights are required."""

import copy
import io
import json
import random
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from datasets.finegrained import few_shot, read_manifest, validate_class_count
from datasets.prepare import split_class_folders
from train import setup_cfg
from trainers.clip_text.model import CLIP
from trainers.optim import build_optimization
from trainers.scpt import CustomCLIP
from trainers.semantics import SemanticCondensation, relation_codes, sample_rank


def config():
    return setup_cfg(
        Namespace(
            root=".",
            dataset="oxford_pets",
            shots=16,
            seed=0,
            output_dir="output/test",
            split_dir="splits",
            clip_checkpoint="",
            cache_dir="~/.cache/clip",
            config_file=str(Path(__file__).resolve().parents[1] / "configs/scpt.yaml"),
            opts=["TRAINER.SCPT.PREC", "fp32", "USE_CUDA", "False"],
        )
    )


def tiny_backbone():
    model = CLIP(
        embed_dim=32,
        image_resolution=32,
        vision_layers=1,
        vision_width=64,
        vision_patch_size=16,
        context_length=77,
        vocab_size=49408,
        transformer_width=64,
        transformer_heads=1,
        transformer_layers=9,
    )
    return model.float().eval().requires_grad_(False)


class SemanticsTests(unittest.TestCase):
    def test_relation_codes_reproducible(self):
        x = torch.eye(5)
        torch.manual_seed(7)
        first = relation_codes(x)
        torch.manual_seed(7)
        self.assertEqual(first, relation_codes(x))
        self.assertEqual([len(c) for c in first], [6] * 5)

    def test_rank_threshold_edges(self):
        self.assertEqual(sample_rank([3.0, 2.0, 1.0], 0.3), 3)
        self.assertEqual(sample_rank([0.3, 0.3], 0.3), 2)
        self.assertEqual(sample_rank([0.001], 100), 1)

    def test_rank_matches_source_normal_case(self):
        values = np.array([2.0, 1.0, 0.4, 0.29, 0.2, 0.1])
        for seed in range(20):
            rng = np.random.RandomState(seed)
            remaining = values[3:]
            keep = rng.rand(len(remaining)) < np.exp(
                -15 * (0.3 - remaining) ** 2 - 0.05 * len(remaining)
            )
            expected = max(1, 3 + keep.sum())
            self.assertEqual(sample_rank(values, 0.3, np.random.RandomState(seed)), expected)

    def test_condensation_full_rank(self):
        x = torch.randn(4, 8)
        result = SemanticCondensation(x, 0)(torch.float32)
        torch.testing.assert_close(result, x / x.norm(dim=-1, keepdim=True), atol=1e-6, rtol=1e-5)


class DatasetTests(unittest.TestCase):
    def test_portable_sampling(self):
        rows = [[f"{label}/{i}.jpg", label, str(label)] for label in range(3) for i in range(6)]
        a = few_shot(rows, 2, random.Random(1))
        b = few_shot(rows, 2, random.Random(1))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 6)
        with self.assertRaises(ValueError):
            few_shot(rows, 7, random.Random(1))

    def test_reject_overlap_and_absolute_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            for name in ("same.jpg", "/absolute.jpg", "../outside.jpg", "C:\\image.jpg"):
                data = {split: [[name, 0, "cat"]] for split in ("train", "val", "test")}
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    read_manifest(path)

    def test_food172_class_count(self):
        data = {"train": [[f"{i}.jpg", i, str(i)] for i in range(172)]}
        validate_class_count(data, "food172")
        data["train"].pop()
        with self.assertRaises(ValueError):
            validate_class_count(data, "food172")

    def test_class_folder_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("cat", "dog"):
                (root / name).mkdir()
                for index in range(20):
                    (root / name / f"{index}.jpg").touch()
            splits = split_class_folders(root, seed=0)
            self.assertEqual(splits, split_class_folders(root, seed=0))
            self.assertEqual([len(splits[s]) for s in ("train", "val", "test")], [20, 8, 12])
            path = root / "split.json"
            path.write_text(json.dumps(splits))
            self.assertEqual(read_manifest(path), splits)


class ModelTests(unittest.TestCase):
    def test_optimizer_and_default_warmup(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer, scheduler = build_optimization([parameter], config().OPTIM)
        self.assertEqual(optimizer.defaults["eps"], 0.001)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.00001)
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.002)
        optimizer.step()
        scheduler.step()
        self.assertLess(optimizer.param_groups[0]["lr"], 0.002)
        self.assertGreater(optimizer.param_groups[0]["lr"], 0.0019)

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_seed_zero_and_scpt_only_config(self):
        cfg = config()
        self.assertEqual(cfg.SEED, 0)
        self.assertEqual(cfg.TRAINER.NAME, "SCPT")
        self.assertNotIn("COOP", cfg.TRAINER)

    def test_gradients_and_checkpoint_roundtrip(self):
        torch.manual_seed(1)
        backbone = tiny_backbone()
        reference = copy.deepcopy(backbone)
        model = CustomCLIP(config(), ["cat", "dog", "bird"], backbone)
        images = torch.randn(2, 3, 32, 32)
        labels = torch.tensor([0, 1])
        optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=0.001)
        logits, loss = model(images, labels)
        self.assertEqual(logits.shape, (2, 3))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.prompt_learner.ctx.grad)
        self.assertIsNotNone(model.prompt_learner.meta_net.linear1.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        optimizer.step()
        expected = model(images).detach()
        data = io.BytesIO()
        torch.save({"state_dict": model.prompt_learner.state_dict()}, data)
        data.seek(0)
        state = torch.load(data, weights_only=True)
        torch.manual_seed(99)
        restored = CustomCLIP(config(), ["cat", "dog", "bird"], reference)
        restored.prompt_learner.load_state_dict(state["state_dict"], strict=True)
        restored.rebuild_condensation()
        with mock.patch(
            "trainers.semantics.sample_rank", side_effect=AssertionError("SVD during inference")
        ):
            torch.testing.assert_close(restored(images), expected, atol=0, rtol=0)
        wrong = CustomCLIP(config(), ["dog", "cat", "bird"], tiny_backbone())
        with self.assertRaises(ValueError):
            wrong.prompt_learner.load_state_dict(state["state_dict"])


if __name__ == "__main__":
    unittest.main()
