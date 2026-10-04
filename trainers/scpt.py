"""SCPT only: SRE prompts, TCP-derived class-aware tokens, and ScLoss."""

from collections import OrderedDict
from pathlib import Path

import torch
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from torch import nn
from torch.nn import functional as F

from datasets import DATASETS

from .clip_text.clip import load_clip, tokenize
from .clip_text.model import QuickGELU
from .optim import build_optimization
from .semantics import SemanticCondensation, relation_codes

# Keep the source templates verbatim, including the documented StanfordDogs
# "do" suffix, to avoid silently changing experimental semantics.
TEMPLATES = {
    "OxfordPets": ("a photo of a {}, a type of pet.", "{}, a type of pet."),
    "OxfordFlowers": ("a photo of a {}, a type of flower.", "{}, a type of flower."),
    "FGVCAircraft": ("a photo of an aircraft {}.", "{}, a type of aircraft."),
    "StanfordCars": ("a photo of a {}.", "{}, a type of car"),
    "WebCar": ("a photo of a {}.", "{}, a type of car"),
    "StanfordDogs": ("a photo of a {}, a type of dog.", "{}, a type of do."),
    "DogBreed": ("a photo of a {}, a type of dog.", "{}, a type of dog."),
    "Cub200": ("a photo of a {}, a type of bird.", "{}, a type of bird."),
    "Fruit92": ("a photo of a {}, a type of fruit.", "{}, a type of fruit."),
    "Veg200": ("a photo of a {}, a type of vegetable.", "{}, a type of vegetable."),
    **{
        name: ("a photo of a {}, a type of food.", "{}, a type of food.")
        for name in ("Food101", "Food172", "Food200", "Food500")
    },
}


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, backbone):
        super().__init__()
        settings = cfg.TRAINER.SCPT
        width = backbone.ln_final.weight.shape[0]
        dtype = backbone.dtype
        device = backbone.positional_embedding.device
        self.n_ctx = settings.N_CTX
        self.width = width
        self.classnames = tuple(classnames)
        self.ctx = nn.Parameter(torch.empty(self.n_ctx, width, dtype=dtype, device=device))
        # Source code initializes on CPU; do so here before moving to the device.
        initial = torch.empty(self.n_ctx, width, dtype=dtype)
        nn.init.normal_(initial, std=0.02)
        with torch.no_grad():
            self.ctx.copy_(initial)
        teacher_template, suffix_template = TEMPLATES[DATASETS[cfg.DATASET.KEY][0]]
        tokens = tokenize([teacher_template.format(c.replace("_", " ")) for c in classnames]).to(
            device
        )
        with torch.no_grad():
            features = backbone.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        self.register_buffer("teacher_features", features)
        codes = relation_codes(features, settings.EXTRA_BITS)
        prefix = " ".join(["X"] * self.n_ctx)
        prompts = [f"{prefix} {suffix_template.format(code)}" for code in codes]
        prompt_tokens = tokenize(prompts).to(device)
        with torch.no_grad():
            embeddings = backbone.token_embedding(prompt_tokens).to(dtype)
        self.register_buffer("token_prefix", embeddings[:, :1])
        self.register_buffer("token_suffix", embeddings[:, 1 + self.n_ctx :])
        self.register_buffer("tokenized_prompts", prompt_tokens)
        output_dim = backbone.visual.output_dim
        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(output_dim, output_dim // 4)),
                    ("relu", QuickGELU()),
                    ("linear2", nn.Linear(output_dim // 4, 4 * width)),
                ]
            )
        ).to(device=device, dtype=dtype)
        print(
            f"SRE: {len(classnames)} classes, {len(codes[0])} bits, {len(set(codes))} unique codes"
        )

    def get_extra_state(self):
        return {"classnames": self.classnames}

    def set_extra_state(self, state):
        if tuple(state["classnames"]) != self.classnames:
            raise ValueError("Checkpoint class names/order do not match this dataset")

    def forward(self):
        context = self.ctx.unsqueeze(0).expand(len(self.classnames), -1, -1)
        prompts = torch.cat([self.token_prefix, context, self.token_suffix], dim=1)
        class_tokens = self.meta_net(self.teacher_features).reshape(-1, 4, self.width)
        return prompts, class_tokens


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, backbone):
        super().__init__()
        self.backbone = backbone
        self.prompt_learner = PromptLearner(cfg, classnames, backbone)
        self.loss_weight = cfg.TRAINER.SCPT.LOSS_WEIGHT
        self.threshold = cfg.TRAINER.SCPT.TAU
        self.rebuild_condensation()

    def rebuild_condensation(self):
        teacher = self.prompt_learner.teacher_features
        self.condensation = SemanticCondensation(teacher, self.threshold).to(teacher.device)

    def forward(self, images, labels=None):
        backbone = self.backbone
        prompts, class_tokens = self.prompt_learner()
        x = (prompts + backbone.positional_embedding.to(backbone.dtype)).permute(1, 0, 2)
        x = backbone.transformer.resblocks([x, class_tokens, 1.0, 0])[0]
        x = backbone.ln_final(x.permute(1, 0, 2)).to(backbone.dtype)
        eot = self.prompt_learner.tokenized_prompts.argmax(dim=-1)
        text = x[torch.arange(len(x), device=x.device), eot] @ backbone.text_projection
        text = text / text.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            visual = backbone.encode_image(images.to(backbone.dtype))
            visual = visual / visual.norm(dim=-1, keepdim=True)
        logits = backbone.logit_scale.exp() * visual @ text.T
        if labels is None:
            return logits
        target = self.condensation(text.dtype)
        alignment = 1.0 - F.cosine_similarity(text, target, dim=1, eps=1e-7).mean()
        loss = F.cross_entropy(logits, labels) + self.loss_weight * alignment
        return logits, loss


@TRAINER_REGISTRY.register()
class SCPT(TrainerX):
    def check_cfg(self, cfg):
        if cfg.TRAINER.SCPT.PREC not in ("fp16", "fp32", "amp"):
            raise ValueError("PREC must be fp16, fp32, or amp")
        if cfg.MODEL.BACKBONE.NAME != "ViT-B/16":
            raise ValueError("This release supports the paper's ViT-B/16 backbone only")
        if cfg.TRAINER.SCPT.N_CTX != 4 or tuple(cfg.INPUT.SIZE) != (224, 224):
            raise ValueError("SCPT release uses four context tokens and 224x224 inputs")

    def build_model(self):
        cfg = self.cfg
        self.best_result = float("-inf")
        precision = cfg.TRAINER.SCPT.PREC
        if self.device.type == "cpu" and precision != "fp32":
            raise ValueError("Use TRAINER.SCPT.PREC fp32 with USE_CUDA False on CPU")
        backbone = load_clip(cfg.MODEL.CLIP_CHECKPOINT, cfg.MODEL.CACHE_DIR)
        if precision in ("fp32", "amp"):
            backbone.float()
        backbone.requires_grad_(False)
        backbone.to(self.device).eval()
        self.model = CustomCLIP(cfg, self.dm.dataset.classnames, backbone).to(self.device)
        learner = self.model.prompt_learner
        print("Trainable parameters:", sum(p.numel() for p in learner.parameters()))
        self.optim, self.sched = build_optimization(learner.parameters(), cfg.OPTIM)
        self.register_model("prompt_learner", learner, self.optim, self.sched)
        self.scaler = torch.amp.GradScaler("cuda", enabled=precision == "amp")

    def forward_backward(self, batch):
        images = batch["img"].to(self.device)
        labels = batch["label"].to(self.device)
        self.optim.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self.device.type, enabled=self.cfg.TRAINER.SCPT.PREC == "amp"
        ):
            logits, loss = self.model(images, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss; try TRAINER.SCPT.PREC amp")
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optim)
        self.scaler.update()
        if self.batch_idx + 1 == self.num_batches:
            self.update_lr()
        return {"loss": loss.item(), "acc": compute_accuracy(logits, labels)[0].item()}

    def load_model(self, directory, epoch=None):
        filename = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"
        path = Path(directory) / "prompt_learner" / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.prompt_learner.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.rebuild_condensation()
        print(f"Loaded {path} (epoch {checkpoint['epoch']})")

    def resume_model_if_exist(self, directory):
        # Do not inherit Dassl's automatic, non-reproducible resume with missing
        # RNG state. Fresh runs and explicit eval are the only release workflows.
        checkpoint_dir = Path(directory) / "prompt_learner"
        if checkpoint_dir.exists() and any(checkpoint_dir.glob("*.pth.tar*")):
            raise FileExistsError(
                "Existing checkpoints: choose a new --output-dir or use --eval-only"
            )
        return 0
