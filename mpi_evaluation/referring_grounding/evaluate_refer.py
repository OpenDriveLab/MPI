"""
evaluate_refer.py

Example script for loading a pretrained V-Cond model (from the `mpi` library), configuring a MAP-based extractor
factory function, and then defining/invoking the ReferDetectionHarness.
"""
import os
import hydra
import torch
import torch.nn as nn
from mpi.models.materialize import load
from mpi.models.util import instantiate_extractor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mpi_evaluation.referring_grounding.harness import ReferDetectionHarness
from omegaconf import MISSING, OmegaConf
from mpi.configs import AcceleratorConfig, DatasetConfig, ModelConfig, TrackingConfig
from hydra.core.config_store import ConfigStore
from mpi.tools.overwatch import OverwatchRich

from einops import rearrange, repeat

import transformers

DEFAULTS = [
    "_self_",
    {"model": "mpi-small"}, # "mpi-small", "mpi-base"
    {"dataset": "ego4d-hoi"},
    {"accelerator": "torchrun"},
    {"tracking": "mpi-tracking"},
    {"override hydra/job_logging": "overwatch_rich"},
]


@dataclass
class PretrainConfig:
    # fmt: off
    defaults: List[Any] = field(default_factory=lambda: DEFAULTS)
    hydra: Dict[str, Any] = field(default_factory=lambda: {
        "run": {"dir": "runs/eval"}
    })

    # experiment setting
    test_only:bool = MISSING
    iou_threshold:float = MISSING
    lr:float = MISSING
    load_checkpoint:str = MISSING

    # Command Line Argu ments
    run_id: Optional[str] = None                                        # Run ID for Logging
    seed: int = 21                                                      # Random Seed (for reproducibility)

    # Resume / Debug Behavior
    resume: bool = False                                                # Whether to resume an existing run...
    wandb_resume_id: Optional[str] = None                               # W&B Run ID for `resume` behavior...

    # Composable / Structured Arguments
    model: ModelConfig = MISSING                                        # Model architecture for pretraining
    dataset: DatasetConfig = MISSING                                    # List of datasets for pretraining
    accelerator: AcceleratorConfig = MISSING                            # Accelerator (should always keep `torchrun`)
    tracking: TrackingConfig = MISSING                                  # Run/experiment tracking configuration
    # fmt: on
    eval_checkpoint_path: str = MISSING
    save_path: str = MISSING

    # DistillBERT path
    language_model_path: str = MISSING

    # Combination of loss functions
    add_rec_loss: bool = True
    add_box_loss: bool = True
    add_contra_loss: bool = True
    

# Hydra Setup :: Retrieve ConfigStore (Singleton) & Register Components
cs = ConfigStore.instance()
cs.store(group="hydra/job_logging", name="overwatch_rich", node=OverwatchRich)
cs.store(name="config", node=PretrainConfig)


# Reproduction of R3M with official checkpoints
class R3M(nn.Module):
    def __init__(self):
        super().__init__()
        
        from r3m import load_r3m
        self.r3m = load_r3m("resnet50") # resnet18, resnet34
        self.r3m.eval()

        self.embed_dim = 768
        self.n_heads = 8

        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
        self.lm = transformers.AutoModel.from_pretrained('distilbert/distilbert-base-uncased')
        self.lm.eval()

        self.proj = nn.Linear(2048, 768)

    def forward(self, imgs, language):
        # Tokenize Language --> note max length is 20!
        if language is None:
            lang, lang_mask = [torch.zeros(imgs.shape[0], 20, dtype=int, device=self.lm.device) for _ in range(2)]
            lang[:, 0], lang_mask[:, 0] = self.tokenizer.cls_token_id, 1
        else:
            tokens = self.tokenizer(language, return_tensors="pt", max_length=20, padding="max_length", truncation=True)
            lang, lang_mask = tokens["input_ids"].to(self.lm.device), tokens["attention_mask"].to(self.lm.device)

            # Tile Language & Language Mask if mismatch with # images!
            if not len(lang) == len(imgs):
                lang = repeat(lang, "b seq -> (bsz b) seq", bsz=imgs.size(0))
                lang_mask = repeat(lang_mask, "b seq -> (bsz b) seq", bsz=imgs.size(0))

        with torch.no_grad():
            transformer_embeddings = self.lm(lang, attention_mask=lang_mask).last_hidden_state

        lang = transformer_embeddings
        img_embed = self.r3m(imgs)
        img_embed = rearrange(img_embed, "bsz (seq d) -> bsz seq d", d=2048)
        img_embed = self.proj(img_embed)

        # Return concatenated multimodal tokens
        return torch.cat([lang, img_embed], dim=1)
        # return torch.cat([lang, img_embed.unsqueeze(1)], dim=1)
    
    def get_representations(self, imgs, language):
        return self.forward(imgs, language)
        
import torchvision.transforms as T



# Reproduction of MVP with official checkpoints
NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
class MVP(nn.Module):
    def __init__(self):
        super().__init__()
        
        import mvp
        self.r3m = mvp.load("vitb-mae-egosoup")
        self.r3m.eval()

        self.embed_dim = 768
        self.n_heads = 8

        self.tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
        self.lm = transformers.AutoModel.from_pretrained('distilbert/distilbert-base-uncased')
        self.lm.eval()

        self.lang2embed = nn.Linear(768, self.embed_dim)

    def forward(self, imgs, language):
        # Tokenize Language --> note max length is 20!
        if language is None:
            lang, lang_mask = [torch.zeros(imgs.shape[0], 20, dtype=int, device=self.lm.device) for _ in range(2)]
            lang[:, 0], lang_mask[:, 0] = self.tokenizer.cls_token_id, 1
        else:
            tokens = self.tokenizer(language, return_tensors="pt", max_length=20, padding="max_length", truncation=True)
            lang, lang_mask = tokens["input_ids"].to(self.lm.device), tokens["attention_mask"].to(self.lm.device)

            # Tile Language & Language Mask if mismatch with # images!
            if not len(lang) == len(imgs):
                lang = repeat(lang, "b seq -> (bsz b) seq", bsz=imgs.size(0))
                lang_mask = repeat(lang_mask, "b seq -> (bsz b) seq", bsz=imgs.size(0))

        with torch.no_grad():
            transformer_embeddings = self.lm(lang, attention_mask=lang_mask).last_hidden_state

        lang = self.lang2embed(transformer_embeddings)
        img_embed = self.r3m.get_representations(imgs) # [B * (patches + 1) * D]

        # Return concatenated multimodal tokens
        return torch.cat([lang, img_embed], dim=1)
    
    def get_representations(self, imgs, language):
        return self.forward(imgs, language)



@hydra.main(config_path=None, config_name="config")
def evaluate_refer(cfg: PretrainConfig) -> None:
    # Load Backbone
    print(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if 'mpi' in cfg.model.identifier: # cfg.model.identifier is not used in func <load>
        checkpoint_path = cfg.eval_checkpoint_path
        backbone, preprocess = load(cfg.model.identifier, device=device, config = cfg, checkpoint_path = checkpoint_path)
        flag_dual = False

    elif 'r3m' in cfg.model.identifier:
        backbone, preprocess = R3M(), T.Compose(
        [
            T.Resize(224, antialias=True),
        ])
        flag_dual = False

    elif 'mvp' in cfg.model.identifier:
        backbone, preprocess = MVP(), T.Compose(
        [
            T.Resize(224, antialias=True),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=NORMALIZATION[0], std=NORMALIZATION[1]),
        ])
        flag_dual = False



    # Create MAP Extractor Factory (single latent =>> we only predict of a single dense vector representation)
    map_extractor_fn = instantiate_extractor(backbone, n_latents=1, mode_extractor='MAP')

    # Create Refer Detection Harness
    refer_evaluator = ReferDetectionHarness(cfg.save_path, backbone, preprocess, map_extractor_fn, 
                                                lr = cfg.lr, load_checkpoint=cfg.load_checkpoint, language_model_path=cfg.language_model_path)

    refer_evaluator.fit(test_only = cfg.test_only, iou_threshold = cfg.iou_threshold)
    refer_evaluator.test()


if __name__ == "__main__":
    evaluate_refer()
