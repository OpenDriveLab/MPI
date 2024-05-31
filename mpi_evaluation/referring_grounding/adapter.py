"""
adapter.py

OCID-Ref Referring Expression Grounding Adapter definition -- as a LightningModule; takes as input both the backbone,
as well an nn.Module representing the randomly initialized representation extractor (e.g., a MAP extractor) that returns
a single (fused) vector representation.

For now, the structure of this adapter is general to any fused single-vector extraction method; can be overridden!
"""
from typing import List, Tuple

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import AdamW, Optimizer
from torchvision.ops import box_convert, box_iou

import transformers

class DetectorMLP(LightningModule):
    def __init__(
        self, backbone: nn.Module, extractor: nn.Module, flag_dual: bool = True, mlp_features: Tuple[int, ...] = (512, 256, 128, 64), 
        iou_threshold = 0.25, lr = 1e-3, language_model_path: str = None,
    ) -> None:
        super().__init__()
        # mlp_features = (128, 128)
        self.mlp_features = [extractor.embed_dim, *list(mlp_features)]

        # Create Network --> Extractor into a "single-shot" detection MLP
        self.backbone, self.extractor, _layers = backbone, extractor, []
        for in_feats, out_feats in zip(self.mlp_features[:-1], self.mlp_features[1:]):
            _layers.append(nn.Linear(in_feats, out_feats))
            _layers.append(nn.GELU())

        # Add final projection =>> xywh bbox coordinates
        _layers.append(nn.Linear(self.mlp_features[-1], 4))
        self.mlp = nn.Sequential(*_layers)

        # MPI takes two input frames
        self.flag_dual = True # False
        self.iou_threshold = iou_threshold
        self.lr = lr

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(language_model_path)
        self.lm = transformers.AutoModel.from_pretrained(language_model_path)
        self.lm.eval()


    def forward(self, img: torch.Tensor, language: Tuple[str]) -> torch.Tensor:
        
        if self.flag_dual:
            img = torch.stack((img, img), dim=1)

        mask = None
        with torch.no_grad():
            patches = self.backbone.get_representations(img, language)

        extracted = self.extractor(patches)
        return self.mlp(extracted)

    def training_step(self, batch: Tuple[torch.Tensor, str, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Unpack batch of RGB frame, language string, bbox (xywh), clutter split, run detector, compute loss."""
        img, lang, bbox, _ = batch

        # Run forward pass, get predicted bbox coordinates =>> [bsz, 4]
        bbox_coords = self.forward(img, lang)

        # Bounding box normalization to stabilize training
        bbox = bbox.float() / 224

        # Compute huber loss (smooth L1) relative to ground-truth bbox coordinates
        return F.l1_loss(bbox_coords, bbox.float())

    def validation_step(
        self, batch: Tuple[torch.Tensor, str, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, ...]:
        """Unpack batch =>> compute Acc @ 0.25 IoU (total, per-split) as the key evaluation metric."""
        img, lang, bbox, clutter_split = batch

        # Run forward pass, get predicted bbox coordinates =>> [bsz, 4]
        bbox_coords = self.forward(img, lang)
        bbox_coords *= 224

        # Compute huber loss (smooth L1) relative to ground-truth bbox coordinates
        loss = F.l1_loss(bbox_coords, bbox.float())

        # Compute various Acc @ 0.25 IoU Metrics --> for *all data* and for each individual split...
        #   => Convert from xywh -> xyxy then use torchvision.ops.box_iou().diagonal() to compute IoU per example
        #   => Threshold based on IoU of 0.25 --> use to compute total accuracy...
        iou = box_iou(box_convert(bbox_coords, "xywh", "xyxy"), box_convert(bbox, "xywh", "xyxy")).diagonal()
        iou_at_25 = iou > self.iou_threshold

        # Total Acc @ 0.25
        total_acc25, n_total = iou_at_25.float().sum(), (clutter_split != -1).float().sum()

        # Compute Acc @ 0.25 for each of the three splits...
        free_mask = clutter_split == 0
        free_acc25, n_free = (iou_at_25 & free_mask).float().sum(), free_mask.sum()

        touching_mask = clutter_split == 1
        touching_acc25, n_touching = (iou_at_25 & touching_mask).float().sum(), touching_mask.sum()

        stacked_mask = clutter_split == 2
        stacked_acc25, n_stacked = (iou_at_25 & stacked_mask).float().sum(), stacked_mask.sum()

        return loss, total_acc25, n_total, free_acc25, n_free, touching_acc25, n_touching, stacked_acc25, n_stacked

    def validation_epoch_end(self, step_outputs: List[Tuple[torch.Tensor, ...]]) -> None:
        """Aggregate and log validation metrics."""
        val_loss, total_acc25, n_total, free_acc25, n_free, touching_acc25, n_touching, stacked_acc25, n_stacked = [
            torch.stack(output) for output in zip(*step_outputs)
        ]

        # Reduce & Log...
        self.log_dict(
            {
                "val_loss": val_loss.mean(),
                "total_acc25": total_acc25.sum() / n_total.sum(),
                "free_acc25": free_acc25.sum() / n_free.sum(),
                "touching_acc25": touching_acc25.sum() / n_touching.sum(),
                "stacked_acc25": stacked_acc25.sum() / n_stacked.sum(),
            },
            prog_bar=True,
        )

    def test_step(
        self, batch: Tuple[torch.Tensor, str, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, ...]:
        """Unpack batch =>> compute Acc @ 0.25 IoU (total, per-split) as the key evaluation metric."""
        img, lang, bbox, clutter_split = batch

        # Run forward pass, get predicted bbox coordinates =>> [bsz, 4]
        bbox_coords = self.forward(img, lang)
        bbox_coords *= 224
        # Compute huber loss (smooth L1) relative to ground-truth bbox coordinates
        loss = F.huber_loss(bbox_coords, bbox.float())

        # Compute various Acc @ 0.25 IoU Metrics --> for *all data* and for each individual split...
        #   => Convert from xywh -> xyxy then use torchvision.ops.box_iou().diagonal() to compute IoU per example
        #   => Threshold based on IoU of 0.25 --> use to compute total accuracy...
        iou = box_iou(box_convert(bbox_coords, "xywh", "xyxy"), box_convert(bbox, "xywh", "xyxy")).diagonal()
        iou_at_25 = iou > self.iou_threshold

        # Total Acc @ 0.25
        total_acc25, n_total = iou_at_25.float().sum(), (clutter_split != -1).float().sum()

        # Compute Acc @ 0.25 for each of the three splits...
        free_mask = clutter_split == 0
        free_acc25, n_free = (iou_at_25 & free_mask).float().sum(), free_mask.sum()

        touching_mask = clutter_split == 1
        touching_acc25, n_touching = (iou_at_25 & touching_mask).float().sum(), touching_mask.sum()

        stacked_mask = clutter_split == 2
        stacked_acc25, n_stacked = (iou_at_25 & stacked_mask).float().sum(), stacked_mask.sum()

        return loss, total_acc25, n_total, free_acc25, n_free, touching_acc25, n_touching, stacked_acc25, n_stacked

    def test_epoch_end(self, step_outputs: List[Tuple[torch.Tensor, ...]]) -> None:
        """Aggregate and log test metrics."""
        test_loss, total_acc25, n_total, free_acc25, n_free, touching_acc25, n_touching, stacked_acc25, n_stacked = [
            torch.stack(output) for output in zip(*step_outputs)
        ]

        # Reduce & Log...
        self.log_dict(
            {
                "test_loss": test_loss.mean(),
                "test_total_acc25": total_acc25.sum() / n_total.sum(),
                "test_free_acc25": free_acc25.sum() / n_free.sum(),
                "test_touching_acc25": touching_acc25.sum() / n_touching.sum(),
                "test_stacked_acc25": stacked_acc25.sum() / n_stacked.sum(),
            },
            prog_bar=True,
        )

    def configure_optimizers(self) -> Optimizer:
        return AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr)
        #return AdamW([p for p in self.parameters() if p.requires_grad], lr=1e-4)


def instantiate_detector(backbone: nn.Module, extractor: nn.Module, flag_dual: bool = True, iou_threshold:float = 0.25, lr: float = 1e-3, load_checkpoint = None, language_model_path=None) -> LightningModule:
    model = DetectorMLP(backbone, extractor, flag_dual, iou_threshold = iou_threshold, lr=lr, language_model_path = language_model_path)
    if load_checkpoint is not None and os.path.exists(load_checkpoint):
        state_dict = torch.load(load_checkpoint)["state_dict"]
        new_state_dict = {}
        for name, param in state_dict.items():
            if 'motion_embedding' in name: # name.replace('motion_embedding', '')
                print(name)
                new_state_dict[name.replace('motion_', 'reconstruction_')] = param
            else:
                new_state_dict[name] = param
        model.load_state_dict(new_state_dict, strict=False)
        print(f"load checkpoint from {load_checkpoint}")
    return model
