"""
datasets.py

Base Hydra Structured Config for defining various pretraining datasets and appropriate configurations. Uses a simple,
single inheritance structure.
"""
from dataclasses import dataclass
from typing import Any, Tuple

from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import MISSING


@dataclass
class DatasetConfig:
    name: str = MISSING
    path: str = MISSING
    artifact_path: str = MISSING

    # Dataset-Specific Parameters
    resolution: int = 224
    normalization: Tuple[Any, Any] = MISSING

    # For preprocessing --> maximum size of saved frames (assumed square)
    preprocess_resolution: int = MISSING


    # Language Modeling Parameters
    language_model: str = "distilbert-base-uncased"
    hf_cache: str = to_absolute_path("data/hf-cache")

    # Maximum Length for truncating language inputs... should be computed after the fact (set to -1 to compute!)
    max_lang_len: int = MISSING

    # Dataset sets the number of pretraining epochs (general rule :: warmup should be ~5% of full)
    warmup_epochs: int = MISSING
    max_epochs: int = MISSING



@dataclass
class Ego4DHOIConfig(DatasetConfig):
    # fmt: off
    name: str = "ego4d-hoi"

    # Dataset Specific arguments
    normalization: Tuple[Any, Any] = (                              # Mean & Standard Deviation (default :: ImageNet)
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    )

    # Sth-Sth-v2 Videos have a fixed height of 240; we'll crop to square at this resolution!
    preprocess_resolution: int = 224

    # Validation Parameters
    n_val_videos: int = 1000                                        # Number of Validation Clips (fast evaluation!)

    # Epochs for Dataset
    warmup_epochs: int = 10
    max_epochs: int = 200 # 400

    # Language Modeling Parameters
    max_lang_len: int = 20
    # fmt: on


# Create a configuration group `dataset` and populate with the above...
#   =>> Note :: this is meant to be extendable --> add arbitrary datasets & mixtures!
cs = ConfigStore.instance()
cs.store(group="dataset", name="ego4d-hoi", node=Ego4DHOIConfig)
