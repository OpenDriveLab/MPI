"""
harness.py

Class defining the evaluation harness for the OCID-Ref Referring Expression Grounding task; a ReferDetectionHarness is
comprised of the following three parts:
    1) __init__  :: Takes backbone, factory function for extractor, factory function for adapter (as LightningModule)
    2) fit       :: Invokes train/fit protocol; for the detection task, this is a traditional supervised learning flow.
                    Uses a Trainer on top of the defined LightningModule --> simple calls to Trainer.fit().
    3) test      :: Function defining the testing (or test metric aggregation) procedure.

By default, assumes a simple MLP bounding-box predictor atop a single (fused) representation; override as you see fit!
"""
import json
import logging
import os
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from mpi_evaluation.referring_grounding.adapter import instantiate_detector
from mpi_evaluation.referring_grounding.preprocessing import build_datamodule
from mpi_evaluation.referring_grounding.util import LOG_CONFIG, set_global_seed

# Grab Logger
logging.config.dictConfig(LOG_CONFIG)
overwatch = logging.getLogger(__file__)


class ReferDetectionHarness:
    def __init__(
        self,
        model_id: str,
        backbone: nn.Module,
        preprocess: Callable[[torch.Tensor], torch.Tensor],
        extractor_init_fn: Callable[[], nn.Module],
        detector_init_fn: Callable[[nn.Module, nn.Module], LightningModule] = instantiate_detector,
        run_dir: Path = Path("runs/evaluation/langref"),
        data: str = "data/langref",
        bsz: int = 256,
        epochs: int = 10,
        seed: int = 6,
        lr: float = 1e-3,
        load_checkpoint: str = None,
        language_model_path: str = None,
    ) -> None:
        overwatch.info("Initializing ReferDetectionHarness")
        self.model_id, self.backbone, self.preprocess = model_id, backbone, preprocess
        self.extractor_init_fn, self.detector_init_fn = extractor_init_fn, detector_init_fn
        self.language_model_path = language_model_path
        data = Path(os.path.join(os.path.split(os.path.realpath(__file__))[0], data))
        self.run_dir, self.data, self.bsz, self.epochs, self.seed = run_dir, data, bsz, epochs, seed
        self.load_checkpoint = load_checkpoint
        self.lr = lr
        # Set Randomness
        set_global_seed(self.seed)

        # Create Run Directory
        os.makedirs(self.run_dir / self.model_id, exist_ok=True)

    def get_datamodule(self) -> LightningDataModule:
        return build_datamodule(self.data, self.bsz, self.preprocess)

    def fit(self, test_only = False, iou_threshold = 0.25) -> None:
        overwatch.info("Invoking ReferDetectionHarness.fit()")

        # Instantiate DataModule
        overwatch.info("Starting Dataset Processing")
        dm = self.get_datamodule()

        # Create Adapter Model & Callbacks
        overwatch.info("Instantiating Adapter Model and Callbacks")
        detector = self.detector_init_fn(self.backbone, self.extractor_init_fn(), \
            flag_dual = False, iou_threshold = iou_threshold, lr = self.lr, \
            load_checkpoint = self.load_checkpoint, language_model_path = self.language_model_path)
            
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(self.run_dir / self.model_id),
            filename="{epoch:02d}-{val_loss:0.4f}-{total_acc25:0.4f}.pt",
            monitor="total_acc25",
            mode="max",
            save_top_k=1,
        )

        trainer = Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                max_epochs=self.epochs,
                log_every_n_steps=-1,
                logger=None,
                callbacks=[checkpoint_callback],
            )

        if not test_only:
            overwatch.info("Training...")
            trainer.fit(detector, datamodule=dm)

        # Get final test metrics & serialize...
        test_metrics = trainer.test(detector, datamodule=dm)
        with open(self.run_dir / self.model_id / "final-metrics.json", "w") as f:
            json.dump(
                {
                    "total_acc25": test_metrics[0]["test_total_acc25"],
                    "free_acc25": test_metrics[0]["test_free_acc25"],
                    "touching_acc25": test_metrics[0]["test_touching_acc25"],
                    "stacked_acc25": test_metrics[0]["test_stacked_acc25"],
                },
                f,
                indent=4,
            )

    def test(self) -> None:
        overwatch.info("Compiling Refer Detection Test Metrics")
        with open(self.run_dir / self.model_id / "final-metrics.json", "r") as f:
            metrics = json.load(f)

        # Print Test Metrics!
        overwatch.info("Referring Expression Grounding =>> Test Metrics")
        for mname, mkey in [
            ("Total Accuracy", "total_acc25"),
            ("Accuracy on Free Split (Easy)", "free_acc25"),
            ("Accuracy on Touching Split (Medium)", "touching_acc25"),
            ("Accuracy on Stacked Split (Hard)", "stacked_acc25"),
        ]:
            overwatch.info(f"\t{mname}: {metrics[mkey]:0.4f}")


