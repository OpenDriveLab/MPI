"""
preprocessing.py

Utilities for loading and compiling the OCID-Ref Referring Expression Grounding Dataset into a Lightning DataModule.
"""
import json
import logging
import os
from bisect import bisect_left
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
import torchvision.transforms.functional as F
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.ops import box_convert
from tqdm import tqdm

# Grab Logger
overwatch = logging.getLogger(__file__)


class ReferRGBDataset(Dataset):
    def __init__(
        self,
        split: str,
        data: Path,
        examples: Dict[str, Dict[str, str]],
        preprocess: Callable[[torch.Tensor], torch.Tensor],
        pad_resolution: Tuple[int, int] = (672, 672),
        input_resolution: Tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        self.split, self.data, self.examples, self.preprocess = split, data, examples, preprocess
        self.pad_resolution, self.input_resolution = pad_resolution, input_resolution

        # Assert downsample factor is an integer, and same across dimensions (square)!
        assert pad_resolution[0] % input_resolution[0] == pad_resolution[1] % input_resolution[1] == 0, "Padding!"
        assert pad_resolution[0] // input_resolution[0] == pad_resolution[1] // input_resolution[1] == 3, "Padding!"

        # Get left/top padding to shift bbox coordinates --> in addition, compute downsample factor...
        self.dx, self.dy = (self.pad_resolution[1] - 640) // 2, (self.pad_resolution[0] - 480) // 2
        self.downsample_factor = self.pad_resolution[0] // self.input_resolution[0]

        # Create Dataset Components
        self.rgb_paths, self.language, self.bboxes, self.clutter_splits = [], [], [], []

        # Dataset is a bit big... let's cache!
        os.makedirs(self.data / "refer-cache", exist_ok=True)
        cached_pt = self.data / "refer-cache" / f"{self.split}-rgb-dataset.pt"
        if cached_pt.exists():
            self.rgb_paths, self.language, self.bboxes, self.clutter_splits = torch.load(str(cached_pt))
            assert (
                {"train": 259839, "val": 18342, "test": 27513}.get(self.split, -1)
                == len(self.rgb_paths)
                == len(self.language)
                == len(self.bboxes)
                == len(self.clutter_splits)
                == len(self.examples)
            ), "Error on load from cache!"

        else:
            self.rgb_paths, self.language, self.bboxes, self.clutter_splits = self.process_dataset()
            torch.save([self.rgb_paths, self.language, self.bboxes, self.clutter_splits], str(cached_pt))

    def process_dataset(self) -> Tuple[List[str], List[str], torch.Tensor, torch.Tensor]:
        # Define Clutter Split and Mappings (from OCID):
        #   The `take_id` corresponds to the index of images taken on a given table-top over time, with the table
        #   getting more cluttered over time. There are three splits that OCID-REF uses:
        #   - "free" [split = 0] --> clearly separated objects, indices [0 --> 9] (inclusive)
        #   - "touching" [split = 1] --> touching objects (moderate clutter), indices [10 --> 16] (inclusive)
        #   - "stacked" [split = 2] --> stacked, touching objects (max clutter), indices [17 --> 20] (inclusive)
        # Reference: https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-clutter-indoor-dataset/
        def get_split(take_id: int) -> int:
            assert 0 <= take_id <= 20, "Bounds violation - `take_id` must be in [0, 20]!"
            return bisect_left([9, 16, 20], take_id)

        # Iterate through `self.examples` --> retrieve input RGBs, language, bbox coordinates, and clutter splits!
        #   => For BBox coordinates (x1, y1, x2, y2) :: shift by padding, then apply `self.downsample_factor`
        #   => Additionally, for OCID-Ref :: keep "clutter split" for each example
        rgb_paths, language, bboxes, clutter_splits = [], [], [], []
        for _idx, key in tqdm(enumerate(self.examples), total=len(self.examples), leave=False):
            example = self.examples[key]

            # Parse Example => rgb_path :: str, lang :: str, bbox :: torch.Tensor, clutter split :: int
            rgb_path, lang = str(self.data / "OCID-dataset" / example["scene_path"]), example["sentence"]
            bbox = torch.tensor(json.loads(example["bbox"]), dtype=torch.int)
            clutter_split = get_split(example["take_id"])

            # Note :: though OCID-Ref claims BBox are in `xywh` format --> they are actually in x1-y1-x2-y2 format!
            #   => Assertion to ensure!
            assert bbox[0] < bbox[2] <= 640 and bbox[1] < bbox[3] <= 480, "Invalid Bounding Box Size!"

            # Compute BBox Coordinates after padding
            bbox_pad = bbox + torch.tensor([self.dx, self.dy, self.dx, self.dy], dtype=torch.int)

            # Scale BBox Coordinates appropriately (floordiv w/ downsample factor)
            bbox_scaled = torch.div(bbox_pad, self.downsample_factor, rounding_mode="floor")

            # Convert BBox to xywh format (make predictions valid)...
            bbox_xywh = box_convert(bbox_scaled, in_fmt="xyxy", out_fmt="xywh")

            # Add to Trackers --> note that we pad/preprocess the RGB images *in __getitem__()*
            rgb_paths.append(rgb_path)
            language.append('pick up' + lang)
            bboxes.append(bbox_xywh)
            clutter_splits.append(clutter_split)

        # Return (as Tensors if applicable)
        return rgb_paths, language, torch.stack(bboxes), torch.tensor(clutter_splits, dtype=torch.int)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, torch.Tensor, torch.Tensor]:
        """Read image from RGB Path, apply padding, downsampling, & image transforms, then yield batch element."""
        rgb_raw = read_image(self.rgb_paths[idx])
        rgb_pad = F.pad(rgb_raw, [self.dx, self.dy], fill=0, padding_mode="constant")
        rgb_final = self.preprocess(rgb_pad)
    
        return rgb_final, self.language[idx], self.bboxes[idx], self.clutter_splits[idx]

    def __len__(self) -> int:
        return len(self.rgb_paths)


class ReferRGBDataModule(LightningDataModule):
    def __init__(
        self,
        data: Path,
        bsz: int,
        preprocess: Callable[[torch.Tensor], torch.Tensor],
        pad_resolution: Tuple[int, int] = (672, 672),
        input_resolution: Tuple[int, int] = (224, 224),
    ) -> None:
        """
        Initialize an OCID-Ref ReferRGBDataModule. Note that we'll be padding each of the images in their native
        (640 x 480) resolution to be (672 x 672); this is because 672 is a multiple of 224, the expected resolution for
        the various backbones; these multiples allow for nice, round numbers!
        """
        super().__init__()
        self.data, self.pad_resolution, self.input_resolution = data, pad_resolution, input_resolution
        self.bsz, self.preprocess = bsz, preprocess

        # Read in language expression data from `train_expressions.json`
        with open(self.data / "train_expressions.json", "r") as f:
            self.train_examples = json.load(f)

        # Read in language expression data from `train_expressions.json`
        with open(self.data / "val_expressions.json", "r") as f:
            self.val_examples = json.load(f)

        # Create Train & Validation Datasets
        overwatch.info("Compiling Referring Expression Train Dataset")
        self.train_dataset = ReferRGBDataset(
            "train",
            self.data,
            self.train_examples,
            self.preprocess,
            self.pad_resolution,
            self.input_resolution,
        )

        overwatch.info("Compiling Referring Expression Validation Dataset")
        self.val_dataset = ReferRGBDataset(
            "val",
            self.data,
            self.val_examples,
            self.preprocess,
            self.pad_resolution,
            self.input_resolution,
        )

        # Read in language expression data from `test_expressions.json` and create Test Dataset
        with open(self.data / "test_expressions.json", "r") as f:
            self.test_examples = json.load(f)

        overwatch.info("Compiling Referring Expression Test Dataset")
        self.test_dataset = ReferRGBDataset(
            "test",
            self.data,
            self.test_examples,
            self.preprocess,
            self.pad_resolution,
            self.input_resolution,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.bsz, num_workers=8, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.bsz, num_workers=4, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.bsz, num_workers=4, shuffle=False)


def build_datamodule(data_path: Path, bsz: int, preprocess: Callable[[torch.Tensor], torch.Tensor]) -> LightningDataModule:
    return ReferRGBDataModule(data_path, bsz, preprocess)
