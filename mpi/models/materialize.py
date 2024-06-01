# ------------------------------------------------------------------------------------------------
# Modified from:
# Voltron: https://github.com/siddk/voltron-robotics
# ------------------------------------------------------------------------------------------------
import json
import os
from pathlib import Path
from typing import Callable, List, Tuple

import gdown
import torch
import torch.nn as nn
import torchvision.transforms as T
from mpi.models.instantiate import get_model_optimizer

# === Define Useful Variables for Loading Models ===
DEFAULT_CACHE = "cache/"
NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# # Pretrained Model Registry :: "model id" -> {"config" -> gdown ID, "checkpoint" -> gdown ID, "cls" -> Model Class}
# MODEL_REGISTRY = {
#     # === Voltron ViT-Small (Sth-Sth) Models ===
#     "v-cond": {
#         "config": "1O4oqRIblfS6PdFlZzUcYIX-Rqe6LbvnD",
#         "checkpoint": "12g5QckQSMKqrfr4lFY3UPdy7oLw4APpG",
#         "cls": VCond,
#     },
#     "v-dual": {
#         "config": "1zgKiK81SF9-0lg0XbMZwNhUh1Q7YdZZU",
#         "checkpoint": "1CCRqrwcvF8xhIbJJmwnCbcWfWTJCK40T",
#         "cls": VDual,
#     },
#     "v-gen": {
#         "config": "18-mUBDsr-2_-KrGoL2E2YzjcUO8JOwUF",
#         "checkpoint": "1TzSQpKVKBWKCSvYJf22c45hrKczTQz24",
#         "cls": VGen,
#     },
#     # === Voltron ViT-Base Model ===
#     "v-cond-base": {
#         "config": "1CLe7CaIzTEcGCijIgw_S-uqMXHfBFSLI",
#         "checkpoint": "1PwczOijL0hfYD8DI4xLOPLf1xL_7Kg9S",
#         "cls": VCond,
#     },
#     # === Data-Locked Reproductions ===
#     "r-mvp": {
#         "config": "1KKNWag6aS1xkUiUjaJ1Khm9D6F3ROhCR",
#         "checkpoint": "1-ExshZ6EC8guElOv_s-e8gOJ0R1QEAfj",
#         "cls": VMVP,
#     },
#     "r-r3m-vit": {
#         "config": "1JGk32BLXwI79uDLAGcpbw0PiupBknf-7",
#         "checkpoint": "1Yby5oB4oPc33IDQqYxwYjQV3-56hjCTW",
#         "cls": VR3M,
#     },
#     "r-r3m-rn50": {
#         "config": "1OS3mB4QRm-MFzHoD9chtzSmVhOA-eL_n",
#         "checkpoint": "1t1gkQYr6JbRSkG3fGqy_9laFg_54IIJL",
#         "cls": VRN3M,
#     },
# }

# def available_models() -> List[str]:
#     return list(MODEL_REGISTRY.keys())


def load(
    model_id: str, device: torch.device = "cpu", freeze: bool = True, config = None, checkpoint_path: str = None,
) -> Tuple[nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    """
    Download & cache specified model configuration & checkpoint, then load & return module & image processor.

    Note :: We *override* the default `forward()` method of each of the respective model classes with the
            `extract_features` method --> by default passing "NULL" language for any language-conditioned models.
            This can be overridden either by passing in language (as a `str) or by invoking the corresponding methods.
    """

    dataset_cfg = config['dataset']
    model_cfg = config['model']
    # print(model_cfg)

    model, _, __ = get_model_optimizer(model_cfg, dataset_cfg)
    
    # By default, the model's `__call__` method defaults to `forward` --> for downstream applications, override!
    #   > Switch `__call__` to `get_representations`
    # MODEL_REGISTRY[model_id]["cls"].__call__ = MODEL_REGISTRY[model_id]["cls"].get_representations

    model.__call__ = model.get_representations
    state_dict = torch.load(checkpoint_path, map_location=device)#['model_state_dict']
    # print(state_dict)
    
    new_state_dict = {}
    for name, param in state_dict.items():
        new_state_dict[name.replace('module.', '')] = param
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()

    # Freeze model parameters if specified (default: True)
    if freeze:
        for _, param in model.named_parameters():
            param.requires_grad = False

    # Build Visual Preprocessing Transform (assumes image is read into a torch.Tensor, but can be adapted)
    # if model_id in {"v-cond", "v-dual", "v-gen", "v-cond-base", "r-mvp"}:
        # All models except R3M are by default normalized subject to default IN1K normalization...
    preprocess = T.Compose(
        [
            T.Resize(224, antialias=True),
            # T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=NORMALIZATION[0], std=NORMALIZATION[1]),
        ]
    )

    return model, preprocess

# if __name__ == '__main__':
#     gdown.download(id=MODEL_REGISTRY['v-cond']["config"], output='./', quiet=False)
#     gdown.download(id=MODEL_REGISTRY['v-cond']["checkpoint"], output='./', quiet=False)