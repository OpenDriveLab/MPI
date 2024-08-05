# ------------------------------------------------------------------------------------------------
# Modified from:
# R3M: https://github.com/facebookresearch/r3m/tree/eval/evaluation
# ------------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import numpy as np
import gym
from gym.spaces.box import Box
import omegaconf
import torch
import torch.nn as nn
from torch.nn.modules.linear import Identity
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import pickle
from torchvision.utils import save_image
import hydra

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import functools
from textwrap import dedent
from typing import Any, Callable, Optional

from omegaconf import DictConfig

from hydra._internal.deprecation_warning import deprecation_warning
from hydra._internal.utils import _run_hydra, get_args_parser
from hydra.types import TaskFunction

_UNSPECIFIED_: Any = object()


import json
import os
from pathlib import Path
from typing import Callable, List, Tuple
import glob
import gdown
import torch
import torch.nn as nn
import torchvision.transforms as T
from mpi.models.mpi_model import MPI
from mpi.models.util.extraction import MeanExtractor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from omegaconf import MISSING, OmegaConf
from mpi.configs import AcceleratorConfig, DatasetConfig, ModelConfig, TrackingConfig
from hydra.core.config_store import ConfigStore

promot_for_env_name = {
    # Franka kitchen task prompts
    'kitchen_knob1_on-v3': 'turn on the knob',
    'kitchen_sdoor_open-v3': 'slide open the cabinet door',
    'kitchen_ldoor_open-v3': 'open the door of the cabinet',
    'kitchen_micro_open-v3': 'open the microwave',
    'kitchen_light_on-v3': 'flip light switch',
}

def load_mpi(model_id: str, path_ckpt: str, device: torch.device = "cpu", freeze: bool = True) -> Tuple[nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    """
    Download & cache specified model configuration & checkpoint, then load & return module & image processor.

    Note :: We *override* the default `forward()` method of each of the respective model classes with the
            `extract_features` method --> by default passing "NULL" language for any language-conditioned models.
            This can be overridden either by passing in language (as a `str) or by invoking the corresponding methods.
    """

    root_dir = os.path.join(path_ckpt, model_id)
    checkpoint_path = glob.glob(os.path.join(root_dir, '*.pt'))[0]
    config_path = glob.glob(os.path.join(root_dir, '*.json'))[0]
    from omegaconf import OmegaConf
    config = OmegaConf.load(config_path)
    dataset_cfg = config['dataset']
    model_cfg = config['model']
    encoder = MPI(
            resolution=dataset_cfg.resolution,
            patch_size=model_cfg.patch_size,
            encoder_depth=model_cfg.encoder_depth,
            encoder_embed_dim=model_cfg.encoder_embed_dim,
            encoder_n_heads=model_cfg.encoder_n_heads,
            decoder_depth=model_cfg.decoder_depth,
            decoder_embed_dim=model_cfg.decoder_embed_dim,
            decoder_n_heads=model_cfg.decoder_n_heads,
            language_model=model_cfg.language_model,
            hf_cache=model_cfg.hf_cache,
            language_dim=model_cfg.language_dim,
            optimizer=model_cfg.optimizer,
            schedule=model_cfg.schedule,
            base_lr=model_cfg.base_lr,
            min_lr=model_cfg.min_lr,
            effective_bsz=model_cfg.effective_bsz,
            betas=model_cfg.betas,
            weight_decay=model_cfg.weight_decay,
            warmup_epochs=dataset_cfg.warmup_epochs,
            max_epochs=dataset_cfg.max_epochs,
            mlp_ratio=model_cfg.mlp_ratio,
            norm_pixel_loss=model_cfg.norm_pixel_loss,
        )
    encoder.__call__ = encoder.get_representations
    print("===" * 50)
    print("start load checkpoint: ", checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=device)
    print("finish load checkpoint")
    print("successfully load checkpoint of MPI")
    new_state_dict = {}
    for name, param in state_dict.items():
        new_state_dict[name.replace('module.', '')] = param
    encoder.load_state_dict(new_state_dict, strict=True)

    # Freeze model parameters if specified (default: True)
    if freeze:
        for _, param in encoder.named_parameters():
            param.requires_grad = False

    # Build Visual Preprocessing Transform (assumes image is read into a torch.Tensor, but can be adapted)
    # if model_id in {"v-cond", "v-dual", "v-gen", "v-cond-base", "r-mvp"}:
        # All models except R3M are by default normalized subject to default IN1K normalization...
    preprocess = T.Compose(
        [
            T.Resize(256, antialias=True),
            T.CenterCrop(224),
            T.ToTensor(), # ToTensor() divides by 255
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
        ]
    )
    model = encoder
    model.to(device)
    model.eval()
    return model, preprocess, model_cfg.encoder_embed_dim


class StateEmbedding(gym.ObservationWrapper):
    """
    This wrapper places a convolution model over the observation.

    From https://pytorch.org/vision/stable/models.html
    All pre-trained models expect input images normalized in the same way,
    i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
    where H and W are expected to be at least 224.

    Args:
        env (Gym environment): the original environment,
        embedding_name (str, 'baseline'): the name of the convolution model,
        device (str, 'cuda'): where to allocate the model.

    """
    def __init__(self, env, embedding_name=None, device='cuda', load_path="", proprio=0, camera_name=None, 
                 env_name=None, path_demo= "", path_ckpt=""):
        gym.ObservationWrapper.__init__(self, env)

        self.proprio = proprio
        self.load_path = load_path
        self.start_finetune = False
        self.env_name = env_name
        if load_path == "r3m":
            from r3m import load_r3m_reproduce
            rep = load_r3m_reproduce("r3m")
            rep.eval()
            embedding_dim = rep.module.outdim
            embedding = rep
            self.transforms = T.Compose([T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor()]) # ToTensor() divides by 255
        elif "mpi" in load_path:
            def evaluate_refer() -> None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                backbone, preprocess, embedding_dim = load_mpi(load_path, path_ckpt=path_ckpt, device=device)
                return backbone, preprocess, embedding_dim
            embedding, self.transforms, embedding_dim = evaluate_refer()
        else:
            raise NameError("Invalid Model")
        embedding.eval()
        if device == 'cuda' and torch.cuda.is_available():
            print('Using CUDA.')
            device = torch.device('cuda')
        else:
            print('Not using CUDA.')
            device = torch.device('cpu')
        self.device = device
        embedding.to(device=device)

        self.embedding, self.embedding_dim = embedding, embedding_dim
        # self.observation_space = Box(
        #             low=-np.inf, high=np.inf, shape=(self.embedding_dim+self.proprio,))
        self.observation_space = Box(
                    low=-np.inf, high=np.inf, shape=(self.embedding_dim,))

    def observation(self, observation):
        ### INPUT SHOULD BE [0,255]
        if self.embedding is not None: # self.embedding: encoder model
            inp = self.transforms(Image.fromarray(observation.astype(np.uint8))).unsqueeze(0)
            if "r3m" in self.load_path:
                ## R3M Expects input to be 0-255, preprocess makes 0-1
                inp *= 255.0
            elif "mpi" in self.load_path:
                inp = torch.stack((inp, inp), dim=1)
                lang = (promot_for_env_name[self.env_name],)
            inp = inp.to(self.device)
            with torch.no_grad():
                if "mpi" in self.load_path:
                    emb = self.embedding.get_representations(inp, lang) # False
                    emb = emb.to('cpu').numpy().squeeze()
                elif "r3m" in self.load_path:
                    emb = self.embedding.module.obtain_feature_before_pooling(inp)
                    emb = emb.view(emb.shape[0], emb.shape[1], -1).permute(0, 2, 1)
                    emb = emb.to('cpu').numpy().squeeze()
                else:
                    emb = self.embedding(inp).view(-1, self.embedding_dim).to('cpu').numpy().squeeze()

            ## IF proprioception add it to end of embedding
            if self.proprio:
                try:
                    proprio = self.env.unwrapped.get_obs()[:self.proprio]
                except:
                    proprio = self.env.unwrapped._get_obs()[:self.proprio]
                emb = {'obs': emb, 'proprio': proprio}
            else:
                emb = {'obs': emb, 'proprio': None}
            return emb
        else:
            return observation

    def encode_batch(self, data, idx, finetune=False):
        ### INPUT SHOULD BE [0,255]
        images = []
        langs = []
        obs = data['observations'][idx]
        if 'mpi' in self.load_path:
            for i_obs in range(len(obs)):
                img = self.transforms(Image.fromarray(obs[i_obs].astype(np.uint8))).unsqueeze(0)
                images.append(torch.stack((img, img), dim=1))
                langs.append(promot_for_env_name[self.env_name])
        else:
            for o in obs:
                img = self.transforms(Image.fromarray(o.astype(np.uint8))).unsqueeze(0)
                if "r3m" in self.load_path:
                    ## R3M Expects input to be 0-255, preprocess makes 0-1
                    img *= 255.0
                    images.append(img)
                else:
                    images.append(img)         
        inp = torch.cat(images)
        inp = inp.to(self.device)
        if finetune and self.start_finetune:
            if "mpi" in self.load_path:
                emb = self.embedding.get_representations(inp, langs)
            else:
                emb = self.embedding(inp).view(-1, self.embedding_dim)
        else:
            with torch.no_grad():
                if "mpi" in self.load_path:
                    emb = self.embedding.get_representations(inp, langs) # False
                elif "r3m" in self.load_path:
                    emb = self.embedding.module.obtain_feature_before_pooling(inp)
                    emb = emb.view(emb.shape[0], emb.shape[1], -1).permute(0, 2, 1)
                    emb = emb.to('cpu').numpy().squeeze()
                else:
                    emb = self.embedding(inp).view(-1, self.embedding_dim).to('cpu').numpy().squeeze()
        return emb

    def get_obs(self):
        if self.embedding is not None:
            return self.observation(self.env.observation(None))
        else:
            # returns the state based observations
            return self.env.unwrapped.get_obs()
          
    def start_finetuning(self):
        self.start_finetune = True


class MuJoCoPixelObs(gym.ObservationWrapper):
    def __init__(self, env, width, height, camera_name, device_id=-1, depth=False, *args, **kwargs):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(low=0., high=255., shape=(3, width, height))
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.depth = depth
        self.device_id = device_id
        if "v2" in env.spec.id:
            self.get_obs = env._get_obs

    def get_image(self):
        img = self.sim.render(width=self.width, height=self.height, depth=self.depth,
                            camera_name=self.camera_name, device_id=self.device_id)
        img = img[::-1,:,:]
        return img

    def observation(self, observation):
        # This function creates observations based on the current state of the environment.
        # Argument `observation` is ignored, but `gym.ObservationWrapper` requires it.
        return self.get_image()
        
