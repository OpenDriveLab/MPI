from .models.materialize import load
from .models.util import instantiate_extractor
from mpi.models.mpi_model import MPI
import os
import glob
import torch
from omegaconf import OmegaConf

def load_mpi(root_dir, device, freeze=True):
    checkpoint_path = glob.glob(os.path.join(root_dir, '*.pt'))[0]
    config_path = glob.glob(os.path.join(root_dir, '*.json'))[0]
    config = OmegaConf.load(config_path)
    dataset_cfg = config['dataset']
    model_cfg = config['model']
    model = MPI(
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
    state_dict = torch.load(checkpoint_path, map_location=device)
    new_state_dict = {}
    for name, param in state_dict.items():
        new_state_dict[name.replace('module.', '')] = param
    model.load_state_dict(new_state_dict, strict=True)
    if freeze:
        for _, param in model.named_parameters():
            param.requires_grad = False
    model.to(device)
    model.eval()
    return model
    
