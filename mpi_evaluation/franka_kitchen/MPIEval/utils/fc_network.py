# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torch.nn as nn
from mpi.models.util.extraction import MAPBlock
from mpi.models.util.transformer import RMSNorm, SwishGLU
import torch.nn.functional as F

class FCNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim,
                 hidden_sizes=(64,64),
                 nonlinearity='relu',   # either 'tanh' or 'relu'
                 in_shift = None,
                 in_scale = None,
                 out_shift = None,
                 out_scale = None):
        super(FCNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        assert type(hidden_sizes) == tuple
        self.layer_sizes = (obs_dim, ) + hidden_sizes + (act_dim, )
        self.set_transformations(in_shift, in_scale, out_shift, out_scale)
        self.proprio_only = False

        # Batch Norm Layers
        self.bn = torch.nn.BatchNorm1d(obs_dim)

        # hidden layers (three layers)
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]) \
                         for i in range(len(self.layer_sizes) -1)])
        self.nonlinearity = torch.relu if nonlinearity == 'relu' else torch.tanh

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        # store native scales that can be used for resets
        self.transformations = dict(in_shift=in_shift,
                           in_scale=in_scale,
                           out_shift=out_shift,
                           out_scale=out_scale
                          )
        # self.in_shift  = torch.from_numpy(np.float32(in_shift)) if in_shift is not None else torch.zeros(self.obs_dim)
        # self.in_scale  = torch.from_numpy(np.float32(in_scale)) if in_scale is not None else torch.ones(self.obs_dim)
        self.out_shift = torch.from_numpy(np.float32(out_shift)) if out_shift is not None else torch.zeros(self.act_dim)
        self.out_scale = torch.from_numpy(np.float32(out_scale)) if out_scale is not None else torch.ones(self.act_dim)

    def forward(self, x):
        # Small MLP runs on CPU
        # Required for the way the Gaussian MLP class does weight saving and loading.
        
        if x.is_cuda:
            out = x.to('cpu')
        else:
            out = x
        
        ## BATCHNORM
        out = self.bn(out)
        for i in range(len(self.fc_layers)-1):
            out = self.fc_layers[i](out)
            out = self.nonlinearity(out)
        out = self.fc_layers[-1](out)
        # print(out[0])
        out = out * self.out_scale + self.out_shift
        # print(out[0])
        # out = F.sigmoid(out)
        return out

class MAP_Adapter(nn.Module):
    def __init__(self, 
                 n_latents,
                 obs_dim, 
                 n_heads):
        super(MAP_Adapter, self).__init__()
        self.adapter = MAPBlock(n_latents = n_latents, embed_dim = obs_dim, n_heads = n_heads)


    def forward(self, x, init_embed = None):
        out = self.adapter(x, init_embed)
        return out


class MLP_Adapter(nn.Module):
    def __init__(
        self,
        n_latents: int,
        obs_dim: int,
        n_heads: int,
        mlp_ratio: float = 1.0,
        do_rms_norm: bool = True,
        do_swish_glu: bool = True,
    ) -> None:
        """Multiheaded Attention Pooling Block -- note that for MAP, we adopt earlier post-norm conventions."""
        super().__init__()
        self.n_latents, self.embed_dim, self.n_heads = n_latents, obs_dim * 2, 2 * n_heads


        # Position-wise Feed-Forward Components
        self.mlp_norm = RMSNorm(self.embed_dim) if do_rms_norm else nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            # Handle SwishGLU vs. GELU MLP...
            (
                SwishGLU(self.embed_dim, int(mlp_ratio * self.embed_dim))
                if do_swish_glu
                else nn.Sequential(nn.Linear(self.embed_dim, int(mlp_ratio * self.embed_dim)), nn.GELU())
            ),
            nn.Linear(int(mlp_ratio * self.embed_dim), self.embed_dim),
        )

    def forward(self, x: torch.Tensor, init_embed = None) -> torch.Tensor:
        
        x = torch.cat([x, init_embed], dim=-1)
        latents = self.mlp_norm(x + self.mlp(x))
        return latents.squeeze(dim=1)