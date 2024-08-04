"""
extraction.py

General Extraction module definitions & associated utilities.

References:
    - Set Transformers (MAP): https://arxiv.org/abs/1810.00825.pdf
"""
from typing import Callable

import torch
import torch.nn as nn
from einops import repeat

from mpi.models.util.transformer import RMSNorm, SwishGLU

# === Multiheaded Attention Pooling ===


# As defined in Set Transformers () -- basically the above, additionally taking in
# a set of $k$ learned "seed vectors" that are used to "pool" information.
class MAPAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int) -> None:
        """Multi-Input Multi-Headed Attention Operation"""
        super().__init__()
        assert embed_dim % n_heads == 0, "`embed_dim` must be divisible by `n_heads`!"
        self.n_heads, self.scale = n_heads, (embed_dim // n_heads) ** -0.5

        # Projections (no bias) --> separate for Q (seed vector), and KV ("pool" inputs)
        self.q, self.kv = nn.Linear(embed_dim, embed_dim, bias=False), nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, seed: torch.Tensor, x: torch.Tensor, attention_mask = None) -> torch.Tensor:
        (B_s, K, C_s), (B_x, N, C_x) = seed.shape, x.shape
        assert C_s == C_x, "Seed vectors and pool inputs must have the same embedding dimensionality!"
        # Project Seed Vectors to `queries`
        q = self.q(seed).reshape(B_s, K, self.n_heads, C_s // self.n_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B_x, N, 2, self.n_heads, C_x // self.n_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        # Attention --> compute weighted sum over values!
        scores = q @ (k.transpose(-2, -1) * self.scale)
        if attention_mask is not None:
            attention_mask = (
                attention_mask[:, None, None, :].repeat(1, self.n_heads, 1, 1) #.flatten(0, 1)
            )
            scores.masked_fill_(attention_mask == 0, float("-inf"))
        attn = scores.softmax(dim=-1)
        vals = (attn @ v).transpose(1, 2).reshape(B_s, K, C_s)

        # Project back to `embed_dim`
        return self.proj(vals)


class MAPBlock(nn.Module):
    def __init__(
        self,
        n_latents: int,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        do_rms_norm: bool = True,
        do_swish_glu: bool = True,
    ) -> None:
        """Multiheaded Attention Pooling Block -- note that for MAP, we adopt earlier post-norm conventions."""
        super().__init__()
        self.n_latents, self.embed_dim, self.n_heads = n_latents, embed_dim, 2 * n_heads

        # Projection Operator
        self.projection = nn.Linear(embed_dim, self.embed_dim)

        # Initialize Latents
        self.latents = nn.Parameter(torch.zeros(self.n_latents, self.embed_dim), requires_grad=True)
        nn.init.normal_(self.latents, std=0.02)

        # Custom MAP Attention (seed, encoder outputs) -> seed
        self.attn_norm = RMSNorm(self.embed_dim) if do_rms_norm else nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.attn = MAPAttention(self.embed_dim, n_heads=self.n_heads)

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
        latents = repeat(self.latents, "n_latents d -> bsz n_latents d", bsz=x.shape[0])
        latents = latents + init_embed.unsqueeze(1) if init_embed is not None else latents
        latents = self.attn_norm(latents + self.attn(latents, self.projection(x)))
        latents = self.mlp_norm(latents + self.mlp(latents))
        return latents.squeeze(dim=1)


class MLP_Adapter(nn.Module):
    def __init__(
        self,
        n_latents: int,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        do_rms_norm: bool = True,
        do_swish_glu: bool = True,
    ) -> None:
        """Multiheaded Attention Pooling Block -- note that for MAP, we adopt earlier post-norm conventions."""
        super().__init__()
        self.n_latents, self.embed_dim, self.n_heads = n_latents, embed_dim, 2 * n_heads


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latents = self.mlp_norm(x + self.mlp(x))
        return latents.squeeze(dim=1)


class TokenAggregation(nn.Module):
    def __init__(
        self,
        n_latents: int,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        do_rms_norm: bool = True,
        do_swish_glu: bool = True,
        #add_internal_latents: bool = False,
    ) -> None:
        """Multiheaded Attention Pooling Block -- note that for MAP, we adopt earlier post-norm conventions."""
        super().__init__()
        self.n_latents, self.embed_dim, self.n_heads = n_latents, embed_dim, 2 * n_heads

        # Projection Operator
        self.projection = nn.Linear(embed_dim, self.embed_dim)

        # Custom MAP Attention (seed, encoder outputs) -> seed
        self.attn_norm = RMSNorm(self.embed_dim) if do_rms_norm else nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.attn = MAPAttention(self.embed_dim, n_heads=self.n_heads)

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



    def forward(self, x: torch.Tensor, latents: torch.Tensor = None, mask = None) -> torch.Tensor:
        latents = repeat(latents, "n_latents d -> bsz n_latents d", bsz=x.shape[0])
        latents = self.attn_norm(latents + self.attn(latents, self.projection(x), mask))
        latents = self.mlp_norm(latents + self.mlp(latents))
        return latents.squeeze(dim=1)



class MeanExtractor(nn.Module):
    def __init__(  
        self,
        n_latents: int,
        embed_dim: int,
        n_heads: int, ) -> None:
        super().__init__()
        self.n_latents, self.embed_dim, self.n_heads = n_latents, embed_dim, 2 * n_heads
        
    def forward(self, x : torch.Tensor):
        return x.mean(dim=1)


class IdentityExtractor(nn.Module):
    def __init__(  
        self,
        n_latents: int,
        embed_dim: int,
        n_heads: int, ) -> None:
        super().__init__()
        self.n_latents, self.embed_dim, self.n_heads = n_latents, embed_dim, 2 * n_heads
        
    def forward(self, x : torch.Tensor):
        return x


# MAP Extractor Instantiation --> factory for creating extractors with the given parameters.
def instantiate_extractor(backbone: nn.Module, n_latents: int = 1, mode_extractor: str = 'MAP') -> Callable[[], nn.Module]:
    if mode_extractor == "MAP":
        def initialize() -> nn.Module:
            return MAPBlock(n_latents, backbone.embed_dim, backbone.n_heads)
    elif mode_extractor == "Mean":
        def initialize() -> nn.Module:
            return MeanExtractor(n_latents, backbone.embed_dim, backbone.n_heads)
    elif mode_extractor == "Identity":
        def initialize() -> nn.Module:
            return IdentityExtractor(n_latents, backbone.embed_dim, backbone.n_heads)
    elif mode_extractor == "MLP":
        def initialize() -> nn.Module:
            return MLP_Adapter(n_latents, backbone.embed_dim, backbone.n_heads)
    else:
        raise NameError("Invalid extractor mode")
    
    return initialize
