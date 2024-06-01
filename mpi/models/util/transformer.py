"""
transformer.py

General Transformer modules & utilities.

References:
    - https://github.com/facebookresearch/mae
    - https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mpi.models.util.ms_deform_attn import MultiScaleDeformableAttention
from mpi.models.util.attention import BiAttentionBlock


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=128,
        dropout=0.
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

# === Position Encoding Utilities ===


# Helper/Utility Function -- computes simple 1D sinusoidal position embeddings for both 1D/2D use cases.
#   > We'll be combining two 1D sin-cos (traditional) position encodings for height/width of an image (grid features).
def get_1D_sine_cosine(dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(dim // 2, dtype=np.float32) / (dim / 2.0)
    omega = 1.0 / (10000**omega)
    out = np.einsum("m,d->md", pos.reshape(-1), omega)  # [flatten(pos) x omega] -- outer product!
    emb_sin, emb_cos = np.sin(out), np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # [flatten(pos) x D]


# 1D Sine-Cosine Position Embedding -- standard from "Attention is all you need!"
def get_1D_position_embeddings(embed_dim: int, length: int) -> np.ndarray:
    return get_1D_sine_cosine(embed_dim, np.arange(length))


# 2D Sine-Cosine Position Embedding (from MAE repository)
#   > https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2D_position_embeddings(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    # Create 2D Position embeddings by taking cross product of height and width and splicing 1D embeddings...
    grid_h, grid_w = np.arange(grid_size, dtype=np.float32), np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape(2, 1, grid_size, grid_size)  # w goes first?

    # Use half of dimensions to encode grid_h, other half to encode grid_w
    emb_h, emb_w = get_1D_sine_cosine(embed_dim // 2, grid[0]), get_1D_sine_cosine(embed_dim // 2, grid[1])
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)

    # CLS token handling (only for R-MVP)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed


# === Vision Transformer Building Blocks ===


# Patch Embedding Module
class PatchEmbed(nn.Module):
    def __init__(
        self,
        resolution: int,
        patch_size: int,
        embed_dim: int,
        in_channels: int = 3,
        flatten: bool = True,
    ):
        super().__init__()
        self.resolution, self.patch_size = (resolution, resolution), (patch_size, patch_size)
        self.grid_size = (self.resolution[0] // self.patch_size[0], self.resolution[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        patch_embeddings = self.proj(patches)
        if self.flatten:
            return rearrange(patch_embeddings, "bsz embed patch_h patch_w -> bsz (patch_h patch_w) embed")
        return patch_embeddings


# === Stability Utilities ===


# LayerScale -- Trainable scaling for residual blocks -- Mistral/CaIT
class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 0.1) -> None:  # CaIT :: 0.1 -> lay 12, 1e-5 -> lay 24, 1e-6...
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


# RMSNorm -- Better, simpler alternative to LayerNorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale, self.eps = dim**-0.5, eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


# SwishGLU -- A Gated Linear Unit (GLU) with the Swish activation; always better than GELU MLP!
class SwishGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.act, self.project = nn.SiLU(), nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected, gate = self.project(x).tensor_split(2, dim=-1)
        return projected * self.act(gate)


# === Fundamental Transformer Building Blocks ===


class Attention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0) -> None:
        """Multi-Headed Self-Attention Operation"""
        super().__init__()
        assert embed_dim % n_heads == 0, "`embed_dim` must be divisible by `n_heads`!"
        self.n_heads, self.scale = n_heads, (embed_dim // n_heads) ** -0.5
        self.attn_softmax = None

        # Projections
        self.qkv, self.proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True), nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape

        # Project to Q-K-V
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Self-attention -- with masking!
        scores = q @ (k.transpose(-2, -1) * self.scale)
        if mask is not None:
            if mask.ndim == 2:
                mask = rearrange(mask, "bsz seq -> bsz 1 seq 1")
            elif mask.ndim != 4:
                raise NotImplementedError("Attention got `mask` of shape not in {2, 4}!")

            # Mask out by filling indices with negative infinity...
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        # Compute weighted sum over values
        self.attn_softmax = scores.softmax(dim=-1)
        vals = (self.attn_softmax @ v).transpose(1, 2).reshape(B, N, C)

        # Project back to `embed_dim` -- with optional dropout
        vals = self.dropout(self.proj(vals))
        return vals


class Block(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        do_rms_norm: bool = False,
        do_swish_glu: bool = False,
        do_layer_scale: bool = False,
    ) -> None:
        """
        Transformer Block Implementation (modality-agnostic).

        :param embed_dim: Core embedding/hidden dimension for vision transformer backbone.
        :param n_heads: Number of heads for multi-headed self-attention.
        :param mlp_ratio: Ratio for embedding size to position-wise feed-forward MLP (gets shrunk back down).
        :param dropout: [Optional] dropout for projection layer and MLPs -- for MAEs, always 0.0!
        :param do_rms_norm: Boolean whether or not to use RMSNorm in lieu of LayerNorm within block.
        :param do_swish_glu: Use the Swish-variant of the Gated Linear Unit for the feed-forward layers.
        :param do_layer_scale: Boolean whether or not to use LayerScale from Mistral/CaIT w/ initialization of 0.1.
        """
        super().__init__()
        self.embed_dim, self.n_heads, self.do_layer_scale = embed_dim, n_heads, do_layer_scale

        # Attention Components
        self.pre_norm_attn = RMSNorm(self.embed_dim) if do_rms_norm else nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.attn = Attention(self.embed_dim, n_heads=n_heads, dropout=dropout)
        if do_layer_scale:
            self.layer_scale_attn = LayerScale(self.embed_dim)

        # Position-wise Feed-Forward Components
        self.pre_norm_mlp = RMSNorm(self.embed_dim) if do_rms_norm else nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            # Handle SwishGLU vs. GELU MLP...
            (
                SwishGLU(embed_dim, int(mlp_ratio * embed_dim))
                if do_swish_glu
                else nn.Sequential(nn.Linear(embed_dim, int(mlp_ratio * embed_dim)), nn.GELU())
            ),
            nn.Dropout(dropout),
            nn.Linear(int(mlp_ratio * embed_dim), embed_dim),
        )
        if self.do_layer_scale:
            self.layer_scale_mlp = LayerScale(self.embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.do_layer_scale:
            x = x + self.layer_scale_attn(self.attn(self.pre_norm_attn(x), mask))
            x = x + self.layer_scale_mlp(self.mlp(self.pre_norm_mlp(x)))
        else:
            x = x + self.attn(self.pre_norm_attn(x), mask)
            x = x + self.mlp(self.pre_norm_mlp(x))
        return x



class MotionFormerDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model=192,
            d_ffn=768,
            dropout=0.1,
            n_heads=6,
            n_points=4,
            batch_first = True,
            use_text_cross_attention = False,
            use_det_query_attention = False
    ):
        super().__init__()

        self.use_text_cross_attention = use_text_cross_attention
        self.use_det_query_attention = use_det_query_attention
        # cross attention text
        if use_text_cross_attention:
            self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first= batch_first)
            self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.catext_norm = nn.LayerNorm(d_model)

        # Bi-direction Cross-attn between Det-query and Motion-query
        if use_det_query_attention:
            self.cross_attn_det_motion = BiAttentionBlock(v_dim = d_model, 
                                                          l_dim = d_model,
                                                          embed_dim = d_model,
                                                          num_heads = n_heads) 


        # cross attention
        self.attn_cross = MultiScaleDeformableAttention(embed_dim  = d_model,
                                                        num_heads  = n_heads,
                                                        num_levels = 1,
                                                        num_points = n_points,
                                                        batch_first= batch_first)
        self.dropout_cross = nn.Dropout(dropout)
        self.norm_cross = nn.LayerNorm(d_model)

        # inter-group self-attention
        self.attn_inter = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=batch_first)
        self.dropout_inter = nn.Dropout(dropout)
        self.norm_inter = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.GELU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, query):
        query2 = self.linear2(self.dropout3(self.activation(self.linear1(query))))
        query = query + self.dropout4(query2)
        query = self.norm3(query)
        return query

    def forward(
            self,
            query,
            pos_embed,
            src,
            reference_points,
            src_spatial_shapes,
            lang = None,
            lang_mask = None,
            det_query = None,
            shared_mlp = True,
    ):
        # self-attention
        q_inter = k_inter = self.with_pos_embed(query, pos_embed)
        query_inter = query
        query2_inter = self.attn_inter(q_inter, k_inter, query_inter)[0]
        query_inter = query_inter + self.dropout_inter(query2_inter)
        query_inter = self.norm_inter(query_inter)


        # language cross-attention
        if self.use_text_cross_attention:
            query_inter_2 = self.ca_text(
                self.with_pos_embed(query_inter, pos_embed),
                lang,
                lang,
                key_padding_mask=lang_mask,
            )[0]
            query_inter = query_inter + self.catext_dropout(query_inter_2)
            query_inter = self.catext_norm(query_inter)


        # visual cross-attention
        query2 = self.attn_cross(query = query_inter, key = src, value = src,
                                 reference_points = reference_points,
                                 spatial_shapes = src_spatial_shapes,
                                 level_start_index = torch.tensor(0).to(src.device))
        query_inter = query_inter + self.dropout_cross(query2)
        query = self.norm_cross(query_inter)


        if self.use_det_query_attention:
            query, det_query = self.cross_attn_det_motion(query, det_query)


        # ffn
        if self.use_det_query_attention and shared_mlp:
            concat_query = self.forward_ffn(torch.cat((query, det_query), dim=1))
            query = concat_query[:,:-1]
            det_query = concat_query[:,-1].unsqueeze(1)
        else:
            query = self.forward_ffn(query)

        return query, det_query

