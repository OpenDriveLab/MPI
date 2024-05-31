from typing import Callable, List, Optional, Tuple, Union
import random
import os
import torch
import torch.nn as nn
import numpy as np
import transformers
from einops import rearrange, repeat
import torch.nn.functional as F

from mpi.models.util.optimization import get_lr_update
from mpi.models.util.transformer import Block, PatchEmbed, RMSNorm, get_2D_position_embeddings, \
                                            MLP, MotionFormerDecoderLayer, ProjectionHead
from mpi.models.util.ms_deform_attn import MultiScaleDeformableAttention
from mpi.models.util.loss_utils import flow_warp, robust_l1, SSIM, get_smooth_loss, xywh2xyxy, get_patches_covered
from mpi.models.util.extraction import TokenAggregation
from torchvision.ops import generalized_box_iou_loss

import torchvision.transforms.functional as transform
# Suppress Transformers Logging
transformers.logging.set_verbosity_error()


class MPI(nn.Module):
    def __init__(
        self,
        resolution: int,
        patch_size: int,
        encoder_depth: int,
        encoder_embed_dim: int,
        encoder_n_heads: int,
        decoder_depth: int,
        decoder_embed_dim: int,
        decoder_n_heads: int,
        language_model: str,
        hf_cache: str,
        language_dim: int,
        optimizer: str,
        schedule: str,
        base_lr: float,
        min_lr: float,
        effective_bsz: int,
        betas: Tuple[float, float],
        weight_decay: float,
        warmup_epochs: int,
        max_epochs: int,
        mask_ratio: float = 0.75,
        mlp_ratio: float = 4.0,
        in_channels: int = 3,
        norm_pixel_loss: bool = True,
        use_cls_token: bool = False,
    ) -> None:
        """
        Initialize a MPI model with the requisite architecture parameters.

        :param resolution: Base image resolution -- usually 224 (ImageNet size).
        :param patch_size: Height/Width of each patch in pixels -- usually 16.
        :param encoder_depth: Number of Transformer blocks in the encoder -- should be greater than decoder.
        :param encoder_embed_dim: Core embedding/hidden dimension for encoder vision transformer backbone.
        :param encoder_n_heads: Number of heads for encoder multi-headed self-attention.
        :param decoder_depth: Number of Transformer blocks in the decoder -- should be relatively shallow.
        :param decoder_embed_dim: Core embedding/hidden dimension for encoder vision transformer backbone.
        :param decoder_n_heads: Number of heads for encoder multi-headed self-attention.
        :param language_model: Language model to freeze for encoding narrations/utterances.
        :param hf_cache: Cache directory to store pretrained models, for safe distributed training.
        :param language_dim: Dimensionality of the language embedding coming out of the pretrained LM.
        :param optimizer: String denoting which optimizer to use (for MAEs, usually `adamw`)
        :param schedule: Learning rate schedule to use; for Transformers a linear warmup + decay is recommended!
        :param base_lr: Base learning rate, to be scaled via a linear scaling rule (from scaling laws).
        :param min_lr: Minimum learning rate to decay to over the course of learning (usually 0.0)
        :param effective_bsz: Global batch size for update, dictates the scaling of the base_lr.
        :param betas: Adam optimizer betas (only applicable for `adam` and `adamw`. Prevents early loss spiking.
        :param weight_decay: Weight decay for global weight regularization (only applied to non-bias, non-LN layers).
        :param warmup_epochs: Number of epochs to warmup learning rate for linear warmup schedule.
        :param max_epochs: Total number of training epochs to be run.
        :param mask_ratio: Ratio for number of patches to mask out for MAE -- should be fairly high!
        :param mlp_ratio: Ratio for embedding size to Position-wise FeedForward MLP (gets shrunk back down).
        :param in_channels: Default number of channels in the base image -- almost always 3.
        :param norm_pixel_loss: Normalize decoder pixel targets for reconstruction (better perf, not interpretable).
        :param use_cls_token: Add <CLS> token for continued pretraining (NOTE: not used in MAE pretraining/finetuning!)
        """
        super().__init__()
        self.resolution, self.patch_size, self.mask_ratio = resolution, patch_size, mask_ratio
        self.in_channels, self.norm_pixel_loss, self.mlp_ratio = in_channels, norm_pixel_loss, mlp_ratio
        self.optimizer, self.schedule, self.betas, self.weight_decay = optimizer, schedule, betas, weight_decay
        self.lr, self.base_lr, self.min_lr, self.effective_bsz = None, base_lr, min_lr, effective_bsz
        self.warmup_epochs, self.max_epochs = warmup_epochs, max_epochs
        self.use_cls_token = use_cls_token
        self.language_dim = language_dim
        # Encoder/Decoder Parameters
        self.encoder_depth, self.decoder_depth = encoder_depth, decoder_depth
        self.encoder_embed_dim, self.encoder_n_heads = encoder_embed_dim, encoder_n_heads
        self.decoder_embed_dim, self.decoder_n_heads = decoder_embed_dim, decoder_n_heads

        # General Parameters (for downstream adaptation)
        self.embed_dim, self.n_heads = self.encoder_embed_dim, self.encoder_n_heads
        # (Optional) <CLS> Token Handling
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))

        # Formulate reference boxes for deformable cross attention
        grid_x, grid_y = torch.meshgrid(torch.linspace(self.patch_size // 2 - 1
                                        , self.resolution - self.patch_size // 2 - 1
                                        , self.resolution // self.patch_size),
                                        torch.linspace(self.patch_size // 2 - 1
                                        , self.resolution - self.patch_size // 2 - 1
                                        , self.resolution // self.patch_size),
                                        indexing = 'xy')

        grid = torch.stack((grid_x, grid_y), dim=-1)
        self.reference_boxes = torch.cat((grid, self.patch_size * torch.ones((grid.shape[0], grid.shape[1], 2))), dim=-1).unsqueeze(0) / self.resolution


        self.img_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))
        self.lang_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))
        self.patch2embed = PatchEmbed(
            self.resolution, self.patch_size, self.encoder_embed_dim, in_channels=self.in_channels
        )
        self.encoder_pe = nn.Parameter(
            torch.zeros(1, self.patch2embed.num_patches + (1 if self.use_cls_token else 0), self.encoder_embed_dim),
            requires_grad=False,
        )
        self.encoder_blocks = nn.ModuleList(
            [   
                Block(
                    self.encoder_embed_dim,
                    self.encoder_n_heads,
                    self.mlp_ratio,
                    do_rms_norm=True,
                    do_swish_glu=True,
                    do_layer_scale=True,
                ) for _ in range(self.encoder_depth)
            ] 
        )
        self.encoder_norm = RMSNorm(self.encoder_embed_dim)

        # Temporal attention to fuse key frames
        self.temporal_attn = MultiScaleDeformableAttention( embed_dim  = self.encoder_embed_dim,
                                                            num_heads  = self.encoder_n_heads,
                                                            num_levels = 1,
                                                            num_points = 4,
                                                            batch_first= True)
        self.temporal_attn_norm = nn.LayerNorm(self.encoder_embed_dim)


        # Token aggregator
        self.token_aggregate = TokenAggregation(n_latents = 1, embed_dim = self.encoder_embed_dim, n_heads = self.encoder_n_heads)
        self.lang_latents = nn.Parameter(torch.zeros(1, self.encoder_embed_dim))
        self.vis_latents = nn.Parameter(torch.zeros(1, self.encoder_embed_dim))
        

        ### Encoder Ended


        # Projection from Encoder to Decoder
        self.encoder2decoder_vis = nn.Linear(self.encoder_embed_dim, self.decoder_embed_dim)
        self.encoder2decoder_lang = nn.Linear(self.encoder_embed_dim, self.decoder_embed_dim)
        self.encoder2decoder_ctx = nn.Linear(self.encoder_embed_dim, self.decoder_embed_dim)
        self.encoder2decoder_det_query = nn.Linear(self.encoder_embed_dim, self.decoder_embed_dim)

        # Initialize motion embedding (of same size with visual patches embedding)
        self.reconstruction_embedding = nn.Embedding( (self.resolution // self.patch_size) ** 2, self.decoder_embed_dim)

        self.decoder_pe = nn.Parameter(
            torch.zeros(1, self.patch2embed.num_patches + (1 if self.use_cls_token else 0), self.decoder_embed_dim),
            requires_grad=False,
        )
        
        self.use_text_cross_attention = True
        self.use_det_query_attention =  True
        self.MotionFormer = nn.ModuleList(
            [
                MotionFormerDecoderLayer(
                    d_model = self.decoder_embed_dim,
                    d_ffn = int(self.decoder_embed_dim * self.mlp_ratio),
                    dropout = 0.,
                    n_heads = self.decoder_n_heads,
                    use_text_cross_attention = self.use_text_cross_attention,
                    use_det_query_attention = self.use_det_query_attention # True if self.add_box_loss else False
                )
                for _ in range(self.decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(self.decoder_embed_dim)
        
        # Reconstruction head
        self.reconstruction_prediction = nn.Linear(self.decoder_embed_dim, (patch_size**2) * in_channels)

        # Bounding box detector head
        self.bbox_regression = nn.Linear(self.encoder_embed_dim, 4)
        self.bbox_regression_decoder = nn.Linear(self.decoder_embed_dim, 4)
        
        # Projection for contrastive loss computation
        self.image_projection = nn.Linear(self.encoder_embed_dim, 128)
        self.lang_projection =  nn.Linear(self.encoder_embed_dim, 128)
        self.logit_scale = nn.Parameter(torch.tensor([np.log(1/0.07)]).double())
        


        self.ctx_enc_pe = nn.Parameter(torch.randn(1, 3, 1, self.encoder_embed_dim))


        # Initialize all Weights
        self.initialize_weights()

        # @AFTER INITIALIZATION -- Create Language Model & Language Reward MLP --> LM has requires_grad = False
        #   > For BERT models, our "embedding" is just going to be the last hidden state
        #   > Assumes inputs to forward pass are pre-tokenized!
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased', cache_dir=hf_cache)
        self.lm = transformers.AutoModel.from_pretrained('distilbert/distilbert-base-uncased', cache_dir=hf_cache)
        self.lm.eval()
        self.lang2encoder = nn.Linear(self.language_dim, self.encoder_embed_dim) # Projection to align dimensions

        # Shape Assertion -- make sure self.language_dim actually is the same as the LM dimension!
        assert self.lm.config.dim == self.language_dim, "Language model embedding dimension != self.language_dim!"

        # Freeze the LM
        for _, param in self.lm.named_parameters():
            param.requires_grad = False

    def initialize_weights(self) -> None:
        # Position Encoding -- Fixed 2D Sine-Cosine Embeddings
        enc_pe = get_2D_position_embeddings(
            self.encoder_embed_dim, int(self.patch2embed.num_patches**0.5), cls_token=self.use_cls_token
        )
        self.encoder_pe.data.copy_(torch.from_numpy(enc_pe).float().unsqueeze(0))
        dec_pe = get_2D_position_embeddings(
            self.decoder_embed_dim, int(self.patch2embed.num_patches**0.5), cls_token=self.use_cls_token
        )
        self.decoder_pe.data.copy_(torch.from_numpy(dec_pe).float().unsqueeze(0))

        # Initialize PatchEmbedding as a Linear...
        nn.init.xavier_uniform_(self.patch2embed.proj.weight.data.view([self.patch2embed.proj.weight.data.shape[0], -1]))

        # Initialize Tokens
        nn.init.normal_(self.img_token, std=0.02)
        nn.init.normal_(self.lang_token, std=0.02)
        nn.init.normal_(self.vis_latents, std=0.02)
        nn.init.normal_(self.lang_latents, std=0.02)

        if self.use_cls_token:
            nn.init.normal_(self.cls_token, std=0.02)

        # Everything else...
        self.apply(self.transformer_initializer)

    @staticmethod
    def transformer_initializer(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # Use xavier_uniform following Jax ViT
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


    def encode_language(self, lang: torch.Tensor, lang_mask: torch.Tensor) -> torch.Tensor:
        """Encode language by feeding the *pre-tokenized text* through the frozen language model."""
        self.lm.eval()
        with torch.no_grad():
            transformer_embeddings = self.lm(lang, attention_mask=lang_mask).last_hidden_state
        return transformer_embeddings

    
    def get_representations(
        self, imgs: torch.Tensor, language: Optional[Union[List[str], Tuple[str]]] = None, mode: str = "all"
    ) -> torch.Tensor:
        """
        Given either a singleton (dual-imgs, language) pair or batch of dual-imgs and language, extract representations
        subject to the specified mode in < multimodal | visual >.

        :param imgs: Processed batch of images :: [bsz, 2, 3, 224, 224]
        :param language: Input language as `List[str] | Tuple[str] | None`
        :param mode: Type of representations to extract -- `multimodal` (both vision+text), `visual` (visual only)

        :return: Extracted representations given (imgs, language) input as sequence.
        """
        assert (
            imgs.ndim == 5
            and imgs.shape[1] == 2
            and (language is None or isinstance(language, list) or isinstance(language, tuple))
        ), "Invalid input to `get_representations()`"
        assert mode in {'all', 'agg_only'}, f"Extraction mode `{mode}` not supported!"

        # Tokenize Language --> note max length is 20!
        if language is None:
            lang, lang_mask = [torch.zeros(imgs.shape[0], 20, dtype=int, device=self.lm.device) for _ in range(2)]
            lang[:, 0], lang_mask[:, 0] = self.tokenizer.cls_token_id, 1
        else:
            tokens = self.tokenizer(language, return_tensors="pt", max_length=20, padding="max_length", truncation=True)
            lang, lang_mask = tokens["input_ids"].to(self.lm.device), tokens["attention_mask"].to(self.lm.device)

            # Tile Language & Language Mask if mismatch with # images!
            if not len(lang) == len(imgs):
                lang = repeat(lang, "b seq -> (bsz b) seq", bsz=imgs.size(0))
                lang_mask = repeat(lang_mask, "b seq -> (bsz b) seq", bsz=imgs.size(0))

        # Extract desired representations...
        representations = self.encode(imgs, lang, lang_mask, mode)
        return representations 


    def encode(self, imgs: torch.Tensor, lang: torch.Tensor, lang_mask: torch.Tensor, mode = 'all') -> torch.Tensor:
        """Default representation extraction function, given a batch of dual-images and tokenized language."""
        lang_embeddings = self.encode_language(lang, lang_mask)
        projected_lang = self.lang2encoder(lang_embeddings)

        # Patchify, broadcast position embedding across ctx_len (0 + K) dimension, unfold, add `ctx_enc_pe` embeddings!
        patches = self.patch2embed(rearrange(imgs, "bsz ctx channels res1 res2 -> (bsz ctx) channels res1 res2"))
        patches_pe = patches + self.encoder_pe
        ctx_patches = rearrange(patches_pe, "(bsz ctx) seq embed -> bsz ctx seq embed", ctx=2)
        b, c, seq, d = ctx_patches.shape


        # Add context embedding to differentiate
        ctx_patches = ctx_patches + torch.index_select(self.ctx_enc_pe, 1, torch.tensor([0, 0]).to(patches.device))
        visible_patches = rearrange(ctx_patches, "bsz ctx seq embed -> bsz (seq ctx) embed", ctx=2)
    
                           
        # Add "modality" embeddings to patches & language & flatten out context patches...
        visible_patches, lang = visible_patches + self.img_token, projected_lang + self.lang_token

        # Create "dummy" visible mask
        visible_mask = torch.ones_like(visible_patches[..., -1], dtype=lang_mask.dtype)
        for idx, block in enumerate(self.encoder_blocks):
            visible_patches = block(visible_patches, visible_mask)
        visible_patches = self.encoder_norm(visible_patches)


        # Temporal Attention
        reference_points = self.reference_boxes.to(ctx_patches.device)\
                           .flatten(1,2).unsqueeze(2).repeat(ctx_patches.shape[0], 1, 1, 1) # (bsz, num_queries=14*14, num_levels=1, 4)
        spatial_shapes = torch.tensor(self.reference_boxes.shape[1:3]).unsqueeze(0).to(ctx_patches.device) # (num_levels=1, 2)
        
        visual_embed = rearrange(visible_patches, "bsz (seq ctx) embed -> bsz ctx seq embed", ctx=2)
        fused_visual_embed = self.temporal_attn(query = visual_embed[:,0], key = visual_embed[:,1], value = visual_embed[:,1],
                                                reference_points = reference_points,
                                                spatial_shapes = spatial_shapes,
                                                level_start_index = torch.tensor(0).to(ctx_patches.device))
        fused_visual_embed = self.temporal_attn_norm( visual_embed[:,0] + fused_visual_embed )


        # Token aggregator
        aggregated_embedding_vis  = self.token_aggregate(fused_visual_embed, self.vis_latents)
        aggregated_embedding_lang = self.token_aggregate(lang, self.lang_latents)


        if mode == 'all':
            return torch.cat([aggregated_embedding_vis.unsqueeze(1), aggregated_embedding_lang.unsqueeze(1), fused_visual_embed, lang], dim=1)
        elif mode =='agg_only':
            return torch.cat([aggregated_embedding_vis, aggregated_embedding_lang], dim=-1)


    def forward_encoder(
        self, img_ctx: torch.Tensor, lang: torch.Tensor, lang_mask: torch.Tensor, task_flag
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Encode language
        lang_embeddings = self.encode_language(lang, lang_mask)
        projected_lang = self.lang2encoder(lang_embeddings)

        # Patchify, broadcast position embedding 
        patches = self.patch2embed(rearrange(img_ctx, "bsz ctx channels res1 res2 -> (bsz ctx) channels res1 res2"))
        patches_pe = patches + self.encoder_pe
        ctx_patches = rearrange(patches_pe, "(bsz ctx) seq embed -> bsz ctx seq embed", ctx=2)
        b, c, seq, d = ctx_patches.shape

        # Prepare context embeddings to denote key frames
        index = torch.tensor([[0, 1] if flag else [0, 2] for flag in task_flag]).to(patches.device)
        ctx_embed = []
        for idx in index:
            pre_img_embed = torch.index_select(self.ctx_enc_pe, 1, idx)
            ctx_embed.append(pre_img_embed)
        ctx_embed = torch.cat(ctx_embed, dim=0)

        # Add context embedding to differentiate
        ctx_patches = ctx_patches + ctx_embed
        visible_patches = rearrange(ctx_patches, "bsz ctx seq embed -> bsz (seq ctx) embed", ctx=2)
        visible_patches, lang = visible_patches + self.img_token, projected_lang + self.lang_token

        # Encode images
        visible_mask = torch.ones_like(visible_patches[..., -1], dtype=lang_mask.dtype)
        for idx, block in enumerate(self.encoder_blocks):
            visible_patches = block(visible_patches, visible_mask)
        visible_patches = self.encoder_norm(visible_patches)

        # Temporal Attention
        reference_points = self.reference_boxes.to(ctx_patches.device)\
                           .flatten(1,2).unsqueeze(2).repeat(ctx_patches.shape[0], 1, 1, 1) # (bsz, num_queries=14*14, num_levels=1, 4)
        spatial_shapes = torch.tensor(self.reference_boxes.shape[1:3]).unsqueeze(0).to(ctx_patches.device) # (num_levels=1, 2)

        visual_embed = rearrange(visible_patches, "bsz (seq ctx) embed -> bsz ctx seq embed", ctx=2)
        fused_visual_embed = self.temporal_attn(query = visual_embed[:,0], key = visual_embed[:,1], value = visual_embed[:,1],
                                            reference_points = reference_points,
                                            spatial_shapes = spatial_shapes,
                                            level_start_index = torch.tensor(0).to(ctx_patches.device))
        fused_visual_embed = self.temporal_attn_norm( visual_embed[:,0] + fused_visual_embed )


        aggregated_embedding_vis  = self.token_aggregate(fused_visual_embed, self.vis_latents)
        aggregated_embedding_lang = self.token_aggregate(lang, self.lang_latents)


        ### Encoder ended


        # Context preparation
        projected_patches = self.encoder2decoder_vis(fused_visual_embed)
        if self.use_text_cross_attention:
            projected_lang = self.encoder2decoder_lang(lang)    # Decoupled Projection
        else:
            projected_lang = None

        # Initialize reconstruction quereis
        decode_ctx_embed = []
        for flag in task_flag:
            idx = 2 if flag else 1
            decode_ctx_embed.append(self.ctx_enc_pe[:,idx])
        decode_ctx_embed = torch.cat(decode_ctx_embed, dim=0)

        reconstruction_queries = self.reconstruction_embedding.weight[None,:,:].repeat(projected_patches.shape[0], 1, 1)
        reconstruction_queries = reconstruction_queries + self.encoder2decoder_ctx(decode_ctx_embed)

        # Use aggregated encoder visual tokens to initialize detection query
        det_query = self.encoder2decoder_det_query(aggregated_embedding_vis).unsqueeze(1)

        det_query_per_layer = []
        for idx, layer in enumerate(self.MotionFormer):
            reconstruction_queries, det_query = layer(
                                query = reconstruction_queries,
                                pos_embed = self.decoder_pe,   # Adaptive PE Scaling
                                src = projected_patches,
                                reference_points = reference_points,
                                src_spatial_shapes = spatial_shapes,
                                lang = projected_lang,
                                lang_mask = (torch.ones_like(lang_mask) - lang_mask).bool(),
                                det_query = det_query
                            )
            det_query_per_layer.append(det_query)
        reconstruction_queries = self.decoder_norm(reconstruction_queries)

        return reconstruction_queries, aggregated_embedding_vis, aggregated_embedding_lang, det_query_per_layer


    def forward_decoder(self, reconstruction_queries, aggregated_embedding, det_query_per_layer) -> torch.Tensor:
        
        # Reconstruction learning: How-to-interact
        reconstructions = self.reconstruction_prediction(reconstruction_queries)

        # Detection learning: Where-to-interact
        box_pred = []
        bbox_encoder = F.sigmoid(self.bbox_regression(aggregated_embedding))
        box_pred.append(bbox_encoder)
        for i in range(self.decoder_depth):
            bbox_decoder = F.sigmoid(self.bbox_regression_decoder(det_query_per_layer[i]).squeeze())
            box_pred.append(bbox_decoder)
        box_pred = torch.cat(box_pred, dim=0)


        return reconstructions, box_pred


    def forward(
        self, imgs: torch.Tensor, target_img: torch.Tensor, lang: torch.Tensor, lang_mask: torch.Tensor, box_target: torch.Tensor, task_flag
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Run a forward pass through the model, computing the language-conditioned MAE reconstruction loss on the
        0th + Kth frame temporal context, given language prefix.

        :param imgs: A [bsz, 2, in_channels, resolution, resolution] tensor of (0th frame, Kth frame) sequences.
        :param lang: A [bsz, seq_len] tensor of language context to condition on.
        :param lang_mask: A [bsz, seq_len] binary mask tensor to indicate padding locations in the lang tensor.
        :param mask_ratio: Optional masking ratio to use instead of the default.

        :return Tuple of losses and intermediates, as follows:
            > (combined loss, [reconstruction loss per frame in {0, K}])
        """
        device = imgs.device
        reconstruction_queries, \
        aggregated_embedding_vis, \
        aggregated_embedding_lang, \
        det_query_per_layer = self.forward_encoder(imgs, lang.squeeze(), lang_mask.squeeze(), task_flag)
        ctx_reconstructions, box_regression = self.forward_decoder(reconstruction_queries, aggregated_embedding_vis, det_query_per_layer)

        # loss computation
        rec_loss = self.compute_reconstruction_loss(target_img, ctx_reconstructions)
        box_loss = self.box_regression_loss(box_regression, box_target)
        contra_loss = self.compute_contrastive_loss(aggregated_embedding_vis, aggregated_embedding_lang)

        loss = rec_loss + box_loss + contra_loss

        return loss, rec_loss, box_loss, contra_loss

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert a batch of (0th + Kth frame) images to their patched equivalents by naive reshaping."""
        return rearrange(
            imgs,
            "bsz ctx c (height patch_h) (width patch_w) -> (bsz ctx) (height width) (patch_h patch_w c)",
            patch_h=self.patch_size,
            patch_w=self.patch_size,
        )

    
    def inv_patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert a batch of (0th + Kth frame) images to their patched equivalents by naive reshaping."""
        return rearrange(
            imgs,
            "bsz (height width) (patch_h patch_w c) -> bsz c (height patch_h) (width patch_w) ",
            patch_h=self.patch_size,
            patch_w=self.patch_size,
            height = self.resolution // self.patch_size
        )


    def compute_reconstruction_loss(
        self, imgs: torch.Tensor, ctx_reconstructions: torch.Tensor, 
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.norm_pixel_loss, "`norm_pixel_loss` should always be true... false only for visualizations!"

        targets = self.patchify(imgs)

        # # Normalize targets...
        # if self.norm_pix_loss:
        mu, var = targets.mean(dim=-1, keepdim=True), targets.var(dim=-1, unbiased=True, keepdim=True)
        targets = (targets - mu) / ((var + 1e-6) ** 0.5)

        
        zero_mse = (ctx_reconstructions - targets) ** 2
        reconstruction_loss = zero_mse.mean()

        # Averaged by spatial dimension and batch size
        return reconstruction_loss

    def compute_contrastive_loss(self, aggregated_embedding_vis, aggregated_embedding_lang):
        image_embeddings = self.image_projection(aggregated_embedding_vis) 
        text_embeddings = self.lang_projection(aggregated_embedding_lang)

        # Normalize Features
        image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_embeddings  = text_embeddings  / text_embeddings.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp().to(torch.float32)
        logits = logit_scale * text_embeddings @ image_embeddings.T 
        labels = torch.arange(logits.shape[0]).to(logits.device)

        loss_t = F.cross_entropy(logits, labels, reduction='none')
        loss_i = F.cross_entropy(logits.T, labels, reduction='none')
        loss =  (loss_t + loss_i) / 2.0

        return loss.mean()


    def box_regression_loss(self, pred: torch.Tensor, target: torch.Tensor):
        
        target = target.repeat(self.decoder_depth + 1, 1)
        giou_loss = generalized_box_iou_loss(xywh2xyxy(pred).float(), xywh2xyxy(target).float(), reduction='mean')
        l1_loss = F.l1_loss(pred, target, reduction='mean')

        return giou_loss + l1_loss

    
    def configure_optimizer(self) -> Tuple[torch.optim.Optimizer, Callable[[int, float], float]]:
        # Short-Circuit on Valid Optimizers
        if self.optimizer not in ["adamw"]:
            raise NotImplementedError(f"Optimizer `{self.optimizer}` not supported - try [`adamw`] instead!")

        # Create Parameter Groups --> Bias terms, Normalization layer parameters shouldn't be decayed...
        #   > This is a compact rewrite of `param_groups_weight_decay()` from TIMM because I don't want the dependency
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Check on any parameters with fewer than 2 dimensions or with "bias" in the name...
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)

        # Build Parameter Groups
        groups = [{"params": decay, "weight_decay": self.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

        # Compute LR -- MAE uses the `linear scaling rule` :: lr = base_lr * (effective_bsz / 256)
        #   > https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md
        self.lr = self.base_lr * (self.effective_bsz / 256)

        # Create Optimizer & LR Scheduler
        optimizer = torch.optim.AdamW(groups, lr=self.lr, betas=self.betas)
        update_lr = get_lr_update(optimizer, self.schedule, self.lr, self.min_lr, self.warmup_epochs, self.max_epochs)
        return optimizer, update_lr
