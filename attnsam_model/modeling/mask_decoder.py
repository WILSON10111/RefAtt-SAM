# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from typing import List, Tuple, Type,Dict
from .module_lib import MLP, LayerNorm2d
from .transformer import TwoWayTransformer
from .common import LayerNorm2d




class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        attn_sim=None,
        target_embedding=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            attn_sim=attn_sim,
            target_embedding=target_embedding
        )

        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        attn_sim=None,
        target_embedding=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens, attn_sim, target_embedding)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


class MaskDecoder_ATTN(MaskDecoder):
    def __init__(self, model_type: str, sam_decoder_state_dict: Dict):
        super().__init__(transformer_dim=256,
                         transformer=TwoWayTransformer(
                             depth=2,
                             embedding_dim=256,
                             mlp_dim=2048,
                             num_heads=8,
                         ),
                         num_multimask_outputs=3,
                         activation=nn.GELU,
                         iou_head_depth=3,
                         iou_head_hidden_dim=256, )
        assert model_type in ["vit_b", "vit_l", "vit_h"]
        self.load_state_dict(sam_decoder_state_dict)
        # Freeze the parameters of the SAM decoder
        for n, p in self.named_parameters():
            p.requires_grad = False
        # Store the names of the frozen modules and parameters
        self.froze_modules = [n for n, _ in self.named_children()]
        self.froze_params = [n for n, _ in self.named_parameters()]

        transformer_dim = 256
        vit_dim_dict = {"vit_b": 768, "vit_l": 1024, "vit_h": 1280}
        vit_dim = vit_dim_dict[model_type]

        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1

        self.compress_vit_feat = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))

        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        )

        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))

    def train(self, mode: bool = True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training: turn the modules of original SAM mask decoder to eval mode
            for n, c in self.named_children():
                if n in self.froze_modules:
                    c.eval()
                else:
                    c.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool,
            interm_embeddings: torch.Tensor,
            rs_token_weight: torch.Tensor = None,
            return_all_hq_masks: bool = False,
            attn_sim = None,
            target_embedding=None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

        if isinstance(self.compress_vit_feat, List):
            rs_features = self.embedding_encoder(image_embeddings)
            for i in range(len(self.compress_vit_feat)):
                vit_features = interm_embeddings[i].permute(0, 3, 1, 2)
                rs_features += self.compress_vit_feat[i](vit_features)
        # for compatibility with the original SAM-HQ ckpt
        else:
            vit_features = interm_embeddings[0].permute(0, 3, 1, 2)  # early-layer ViT feature, after 1st global attention block in ViT
            rs_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)

        batch_size = len(image_embeddings)
        masks_sam_batch, masks_hq_batch = [], []
        for i_batch in range(batch_size):
            masks, iou_preds = self.predict_masks(
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                image_pe=image_pe[i_batch],
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch],
                hq_feature=rs_features[i_batch].unsqueeze(0),
                hq_token_weight=rs_token_weight,
                attn_sim=attn_sim,
                target_embedding=target_embedding
            )

            mask_slice = slice(0, 1)
            masks_sam = masks[:, mask_slice]

            masks_hq = masks[:, slice(self.num_mask_tokens - 1, self.num_mask_tokens), :, :]
            masks_sam_batch.append(masks_sam)
            masks_hq_batch.append(masks_hq)

        return masks_hq_batch,iou_preds,masks_sam_batch

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            hq_feature: torch.Tensor,
            hq_token_weight: torch.Tensor = None,
            attn_sim=None,
            target_embedding = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if hq_token_weight is None:
            hq_token_weight = self.hf_token.weight
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, hq_token_weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        if sparse_prompt_embeddings.dim() == 2:
            sparse_prompt_embeddings = sparse_prompt_embeddings.unsqueeze(1)
        elif sparse_prompt_embeddings.dim() == 4:
            sparse_prompt_embeddings = sparse_prompt_embeddings.squeeze(0)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens, attn_sim, target_embedding)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_ours = self.embedding_maskfeature(upscaled_embedding_sam) + hq_feature
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < 4:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        if hyper_in.size(0) == 0:
            return torch.empty(0, upscaled_embedding_sam.shape[1], upscaled_embedding_sam.shape[2], upscaled_embedding_sam.shape[3],
                               dtype=image_embeddings.dtype, device=image_embeddings.device), \
                torch.empty(0, dtype=image_embeddings.dtype, device=image_embeddings.device)

        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:, :4] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_ours = (hyper_in[:, 4:] @ upscaled_embedding_ours.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam, masks_ours], dim=1)

        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred