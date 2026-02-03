# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
import warnings
import einops
import torch
import torch.distributed as dist
import torch.nn as nn

from ..constants import IGNORE_INDEX, MODAL_INDEX_MAP, NUM_FRAMES
from .encoder import build_vision_encoder
from .projector import build_vision_projector, load_mm_projector
from .compressor import build_token_compressor


def spatial_downsampling(features, grid_thws, stride=2):
    n, c = features.shape

    flatten_grid_thws = torch.cat([grid_thw for batch_grid_thws in grid_thws for grid_thw in batch_grid_thws])
    split_sizes = [grid_thw.prod() for grid_thw in flatten_grid_thws]
    features = torch.split(features, split_sizes)

    new_features = []
    for feature, grid_thw in zip(features, flatten_grid_thws):
        # NOTE: adapted for reshape in image processor 
        feature = feature.view(grid_thw[0], grid_thw[1] // stride, grid_thw[2] // stride, stride, stride,  c).permute(0, 1, 3, 2, 4, 5)
        feature = feature.reshape(grid_thw[0], grid_thw[1], grid_thw[2], c).permute(0, 3, 1, 2)
        # NOTE: previous version model is align_corners=True
        new_feature = torch.nn.functional.interpolate(feature, (math.ceil(grid_thw[1] / stride), math.ceil(grid_thw[2] / stride)), mode='bilinear')
        # new_feature = nn.functional.avg_pool2d(feature, stride)
        # new_feature = nn.functional.max_pool2d(feature, stride)
        new_features.append(new_feature.permute(0, 2, 3, 1).view(-1, c))
    new_features = torch.cat(new_features)

    return new_features


class Videollama3MetaModel:

    def __init__(self, config):
        super(Videollama3MetaModel, self).__init__(config)

        if hasattr(config, "vision_encoder") or hasattr(config, "mm_vision_encoder"):
            self.vision_encoder = build_vision_encoder(config, delay_load=False)
            self.mm_projector = build_vision_projector(config, self.vision_encoder.hidden_size)
        if hasattr(config, "trainable_mm_compressor") and config.trainable_mm_compressor:
            self.token_compressor = build_token_compressor(config)
            
    def get_vision_encoder(self):
        vision_encoder = getattr(self, 'vision_encoder', None)
        if type(vision_encoder) is list:
            vision_encoder = vision_encoder[0]
        return vision_encoder

    def get_mm_projector(self):
        return self.mm_projector

    def get_token_compressor(self):
        compressor = getattr(self, 'token_compressor', None)
        return compressor
    
    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_encoder = model_args.vision_encoder
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature

        self.config.mm_vision_encoder = vision_encoder

        if self.get_vision_encoder() is None:
            vision_encoder = build_vision_encoder(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_encoder = [vision_encoder]
            else:
                self.vision_encoder = vision_encoder
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_encoder = self.vision_encoder[0]
            else:
                vision_encoder = self.vision_encoder
            # NOTE: only compatible with delay_load encoder
            # vision_encoder.load_model(vision_encoder.cfg_only)

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_encoder.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

class Videollama3MetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_encoder(self):
        return self.get_model().get_vision_encoder()

    def get_mm_projector(self):
        return self.get_model().get_mm_projector()
    def get_token_compressor(self):
        return self.get_model().get_token_compressor()

    def compress_visual_tokens_with_compressor(
        self, 
        vision_tokens: torch.FloatTensor, 
        compression_mask: torch.BoolTensor,
        position_ids: torch.LongTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        # compression_mask: [1, num_tokens]
        # vision_tokens: [1, num_tokens, dim]
        # position_ids: [1, num_tokens]
        device = vision_tokens.device
        if not compression_mask.any():
            keep_mask = torch.ones_like(compression_mask.view(-1), dtype=torch.bool, device=device)
            return vision_tokens, keep_mask

        need_compress_parts = vision_tokens[compression_mask] # [num_compress_tokens, dim]
        #
        compression_mask_tmp = torch.cat([
            torch.tensor([0], device=device), 
            compression_mask.view(-1).int(), 
            torch.tensor([0], device=device)], dim=0)
        diff = compression_mask_tmp[1:] - compression_mask_tmp[:-1]
        starts = (diff == 1).nonzero(as_tuple=False).squeeze(-1)
        ends = (diff == -1).nonzero(as_tuple=False).squeeze(-1)
        parts = ends - starts
        compression_cu_seqlens = torch.cat([torch.tensor([0], device=device), parts.cumsum(dim=0)], dim=0)
        # compressed vision tokens should have shape: [n, dim]
        compressed = self.get_model().get_token_compressor()(
            need_compress_parts,
            compression_cu_seqlens
        )
        vision_tokens_flat = vision_tokens.view(-1, vision_tokens.size(-1)) # shape: [num_tokens, dim]
        compression_mask_flat = compression_mask.view(-1)
        # start and end indices of each batch in the sample.
        position_ids_flat = position_ids.view(-1) # shape: [num_tokens]
        pos_start = torch.nonzero(position_ids_flat == 0, as_tuple=False).squeeze(-1).tolist()
        pos_end = pos_start[1:] + [position_ids_flat.numel()]
        rebuilt_tokens = []
        keeping_masks = []
        part_idx = 0
        for idx, (start, end) in enumerate(zip(pos_start, pos_end)):
            tokens = vision_tokens_flat[start:end]
            mask = compression_mask_flat[start:end]
            # If there is no token to be compressed, skip
            if not mask.any():
                rebuilt_tokens.append(tokens)
                keeping_masks.append(~mask)
                continue

            mask_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1) # indices of tokens to be compressed
            sample_part_start = part_idx
            while part_idx < starts.numel() and ends[part_idx].item() <= end:
                part_idx += 1
            sample_part_end = part_idx
            comp = compressed[sample_part_start:sample_part_end].reshape(-1, compressed.size(-1))
            if comp.numel() == 0:
                keep_mask = ~mask
                rebuilt_tokens.append(tokens[keep_mask])
                keeping_masks.append(keep_mask)
                continue

            k = min(comp.size(0), mask_idx.numel())  # compressed tokens kept for this sample
            keep_mask = ~mask # Token not been modified in this sample
            keep_mask[mask_idx[:k]] = True  # The first k places of compressed tokens should be replaced by compressed tokens.
            replace_mask = torch.zeros_like(keep_mask)
            replace_mask[mask_idx[:k]] = True
            tokens[replace_mask] = comp[:k]
            keeping_masks.append(keep_mask)
            rebuilt_tokens.append(tokens[keep_mask])     
        vision_tokens = torch.cat(rebuilt_tokens, dim=0)
        keeping_masks = torch.cat(keeping_masks, dim=0)
        vision_tokens = vision_tokens.to(device)
        keeping_masks = keeping_masks.to(device)
        if vision_tokens.dim() == 2 and vision_tokens.size(-1) == vision_tokens_flat.size(-1):
            vision_tokens = vision_tokens.unsqueeze(0)
        return vision_tokens, keeping_masks
    def encode_images(
        self,
        pixel_values: torch.FloatTensor,
        grid_sizes: torch.LongTensor,
        merge_sizes: torch.LongTensor,
        compression_mask: Optional[torch.BoolTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        mm_features = self.get_model().get_vision_encoder()(
            pixel_values=pixel_values,
            grid_sizes=grid_sizes,
            merge_sizes=merge_sizes,
        )
        keepmask = None
        if self.config.trainable_mm_compressor:
            assert compression_mask is not None, "compression_mask is required for trainable token compression."
            assert position_ids is not None, "position_ids is required for trainable token compression."
            mm_features, keepmask = self.compress_visual_tokens_with_compressor(
                mm_features,
                compression_mask,
                position_ids
            )
        mm_features = self.get_model().mm_projector(mm_features)
        return mm_features, keepmask
    
    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        compression_mask: Optional[torch.BoolTensor] = None,
    ):
        B, N = input_ids.shape
        if self.config.trainable_mm_compressor:
            assert position_ids is not None, "Currently model only supports position_ids and flatten input."
            assert compression_mask is not None, "compression_mask is required for trainable token compression."
            assert B == 1, "Currently model only supports batch size 1 for trainable token compression."
        vision_encoder = self.get_vision_encoder()
        # NOTE: text-only situation
        if vision_encoder is None or pixel_values is None or input_ids.shape[1] == 1:
            return input_ids, attention_mask, position_ids, past_key_values, None, labels
        # 1. flatten text inputs
        
        input_ids = input_ids.view(B * N)
        if attention_mask is not None:
            attention_mask = attention_mask.view(B * N)
        if position_ids is not None:
            position_ids = position_ids.view(B * N)
        if labels is not None:
            labels = labels.view(B * N)

        # 2. embed visual tokens and compress if needed
        batched_num_patches = grid_sizes.prod(dim=1).div(merge_sizes ** 2).long()
        mm_features, keepmask = self.encode_images(
            pixel_values, grid_sizes, merge_sizes, compression_mask, position_ids
        )
        # modify input_ids, position_ids, attention_mask, labels accordingly
        if keepmask is not None:
            if position_ids is not None:
                position_ids = position_ids[keepmask]
            if attention_mask is not None:
                attention_mask = attention_mask[keepmask]
            if labels is not None:
                labels = labels[keepmask]
            input_ids = input_ids[keepmask]
            
        # 3. embed text tokens
        inputs_embeds = self.get_model().embed_tokens(input_ids).clone()

        # 4. replace multimodal tokens with features
        image_selected = (input_ids == self.config.image_token_index)
        inputs_embeds[image_selected] = inputs_embeds[image_selected] * 0.0 + mm_features   

        # 5. reshape back to batched format
        C = inputs_embeds.shape[-1]
        inputs_embeds = inputs_embeds.reshape(B, -1, C)
        if attention_mask is not None:
            attention_mask = attention_mask.view(B, -1)
        if labels is not None:
            labels = labels.view(B, -1)
        if position_ids is not None:
            position_ids = position_ids.view(B, -1)

        return None, attention_mask, position_ids, past_key_values, inputs_embeds, labels
