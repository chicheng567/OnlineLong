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
import torch.nn.functional as F

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
        compression_parts: List[List[int]],
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        # compression_parts: [[start, end], [start, end], ...]
        # vision_tokens: [1, num_tokens, dim]
        device = vision_tokens.device
        vision_tokens = vision_tokens.squeeze(0) # [num_tokens, dim]
        compression_cu_seqlens = [0]
        need_compress_parts = torch.zeros(vision_tokens.shape[0], device=device, dtype=torch.bool)
        replace_mask = torch.zeros(vision_tokens.shape[0], device=device, dtype=torch.bool)
        compressor_output_length = self.get_token_compressor().compress_image_wh
        for part in compression_parts:
            part_len = part[1] - part[0]
            need_compress_parts[part[0]: part[1]] = True
            replace_mask[part[0]: part[0] + compressor_output_length] = True
            compression_cu_seqlens.append(compression_cu_seqlens[-1] + part_len)
        compression_cu_seqlens = torch.tensor(compression_cu_seqlens, device=device, dtype=torch.long)
        
        # compressed vision tokens should have shape: [n, dim]
        original_tokens_to_reconstruct = vision_tokens[need_compress_parts]
        compressed = self.get_token_compressor()(
            original_tokens_to_reconstruct,
            compression_cu_seqlens
        )
        reconstruction_mse_loss = None
        if getattr(self.get_token_compressor(), "compression_decoder", None) is not None:
            reconstructed = self.get_token_compressor().decode_tokens(compressed)
            reconstruction_mse_loss = F.mse_loss(reconstructed, original_tokens_to_reconstruct, reduce="sum")
            reconstruction_mse_loss /= compression_cu_seqlens.shape[0] - 1 # average mse loss per sample
        keeping_masks = ~need_compress_parts | replace_mask
        vision_tokens[replace_mask] = compressed.view(-1, vision_tokens.shape[-1])
        vision_tokens = vision_tokens[keeping_masks]
        return vision_tokens, reconstruction_mse_loss
    def encode_images(
        self,
        pixel_values: torch.FloatTensor,
        grid_sizes: torch.LongTensor,
        merge_sizes: torch.LongTensor,
        compression_parts: Optional[List[List[int]]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        reconstruction_mse_loss = None
        mm_features = self.get_model().get_vision_encoder()(
            pixel_values=pixel_values,
            grid_sizes=grid_sizes,
            merge_sizes=merge_sizes,
        )
        if self.config.trainable_mm_compressor and compression_parts is not None and len(compression_parts) > 0:
            assert compression_parts is not None, "compression_parts is required for trainable token compression."
            mm_features, reconstruction_mse_loss = self.compress_visual_tokens_with_compressor(
                mm_features,
                compression_parts,
            )
        mm_features = self.get_model().mm_projector(mm_features)
        return mm_features, reconstruction_mse_loss
    
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
        compression_parts: Optional[List[List[int]]] = None,
    ):
        B, N = input_ids.shape
        device = input_ids.device
        if self.config.trainable_mm_compressor:
            assert position_ids is not None, "Currently model only supports position_ids and flatten input."
            # Compression parts should like: [[1, 3], [4, 10], [16, 20]],  where each part indicates the start and end position of vision tokens to be compressed.
            assert compression_parts is not None, "compression_parts is required for trainable token compression."
            assert B == 1, "Currently model only supports batch size 1 for trainable token compression."
        vision_encoder = self.get_vision_encoder()
        # NOTE: text-only situation
        if vision_encoder is None or pixel_values is None or input_ids.shape[1] == 1:
            return input_ids, attention_mask, position_ids, past_key_values, None, labels, None
        # 1. flatten text inputs
        input_ids = input_ids.view(B * N)
        if attention_mask is not None:
            attention_mask = attention_mask.view(B * N)
        if position_ids is not None:
            position_ids = position_ids.view(B * N)
        if labels is not None:
            labels = labels.view(B * N)

        # 2. embed visual tokens and compress if needed
        image_selected = (input_ids == self.config.image_token_index)
        image_positions = torch.nonzero(image_selected, as_tuple=False).squeeze(-1) # vision token's positions among all tokens
        mm_features, reconstruction_mse_loss = self.encode_images(
            pixel_values, grid_sizes, merge_sizes, compression_parts
        )
        
        if compression_parts is not None and len(compression_parts) > 0:
            compact_vision_token_size = self.get_token_compressor().compress_image_wh
            replace_mask = torch.zeros(input_ids.shape[0], device=device, dtype=torch.bool)
            keep_mask = torch.ones(input_ids.shape[0], device=device, dtype=torch.bool)
            compression_starts = torch.zeros(input_ids.shape[0], device=device, dtype=torch.bool)
            compression_ends = torch.zeros(input_ids.shape[0], device=device, dtype=torch.bool)
            for part in compression_parts:
                part_start = image_positions[part[0]]
                part_end = image_positions[part[1] - 1]
                compression_starts[part_start] = True
                replace_mask[part_start + 1: part_start + compact_vision_token_size + 1] = True
                compression_ends[part_start + compact_vision_token_size + 1] = True
                keep_mask[part_start + compact_vision_token_size + 2: part_end + 1] = False
            input_ids[replace_mask] = self.config.image_token_index
            input_ids[compression_starts] = self.config.compression_start_token_id
            input_ids[compression_ends] = self.config.compression_end_token_id
            input_ids = input_ids[keep_mask]
            if attention_mask is not None:
                attention_mask = attention_mask[keep_mask]
            if position_ids is not None:
                start = position_ids == 0
                #NOTE: First token of sample will never be compressed because of BOS and system prompt, so we can use it to determine the start of each sample.
                start = start[keep_mask]
                start = torch.nonzero(start, as_tuple=False).squeeze(-1)
                ends = torch.cat([start[1:], torch.tensor([position_ids[keep_mask].shape[0]], device=device)])
                new_position_ids = torch.zeros_like(position_ids[keep_mask], device=device)
                for i in range(start.shape[0]):
                    new_position_ids[start[i]:ends[i]] = torch.arange(ends[i] - start[i], device=device)
                position_ids = new_position_ids
            if labels is not None:
                labels = labels[keep_mask]
            
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

        return None, attention_mask, position_ids, past_key_values, inputs_embeds, labels, reconstruction_mse_loss
