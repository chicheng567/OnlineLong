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
        visual_segment_boundaries: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        # compression_mask: [num_visual_tokens] or [1, num_visual_tokens]
        # vision_tokens: [1, num_tokens, dim]
        device = vision_tokens.device
        if compression_mask.dim() == 2:
            compression_mask = compression_mask.view(-1)
        compression_mask = compression_mask.to(device=device, dtype=torch.bool)
        if not compression_mask.any():
            keep_mask = torch.ones_like(compression_mask, dtype=torch.bool, device=device)
            empty = torch.empty(0, dtype=torch.long, device=device)
            return vision_tokens, keep_mask, empty, empty

        vision_tokens_flat = vision_tokens.view(-1, vision_tokens.size(-1))
        need_compress_parts = vision_tokens_flat[compression_mask] # [num_compress_tokens, dim]
        #
        compression_mask_tmp = torch.cat([
            torch.tensor([0], device=device), 
            compression_mask.int(), 
            torch.tensor([0], device=device)], dim=0)
        diff = compression_mask_tmp[1:] - compression_mask_tmp[:-1]
        starts = (diff == 1).nonzero(as_tuple=False).squeeze(-1)
        ends = (diff == -1).nonzero(as_tuple=False).squeeze(-1)
        if visual_segment_boundaries is not None and visual_segment_boundaries.numel() > 0 and starts.numel() > 0:
            split_points = set(int(x) for x in visual_segment_boundaries.tolist())
            split_points.discard(0)
            split_points.discard(int(compression_mask.numel()))
            if split_points:
                new_segments = []
                for s, e in zip(starts.tolist(), ends.tolist()):
                    points = [p for p in split_points if s < p < e]
                    if not points:
                        new_segments.append((s, e))
                        continue
                    points = [s] + sorted(points) + [e]
                    for p0, p1 in zip(points[:-1], points[1:]):
                        if p1 > p0:
                            new_segments.append((p0, p1))
                starts = torch.tensor([x[0] for x in new_segments], device=device, dtype=torch.long)
                ends = torch.tensor([x[1] for x in new_segments], device=device, dtype=torch.long)
        parts = ends - starts
        compressor = self.get_model().get_token_compressor()
        required_tokens = getattr(compressor, "compress_image_wh", None)
        if required_tokens is not None:
            required_tokens = int(required_tokens)
            too_short = torch.nonzero(parts < required_tokens, as_tuple=False).squeeze(-1)
            if too_short.numel() > 0:
                bad_lengths = parts[too_short].detach().cpu().tolist()
                bad_segments = too_short.detach().cpu().tolist()
                raise ValueError(
                    "Found compression segment shorter than compress_image_wh: "
                    f"compress_image_wh={required_tokens}, "
                    f"segment_ids={bad_segments}, segment_lengths={bad_lengths}."
                )
        compression_cu_seqlens = torch.cat([torch.tensor([0], device=device), parts.cumsum(dim=0)], dim=0)
        # compressed vision tokens should have shape: [n, dim]
        compressed = compressor(
            need_compress_parts,
            compression_cu_seqlens
        )
        keeping_masks = ~compression_mask
        write_ptr = 0
        for seg_start, seg_end in zip(starts.tolist(), ends.tolist()):
            comp = compressed[write_ptr].reshape(-1, compressed.size(-1))
            write_ptr += 1
            seg_len = seg_end - seg_start
            if comp.numel() == 0:
                continue
            k = min(comp.size(0), seg_len)
            keep_positions = torch.arange(seg_start, seg_start + k, device=device, dtype=torch.long)
            keeping_masks[keep_positions] = True
            vision_tokens_flat[keep_positions] = comp[:k]
        vision_tokens = vision_tokens_flat[keeping_masks]
        vision_tokens = vision_tokens.to(device).unsqueeze(0)
        keeping_masks = keeping_masks.to(device)
        if vision_tokens.dim() == 2 and vision_tokens.size(-1) == vision_tokens_flat.size(-1):
            vision_tokens = vision_tokens.unsqueeze(0)
        return vision_tokens, keeping_masks, starts, ends
    def encode_images(
        self,
        pixel_values: torch.FloatTensor,
        grid_sizes: torch.LongTensor,
        merge_sizes: torch.LongTensor,
        compression_mask: Optional[torch.BoolTensor] = None,
        visual_segment_boundaries: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[torch.BoolTensor], Optional[torch.LongTensor], Optional[torch.LongTensor]]:
        mm_features = self.get_model().get_vision_encoder()(
            pixel_values=pixel_values,
            grid_sizes=grid_sizes,
            merge_sizes=merge_sizes,
        )
        keepmask = None
        starts, ends = None, None
        if self.config.trainable_mm_compressor:
            assert compression_mask is not None, "compression_mask is required for trainable token compression."
            mm_features, keepmask, starts, ends = self.compress_visual_tokens_with_compressor(
                mm_features,
                compression_mask,
                visual_segment_boundaries=visual_segment_boundaries,
            )
        mm_features = self.get_model().mm_projector(mm_features)
        return mm_features, keepmask, starts, ends
    
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
        compression_parts: Optional[torch.BoolTensor] = None,
    ):
        #TODO: Change compression logic. From compression mask -> list with compression idx. Avoiding cross sample compression and make it more efficient.
        B, N = input_ids.shape
        if self.config.trainable_mm_compressor:
            assert position_ids is not None, "Currently model only supports position_ids and flatten input."
            # Compression parts should like: [[0, 3], [5, 8], [10, 15]....]
            assert compression_parts is not None, "compression_parts is required for trainable token compression."
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
        image_selected = (input_ids == self.config.image_token_index)
        image_positions = torch.nonzero(image_selected, as_tuple=False).squeeze(-1)
        compression_mask_visual = None
        visual_segment_boundaries = None
        if compression_mask is not None:
            compression_mask = compression_mask.view(-1).to(dtype=torch.bool, device=input_ids.device)
            assert compression_mask.numel() == input_ids.numel(), "compression_mask must have same length as input_ids."
            compression_mask_visual = compression_mask[image_selected]
            if position_ids is not None:
                sample_starts = torch.nonzero(position_ids == 0, as_tuple=False).squeeze(-1)
                boundaries = []
                for token_start in sample_starts.tolist():
                    if token_start <= 0:
                        continue
                    boundaries.append(int(image_selected[:token_start].sum().item()))
                if boundaries:
                    visual_segment_boundaries = torch.tensor(boundaries, dtype=torch.long, device=input_ids.device)
        mm_features, keepmask_visual, seg_starts, seg_ends = self.encode_images(
            pixel_values, grid_sizes, merge_sizes, compression_mask_visual, visual_segment_boundaries
        )
        # modify input_ids, position_ids, attention_mask, labels accordingly
        # TODO: Generate keepmask directly from compression idx and avoiding return keepmask_visual from encode_images.
        if keepmask_visual is not None:
            keepmask = torch.ones_like(input_ids, dtype=torch.bool)
            keep_image_positions = image_positions[keepmask_visual]
            keep_image_mask = torch.zeros_like(image_selected)
            keep_image_mask[keep_image_positions] = True
            keepmask[image_selected] = keep_image_mask[image_selected]

            # Drop all text tokens between compressed frame blocks to keep compact windows.
            if seg_starts is not None and seg_ends is not None:
                for s, e in zip(seg_starts.tolist(), seg_ends.tolist()):
                    if e - s <= 0:
                        continue
                    left = int(image_positions[s].item())
                    right = int(image_positions[e - 1].item())
                    if right <= left:
                        continue
                    interval = torch.arange(left + 1, right, device=input_ids.device)
                    if interval.numel() > 0:
                        keepmask[interval] = keepmask[interval] & image_selected[interval]
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
