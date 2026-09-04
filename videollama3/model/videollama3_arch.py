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


def _grid_hw_for_compression_parts(compression_parts, grid_sizes, merge_sizes):
    """
    Map each compression part's [start, end) vision-token range to the post-merge
    (h, w) patch grid of the single grid_sizes entry it falls inside.

    grid_sizes: (num_grids, 3) rows of (t, h, w) in pre-merge patch units, one entry
    per video/image in the sample. merge_sizes: (num_grids,) merge_size per entry.
    A compression part is expected to come from exactly one video/image (compression
    windows are built per-video upstream), so it must fall entirely within one grid
    entry's token range.

    Returns: List[Tuple[int, int]], one (h, w) per compression part, aligned to
    compression_parts' input order (NOT sorted).
    """
    tokens_per_grid = []
    grid_hw = []
    for (t, h, w), m in zip(grid_sizes.tolist(), merge_sizes.tolist()):
        oh, ow = h // m, w // m
        tokens_per_grid.append(t * oh * ow)
        grid_hw.append((oh, ow))
    offsets = [0]
    for n in tokens_per_grid:
        offsets.append(offsets[-1] + n)

    result = []
    for start, end in compression_parts:
        owner = None
        for i in range(len(tokens_per_grid)):
            if start >= offsets[i] and end <= offsets[i + 1]:
                owner = i
                break
        assert owner is not None, (
            f"Compression part [{start}, {end}) does not fall within a single "
            f"grid_sizes entry (offsets={offsets}); every compression window must "
            f"come from one video/image so the compressor can assign it one (h, w)."
        )
        result.append(grid_hw[owner])
    return result


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
        grid_hws: List[Tuple[int, int]],
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        # compression_parts: [[start, end], [start, end], ...]
        # grid_hws: [(h, w), ...], one per compression part, same order as
        # compression_parts — the ACTUAL input frame grid for that part (see
        # _grid_hw_for_compression_parts). Determines each part's compressed OUTPUT
        # length via the compressor's own output_hw_for (fixed for
        # TransformerDecoderCompressor, == input (h, w) for SiglipAECompressor).
        # vision_tokens: [1, num_tokens, dim]
        device = vision_tokens.device
        vision_tokens = vision_tokens.squeeze(0) # [num_tokens, dim]
        compressor = self.get_token_compressor()
        compression_cu_seqlens = [0]
        need_compress_parts = torch.zeros(vision_tokens.shape[0], device=device, dtype=torch.bool)
        replace_mask = torch.zeros(vision_tokens.shape[0], device=device, dtype=torch.bool)
        for part, (h, w) in zip(compression_parts, grid_hws):
            part_len = part[1] - part[0]
            need_compress_parts[part[0]: part[1]] = True
            oh, ow = compressor.output_hw_for(h, w)
            replace_mask[part[0]: part[0] + oh * ow] = True
            compression_cu_seqlens.append(compression_cu_seqlens[-1] + part_len)
        compression_cu_seqlens = torch.tensor(compression_cu_seqlens, device=device, dtype=torch.long)

        # compressed vision tokens should have shape: [n, dim]
        original_tokens_to_reconstruct = vision_tokens[need_compress_parts]
        if hasattr(compressor, "compress_windows"):
            # Two-stage (Stage-1 per-segment lift + Stage-2 Mamba fold): each part is
            # one readout unit, subdivided into <= frames_per_segment-frame segments
            # inside the compressor. Output is (sum_parts M, dim), part-major — same
            # layout the single-stage call returns, so the scatter below is unchanged.
            compressed = compressor.compress_windows(
                original_tokens_to_reconstruct,
                compression_cu_seqlens,
                grid_hws,
            )
        else:
            compressed = compressor(
                original_tokens_to_reconstruct,
                compression_cu_seqlens,
                grid_hws,
            )
        keeping_masks = ~need_compress_parts | replace_mask
        vision_tokens[replace_mask] = compressed
        vision_tokens = vision_tokens[keeping_masks]
        return vision_tokens
    def encode_images(
        self,
        pixel_values: torch.FloatTensor,
        grid_sizes: torch.LongTensor,
        merge_sizes: torch.LongTensor,
        compression_parts: Optional[List[List[int]]] = None,
        grid_hws: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        reconstruction_mse_loss = None
        mm_features = self.get_model().get_vision_encoder()(
            pixel_values=pixel_values,
            grid_sizes=grid_sizes,
            merge_sizes=merge_sizes,
        )
        if getattr(self.config, "trainable_mm_compressor", False) and compression_parts is not None and len(compression_parts) > 0:
            assert compression_parts is not None, "compression_parts is required for trainable token compression."
            mm_features = self.compress_visual_tokens_with_compressor(
                mm_features,
                compression_parts,
                grid_hws,
            )
        mm_features = self.get_model().mm_projector(mm_features)
        return mm_features
    
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
        modals: Optional[torch.LongTensor] = None, # This parameter is currently not used in the model, but can be used to indicate the modality of each token for more flexible multimodal modeling.
        compression_parts: Optional[List[List[int]]] = None,
        compression_ts_info: Optional[List[Tuple[int, List[int]]]] = None,
    ):
        B, N = input_ids.shape
        device = input_ids.device
        if getattr(self.config, "trainable_mm_compressor", False) and pixel_values is not None:
            assert position_ids is not None, "Currently model only supports position_ids and flatten input."
            # Compression parts should like: [[1, 3], [4, 10], [16, 20]],  where each part indicates the start and end position of vision tokens to be compressed.
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
        image_positions = torch.nonzero(image_selected, as_tuple=False).squeeze(-1) # vision token's positions among all tokens
        grid_hws = None
        if compression_parts is not None and len(compression_parts) > 0:
            grid_hws = _grid_hw_for_compression_parts(compression_parts, grid_sizes, merge_sizes)
        mm_features = self.encode_images(
            pixel_values, grid_sizes, merge_sizes, compression_parts, grid_hws
        )

        if compression_parts is not None and len(compression_parts) > 0:
            compressor = self.get_token_compressor()
            # List-based construction: build the new token sequence piece-by-piece.
            # This lets us replace the per-frame "Time X.0s:" text with a range
            # "Time:Xs-Ye:" before each compression block.
            ids_segs, lbl_segs, attn_segs, is_start_segs = [], [], [], []

            def _append_seg(tok_ids, lbl_fill, attn_fill, is_sample_start_mask=None):
                if tok_ids is None or len(tok_ids) == 0:
                    return
                ids_segs.append(tok_ids)
                if labels is not None:
                    lbl_segs.append(lbl_fill)
                if attention_mask is not None:
                    attn_segs.append(attn_fill)
                # Track which positions are sample-starts (position_id == 0) so we can
                # re-number positions correctly after insertion of new tokens.
                if position_ids is not None:
                    if is_sample_start_mask is not None:
                        is_start_segs.append(is_sample_start_mask)
                    else:
                        is_start_segs.append(torch.zeros(len(tok_ids), device=device, dtype=torch.bool))

            prev = 0
            parts_with_hw = sorted(zip(compression_parts, grid_hws), key=lambda pair: pair[0][0])
            for part_idx, (part, (part_h, part_w)) in enumerate(parts_with_hw):
                out_h, out_w = compressor.output_hw_for(part_h, part_w)
                compact_vision_token_size = out_h * out_w
                part_start = image_positions[part[0]].item()
                part_end = image_positions[part[1] - 1].item()

                # How many tokens does the old "Time X.0s:" string occupy before part_start?
                old_ts_len = 0
                new_ts_ids: List[int] = []
                if compression_ts_info is not None and part_idx < len(compression_ts_info):
                    old_ts_len, new_ts_ids = compression_ts_info[part_idx]

                # 1. Keep everything from prev up to (but not including) old timestamp text.
                # Clamp: if old_ts_len is somehow larger than the gap (shouldn't happen with
                # well-formed data), fall back to keeping up to part_start (no replacement).
                keep_end = max(prev, part_start - old_ts_len)
                if keep_end > prev:
                    seg = input_ids[prev:keep_end]
                    _append_seg(
                        seg,
                        labels[prev:keep_end] if labels is not None else None,
                        attention_mask[prev:keep_end] if attention_mask is not None else None,
                        (position_ids[prev:keep_end] == 0) if position_ids is not None else None,
                    )

                # 2. Insert new range timestamp tokens (replaces old "Time X.0s:" text).
                if new_ts_ids:
                    ts_tensor = torch.tensor(new_ts_ids, device=device, dtype=input_ids.dtype)
                    _append_seg(
                        ts_tensor,
                        torch.full([len(new_ts_ids)], IGNORE_INDEX, device=device, dtype=labels.dtype) if labels is not None else None,
                        torch.ones(len(new_ts_ids), device=device, dtype=attention_mask.dtype) if attention_mask is not None else None,
                    )

                # 3. <compression_start>
                cs_tok = torch.tensor([self.config.compression_start_token_id], device=device, dtype=input_ids.dtype)
                _append_seg(
                    cs_tok,
                    torch.tensor([IGNORE_INDEX], device=device, dtype=labels.dtype) if labels is not None else None,
                    torch.ones(1, device=device, dtype=attention_mask.dtype) if attention_mask is not None else None,
                )

                # 4. Compressed image token placeholders (features filled in later by embed step).
                img_toks = torch.full([compact_vision_token_size], self.config.image_token_index, device=device, dtype=input_ids.dtype)
                _append_seg(
                    img_toks,
                    torch.full([compact_vision_token_size], IGNORE_INDEX, device=device, dtype=labels.dtype) if labels is not None else None,
                    torch.ones(compact_vision_token_size, device=device, dtype=attention_mask.dtype) if attention_mask is not None else None,
                )

                # 5. <compression_end>
                ce_tok = torch.tensor([self.config.compression_end_token_id], device=device, dtype=input_ids.dtype)
                _append_seg(
                    ce_tok,
                    torch.tensor([IGNORE_INDEX], device=device, dtype=labels.dtype) if labels is not None else None,
                    torch.ones(1, device=device, dtype=attention_mask.dtype) if attention_mask is not None else None,
                )

                prev = part_end + 1

            # 6. Everything after the last compression part.
            if prev < input_ids.shape[0]:
                seg = input_ids[prev:]
                _append_seg(
                    seg,
                    labels[prev:] if labels is not None else None,
                    attention_mask[prev:] if attention_mask is not None else None,
                    (position_ids[prev:] == 0) if position_ids is not None else None,
                )

            input_ids = torch.cat(ids_segs)
            if labels is not None:
                labels = torch.cat(lbl_segs)
            if attention_mask is not None:
                attention_mask = torch.cat(attn_segs)
            if position_ids is not None:
                is_start = torch.cat(is_start_segs)
                start = torch.nonzero(is_start, as_tuple=False).squeeze(-1)
                if start.dim() == 0:
                    start = start.unsqueeze(0)
                ends = torch.cat([start[1:], torch.tensor([input_ids.shape[0]], device=device)])
                new_position_ids = torch.zeros(input_ids.shape[0], device=device, dtype=torch.long)
                for i in range(start.shape[0]):
                    new_position_ids[start[i]:ends[i]] = torch.arange(ends[i] - start[i], device=device)
                position_ids = new_position_ids
            
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
