#!/usr/bin/env python3
"""qbase LLM-SFT -- LoRA-unfreeze the LLM on top of a frozen qbase, QA-dominant data.

Why (2026-10-01): with the LLM frozen, qbase_enhance ties raw frames on Video-MME long
perception questions but trails on reasoning / synopsis (``eval_ablation/long_mcq_budget.py``
+ ``long_mcq_segment.py``; segmentation ruled out). Reading compressed tokens is learned;
reasoning over a token format the LLM never trained on is the open half, and only the
LLM can learn it. Data: ``dataset_util/build_qa_sft_blend.py`` (QA turns, a small caption
replay, and a raw-frame stream).

Thin wrapper over ``qbase_enhance_pretrain.py`` (its dataset, args and hooks, through
``compressor_pretrain_with_videollama3.train``). One addition:

* **Raw-frame stream.** A registry entry flagged ``"raw_frames": true`` is fed to the LLM
  UNCOMPRESSED: each sample draws a budget ``B`` from ``RAW_BUDGETS`` and a per-frame size
  from ``RAW_TOK_PER_FRAME``, keeps ``min(T, B / tok_per_frame)`` uniformly spaced frames
  and lets the image processor fit them into ``B`` tokens (stock floor of 16 / frame) --
  the operating points of the raw arms in ``long_mcq_budget.py``. The sample carries no
  compression part, so its tokens bypass the compressor and only train the LoRA (and the
  shared ``mm_projector``). It keeps the LLM's native raw-frame skill, which is what makes
  the raw arm a valid control after the unfreeze. Raw and compressed tokens never share a
  sample (the training rule in CLAUDE.md).

Trainable: LoRA on the LLM (``--llm_lr``) + ``mm_projector`` (``--mm_projector_lr``). The
qbase is frozen (``--compressor_lr 0``) so any gain is attributable to the LLM side.
Canonical invocation: ``shell/sft_qbase_lora.sh``. The run writes a PEFT adapter; merge
with ``eval_ablation/merge_lora_checkpoint.py --base work_dirs/qbase_enhance`` before eval.
"""
from __future__ import annotations

import random
import sys
from typing import Dict, Optional

sys.path.append("./")

import videollama3.train.qbase_enhance_pretrain as qe
from videollama3.train.data.global_compressor import get_video_content, resample_video_frames

base = qe.base
RAW_BUDGETS = (1024, 2048, 4096)
RAW_TOK_PER_FRAME = (16, 32, 64)
RAW_MIN_TOKENS = 16                     # the stock VideoLLaMA3 per-frame floor


class QbaseLLMSFTDataset(qe.QbaseEnhanceDataset):
    """QbaseEnhanceDataset + the uncompressed raw-frame stream (``self.raw_frames``)."""

    def _convert_normal(self, data_dict):
        out = super()._convert_normal(data_dict)
        if not self.raw_frames or out[0] != "video":
            return out
        modal, images, messages, merge_size = out
        content = get_video_content(messages)
        B = random.choice(RAW_BUDGETS)
        n = max(1, min(int(content["num_frames"]), B // random.choice(RAW_TOK_PER_FRAME)))
        images = resample_video_frames(images, content, n)
        # the vlprocessor call that follows (in GlobalCompressorLazySupervisedDataset.
        # __getitem__) packs these n frames into B tokens; __getitem__ restores the settings
        ip = self.vlprocessor.image_processor
        ip.max_tokens, ip.min_tokens, ip.max_tokens_per_frame = B, RAW_MIN_TOKENS, None
        return modal, images, messages, merge_size

    def _reject_if_too_small(self, i, n_out, total_vision_tokens, total_frames, what=""):
        # a raw sample is not compressed, so the compressed length cannot overflow it
        return False if self.raw_frames else super()._reject_if_too_small(
            i, n_out, total_vision_tokens, total_frames, what)

    def __getitem__(self, i, _retries: int = 0) -> Dict:
        if not self.raw_frames:
            return super().__getitem__(i, _retries)
        ip = self.vlprocessor.image_processor
        saved = (ip.max_tokens, ip.min_tokens, ip.max_tokens_per_frame)
        try:
            data_dict = super().__getitem__(i, _retries)
        finally:
            ip.max_tokens, ip.min_tokens, ip.max_tokens_per_frame = saved
        # no compression part: the collator then adds no part / seed / ts entry for this
        # sample and prepare_inputs_labels_for_multimodal leaves its tokens raw
        data_dict["compression_parts"] = []
        data_dict["compression_ts_info"] = []
        data_dict["compression_is_image"] = []
        data_dict["compression_seed"] = []
        return data_dict

    @property
    def encode_costs(self) -> Optional[list]:
        if not self.raw_frames:
            return super().encode_costs
        if not hasattr(self, "_encode_costs"):
            b = sum(RAW_BUDGETS) / len(RAW_BUDGETS)          # ~B tokens = 4B patches, B LLM tokens
            self._encode_costs = [qe._ENC_S_PER_PATCH * 4 * b + qe._LLM_S_PER_TOKEN * b] * len(self)
        return self._encode_costs


if __name__ == "__main__":
    base.train(
        attn_implementation="flash_attention_2",
        model_args_cls=qe.QbaseEnhanceModelArguments,
        data_args_cls=qe.QbaseEnhanceDataArguments,
        dataset_cls=QbaseLLMSFTDataset,
        build_token_compressor_config=qe._build_config,
        configure_image_processor=base._configure_image_processor,
        on_compressor_built=qe._on_built,
    )
