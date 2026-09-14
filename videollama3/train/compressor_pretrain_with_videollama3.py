#!/usr/bin/env python3
"""
Compressor finetuning with a SINGLE, whole-video compression window.

Same pipeline as ``videollama3_chat_finetune_compressor.py`` (frozen vision encoder
-> trainable token compressor -> LLM, CE on the assistant turns), with one change:
instead of sampling several fixed-size windows to compress
(``select_compression_parts`` / ``select_full_compression_parts``), **the whole video
is compressed by one compressor call into a single ``HW``-token summary**, whatever
its frame count.

    T frames x HW tokens  ->  compressor  ->  HW tokens  ->  mm_projector -> LLM

* One compression part per sample, so there is no partial/full split and no dual
  forward: the trainer runs a single CE forward (``use_dual_forward=False``).
* Every video yields exactly one part, so all ranks always take the compression path
  -- the NCCL desync the windowed script guards against (some ranks compressing, some
  not) cannot happen here.
* Timestamps still mark the *whole* span: the chat template's per-frame
  ``"Time X.0s:"`` text before the first frame is replaced by
  ``"Time:{first}s-{last}s:"``, and the per-frame timestamps inside the span are
  dropped along with the frame tokens they annotate (they sit between the first and
  last image token, which ``prepare_inputs_labels_for_multimodal`` cuts out).
* One video block per sample is assumed. Multi-turn online data (which
  ``preprocess_videollama3`` splits into one block per user turn) is out of scope and
  is skipped.

Frames are always decoded and encoded on the fly -- there is no pre-extracted
vision-feature cache (see ``docs/two_stage_compression_design.md`` §"On-the-fly
encoding").

``--max_frames N`` caps T at N (uniform subsample when the video is longer,
shorter clips untouched); ``--fixed_frames N`` resamples every video to exactly N
frames before the encoder (uniform subsample when longer, last frame repeated when
shorter) and takes precedence over ``--max_frames``. ``fixed_frames`` is optional for
``transformer_decoder`` / ``local_attn_conv``, which take any T at runtime, and
**required (a power of two) for ``siglip_ae``**: its ``log2(N)`` stride-2 Conv3d
stages are built at construction time, and while the convs themselves accept any T,
the stack only lands on the T=1 that the HW-token output contract requires when
T == 2^stages; its ``DynamicTokenSynthesizer`` bias table is sized by N as well.

Sequence-length note: the compression happens *inside* the model, so the sequence the
collator builds still holds the uncompressed ``T x HW`` image tokens (256 per frame at
448px / merge_size 2). It must fit in ``model_max_length`` or the collator truncates
it and the compression part no longer lines up; samples that would overflow are
skipped with a warning. Bound it with ``--max_frames`` / ``--fixed_frames``.
"""
import bisect
import copy
from dataclasses import dataclass, field
import json
import logging
import os
import pathlib
import random
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import transformers
from packaging import version

sys.path.append("./")

from videollama3.constants import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN,
    NUM_FRAMES,
    COMPRESSION_START_TOKEN,
    COMPRESSION_END_TOKEN,
)
from videollama3.model import Videollama3Qwen2Config, Videollama3Qwen2ForCausalLM  # noqa: E402
from videollama3.model.processor import Videollama3Processor  # noqa: E402
from videollama3.train.videollama3_trainer import (  # noqa: E402
    VideoLLaMA3Trainer,
    find_all_linear_names,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    safe_save_model_for_hf_trainer,
)
from videollama3.train.data import common as _data_common  # noqa: E402
from videollama3.train.data.common import (  # noqa: E402
    int_with_none, rank0_print, set_seed, _is_trainable_lr, _set_module_trainable,
)
from videollama3.train.data.compressor import DataCollatorWithCompressor  # noqa: E402
from videollama3.train.data.global_compressor import (  # noqa: E402
    GlobalCompressorLazySupervisedDataset,
    make_global_compressor_data_module,
)
from functools import partial

logger = logging.getLogger(__name__)
local_rank = None
torch.load = partial(torch.load, weights_only=False)
try:
    from deepspeed.runtime.fp16.loss_scaler import LossScaler
    from deepspeed.runtime.zero.config import ZeroStageEnum
    torch.serialization.add_safe_globals([LossScaler, ZeroStageEnum])
except ImportError:
    pass


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="pretrained_models/videollama3_7b_local")
    tokenizer_name_or_path: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v1")
    mm_projector_type: Optional[str] = field(default="linear")
    vision_encoder: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_attn_implementation: Optional[str] = field(default="flash_attention_2")
    use_token_compression: Optional[bool] = field(default=True)
    # transformer_decoder | transformer_decoder_flat | local_attn_conv | siglip_ae
    # (siglip_ae needs --fixed_frames 2**k)
    compressor_type: str = field(default="transformer_decoder")
    compressor_num_layers: int = field(default=8)
    compressor_num_attention_heads: int = field(default=8)
    compressor_intermediate_size: Optional[int] = field(default=None)
    compressor_attention_dropout: float = field(default=0.0)
    compressor_layer_norm_eps: float = field(default=1e-6)
    compress_image_w: int = field(default=16)
    compress_image_h: int = field(default=16)
    num_queries: int = field(
        default=32,
        metadata={"help": "Only used by compressor_type=transformer_decoder_flat: number of "
                          "flat output query tokens (sin/cos positional encoding over their "
                          "flat index, replacing the other transformer_decoder variant's "
                          "compress_image_h x compress_image_w spatial grid)."},
    )
    # Training-only common-component TOKEN pruning of the compressor input. A pure
    # compressor hyperparameter (like num_queries): it rides into the compressor via
    # token_compressor_config and only the compressor's .forward acts on it, in
    # .train() mode. Only transformer_decoder / transformer_decoder_flat honour it
    # (their cross-RoPE is rebuilt from each kept token's original (t, h, w)).
    token_prune_ratio: float = field(
        default=0.0,
        metadata={"help": "Fraction of each window's vision tokens the compressor drops (per window) "
                          "before its cross-attention, choosing the tokens whose direction is closest "
                          "to the window's common component (L2-normalised mean). 0 disables. "
                          "Inference always sees the full set."},
    )
    token_prune_min_tokens: int = field(
        default=0,
        metadata={"help": "Floor on tokens the compressor keeps per window (0 = one frame's worth, h*w)."},
    )
    pretrained_compressor_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional saved compressor weights (.pt/.bin) to warm-start token_compressor."},
    )
    # Keep the compressed tokens on the frozen encoder / mm_projector's feature
    # scale. Both ride into the compressor via token_compressor_config;
    # transformer_decoder / transformer_decoder_flat only.
    match_encoder_scale: bool = field(
        default=False,
        metadata={"help": "Option A: affine-map each compressed window's per-dim mean/std onto the "
                          "compressor's own KV input (the frozen encoder tokens) + a learnable "
                          "per-channel gamma/beta. Applies at train AND inference. Fixes the ~26x "
                          "compressed/raw norm gap deterministically."},
    )
    compressor_distr_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Option B: weight on the CORAL-style distribution-match aux loss "
                          "(compressed token cloud -> encoder token manifold: centroid + covariance "
                          "+ token-norm), added to the CE loss by the trainer. 0 = off. Composes "
                          "with --match_encoder_scale (which only fixes the covariance diagonal)."},
    )
    # Phase-1 fixed-count adaptive segmenter (transformer_decoder_flat only).
    adaptive_segmentation: bool = field(
        default=False,
        metadata={"help": "Subdivide the one whole-video compression window into "
                          "N = n_frames // segment_target_frames + 1 segments model-side, "
                          "each -> num_queries qbase tokens (output N*num_queries). Boundaries: "
                          "forced cut every segment_force_every frames + remaining budget on the "
                          "largest consecutive-frame encoder-feature cosine distances. "
                          "See docs/two_stage_compression_design.md §4 Phase 1."},
    )
    segment_target_frames: int = field(
        default=4,
        metadata={"help": "Adaptive segmenter: avg frames/segment; sets N = ⌊T/this⌋ + 1."},
    )
    segment_force_every: int = field(
        default=8,
        metadata={"help": "Adaptive segmenter: forced boundary every this many frames (the [1,N] clamp max)."},
    )
    segment_sample_tau: float = field(
        default=0.0,
        metadata={"help": "Adaptive segmenter: >0 draws the non-forced boundaries with Gumbel-top-k "
                          "over softmax(diff / tau) (train only) so the segmentation varies per epoch; "
                          "0 = deterministic top-k."},
    )


@dataclass
class DataArguments:
    data_path: List[str] = field(default=None)
    data_folder: Optional[str] = field(default=None)
    fps: Optional[int] = field(default=None)
    max_frames: Optional[int_with_none] = field(
        default=200,
        metadata={"help": "Upper bound on frames per sample (uniform subsample when longer, "
                          "shorter clips untouched). Overridden by --fixed_frames."},
    )
    multi_dataset: bool = field(default=False)
    image_merge_size: Optional[int] = field(default=1)
    video_merge_size: Optional[int] = field(default=1)
    mm_max_length: Optional[int] = field(default=10240)
    image_aspect_ratio: str = "square"
    use_batch_flattening: bool = field(default=True)
    dataset_cache_dir: Optional[str] = field(default=None)
    force_image_size: Optional[int] = field(default=None)
    fixed_frames: int = field(
        default=0,
        metadata={"help": "Resample every video to exactly this many frames (0 = keep as decoded). "
                          "Must be a power of two >= 2 for compressor_type=siglip_ae."},
    )
    validation_split_rate: float = field(
        default=0,
        metadata={"help": "Percentage of the train set used as validation set."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    # Default to compressor-only training.
    vision_encoder_lr: Optional[float] = field(default=0.0)
    mm_projector_lr: Optional[float] = field(default=0.0)
    compressor_lr: Optional[float] = field(default=1e-4)
    # Two-stage compressor (qbase + Mamba-2/SSD fold) LR split -- both no-ops for a
    # single-stage compressor (Phase 1: just leave them at 0, compressor_lr is the
    # qbase's only rate):
    #   qbase_lr  > 0 -> token_compressor.stage1.* (the qbase) gets its OWN
    #     optimizer group at this LR instead of sharing the fold's rate -- used by
    #     the joint-polish window that unfreezes the qbase and moves it ~10x
    #     slower than the fold.
    #   mamba_lr  > 0 -> the rate for everything else in the compressor
    #     (token_compressor.stage2.* / embed_tokens), overriding compressor_lr.
    #     0 -> falls back to compressor_lr.
    qbase_lr: Optional[float] = field(default=0.0)
    mamba_lr: Optional[float] = field(default=0.0)
    llm_lr: Optional[float] = field(default=0.0)
    group_by_modality_length: bool = field(default=False)
    # Two-stage fold: fill each grad-accum window from one depth class (N = ⌊T/4⌋+1)
    # so max_u N_u — the SSD recurrence depth / backward-graph depth — stays
    # homogeneous. Needs the dataset to expose `compression_depths`.
    group_by_compression_depth: bool = field(default=False)
    model_max_length: int = field(default=32768)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


def _build_token_compressor_config(
    model_config: Videollama3Qwen2Config, model_args: ModelArguments, data_args: DataArguments
) -> Dict:
    return {
        "compressor_type": model_args.compressor_type,
        "hidden_size": model_config.mm_hidden_size,
        "intermediate_size": model_args.compressor_intermediate_size or model_config.mm_hidden_size * 4,
        "num_layers": model_args.compressor_num_layers,
        "num_attention_heads": model_args.compressor_num_attention_heads,
        "attention_probs_dropout_prob": model_args.compressor_attention_dropout,
        "layer_norm_eps": model_args.compressor_layer_norm_eps,
        "compress_image_w": model_args.compress_image_w,
        "compress_image_h": model_args.compress_image_h,
        # siglip_ae builds log2(window_size) stride-2 stages and a per-frame bias table,
        # so it needs the exact frame count; the other two derive T from cu_seqlens.
        "window_size": data_args.fixed_frames if model_args.compressor_type == "siglip_ae" else 0,
        # Only transformer_decoder_flat reads this; harmless for the other types.
        "num_queries": model_args.num_queries,
        # Training-only KV pruning; only transformer_decoder / transformer_decoder_flat act on it.
        "token_prune_ratio": model_args.token_prune_ratio,
        "token_prune_min_tokens": model_args.token_prune_min_tokens,
        # Option A / Option B -- keep compressed tokens on the frozen-encoder scale.
        "match_encoder_scale": model_args.match_encoder_scale,
        "distr_loss_weight": model_args.compressor_distr_loss_weight,
        # Phase-1 fixed-count adaptive segmenter (transformer_decoder_flat only).
        "adaptive_segmentation": model_args.adaptive_segmentation,
        "segment_target_frames": model_args.segment_target_frames,
        "segment_force_every": model_args.segment_force_every,
        "segment_sample_tau": model_args.segment_sample_tau,
    }


def train(attn_implementation=None, *,
          model_args_cls=None,
          data_args_cls=None,
          dataset_cls=None,
          build_token_compressor_config=None,
          configure_image_processor=None,
          on_compressor_built=None):
    """Phase-1 qbase CE pretraining (and, via the hooks, the Phase-2 fold script).

    The keyword-only hooks let a thin wrapper (phase2_pretrain_fold.py)
    swap pieces without monkeypatching this module:
      model_args_cls / data_args_cls   -- dataclasses handed to HfArgumentParser
      dataset_cls                      -- dataset class make_global_compressor_data_module builds
      build_token_compressor_config    -- fn(model_config, model_args, data_args) -> dict
      configure_image_processor        -- fn(image_processor, model_args, data_args) -> None, run pre-wrap
      on_compressor_built              -- fn(compressor_module, model_args, data_args, training_args) -> None, run after requires_grad is set
    """
    global local_rank
    set_seed(42)
    model_args_cls = model_args_cls or ModelArguments
    data_args_cls = data_args_cls or DataArguments
    build_token_compressor_config = build_token_compressor_config or _build_token_compressor_config

    parser = transformers.HfArgumentParser((model_args_cls, data_args_cls, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.token_prune_ratio > 0 and model_args.compressor_type not in (
        "transformer_decoder", "transformer_decoder_flat"
    ):
        raise ValueError(
            f"--token_prune_ratio is only supported with compressor_type 'transformer_decoder' or "
            f"'transformer_decoder_flat' (got '{model_args.compressor_type}'): only their cross-attention "
            f"RoPE is rebuilt from each kept token's original (t, h, w). Disable one of them."
        )

    if model_args.compressor_type == "siglip_ae":
        n = data_args.fixed_frames
        if n < 2 or (n & (n - 1)) != 0:
            raise ValueError(
                "compressor_type='siglip_ae' halves T with log2(fixed_frames) stride-2 stages built "
                f"at construction time, and every window must end at T=1 to produce the "
                f"{model_args.compress_image_w * model_args.compress_image_h}-token output the model "
                f"scatters back. Set --fixed_frames to a power of two >= 2 (got {n})."
            )

    log_file = os.path.join(training_args.output_dir, "training.log")
    error_log_file = os.path.join(training_args.output_dir, "training_errors.log")
    os.makedirs(training_args.output_dir, exist_ok=True)

    log_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(log_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    error_handler = logging.FileHandler(error_log_file, mode="a")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(log_formatter)
    root_logger.addHandler(error_handler)

    local_rank = training_args.local_rank
    _data_common.local_rank = local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    config = Videollama3Qwen2Config.from_pretrained(model_args.model_name_or_path)
    config._attn_implementation = attn_implementation
    config.mm_attn_implementation = attn_implementation
    config.use_token_compression = True
    config.trainable_mm_compressor = True
    if model_args.vision_encoder is not None:
        config.vision_encoder = model_args.vision_encoder

    model = Videollama3Qwen2ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=compute_dtype,
        do_sample=True,
    )
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, _input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # initialize_vision_modules unconditionally overwrites config.mm_projector_type
    # with model_args.mm_projector_type (default "linear"), but it only *rebuilds*
    # the projector when there is none — and from_pretrained already built the base
    # model's projector (mlp2x_gelu). This script never swaps the projector, so keep
    # the base config's label; otherwise the saved checkpoint's config says "linear"
    # while the weights are the mlp2x_gelu `readout.*` and every reload builds a
    # random Linear.
    _base_mm_projector_type = getattr(model.config, "mm_projector_type", None)
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
    if _base_mm_projector_type is not None:
        model.config.mm_projector_type = _base_mm_projector_type
    vision_encoder = model.get_vision_encoder()
    vision_encoder.to(dtype=compute_dtype, device=training_args.device)

    mm_projector = model.get_mm_projector()
    mm_projector.to(dtype=compute_dtype if training_args.bf16 else torch.float16, device=training_args.device)

    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_hidden_size = vision_encoder.hidden_size
    model.config.token_compressor_config = build_token_compressor_config(model.config, model_args, data_args)

    # Rebuild compressor with latest config dict.
    from videollama3.model.compressor import build_token_compressor

    model.get_model().token_compressor = build_token_compressor(model.config)
    if model.get_model().token_compressor is None:
        raise RuntimeError("Failed to build token_compressor. Check token_compressor_config.")
    if model_args.pretrained_compressor_path:
        _p = model_args.pretrained_compressor_path
        if os.path.isdir(_p):
            # HF checkpoint dir -> pull token_compressor.* out of the safetensors
            # shards (same loader TwoStageCompressor.load_stage1_pretrained uses).
            from videollama3.model.compressor import _load_flat_compressor_state_dict
            state = _load_flat_compressor_state_dict(_p)
        else:
            state = torch.load(_p, map_location="cpu")
            state = state.get("compressor", state) if isinstance(state, dict) else state
        missing, unexpected = model.get_model().token_compressor.load_state_dict(state, strict=False)
        rank0_print(
            f"[INFO] Loaded pretrained compressor from {model_args.pretrained_compressor_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        if missing or unexpected:
            rank0_print(f"[WARN] missing keys: {missing}\n[WARN] unexpected keys: {unexpected}")
    model.get_model().token_compressor.to(dtype=compute_dtype, device=training_args.device)

    # TwoStageCompressor (`+mamba`): the qbase-only replay stream and the fold
    # readout are two distinct distributions that used to share one mm_projector.
    # Give each a fresh, independent copy of the (now correctly loaded) shared
    # mm_projector -- unconditionally, same as token_compressor's own rebuild
    # above: a genuine resume (same OUTPUT_DIR) has the Trainer's own checkpoint
    # load restore the real, by-then-diverged weights over this right afterwards.
    if hasattr(model.get_model().token_compressor, "compress_windows"):
        import copy
        mm_projector = model.get_mm_projector()
        model.get_model().mm_projector_qbase = copy.deepcopy(mm_projector)
        model.get_model().mm_projector_fold = copy.deepcopy(mm_projector)
        model.get_model().mm_projector_qbase.to(dtype=compute_dtype, device=training_args.device)
        model.get_model().mm_projector_fold.to(dtype=compute_dtype, device=training_args.device)
        rank0_print("[INFO] TwoStageCompressor: split mm_projector_qbase / mm_projector_fold "
                    "from the shared mm_projector")

    model.config.llm_lr = training_args.llm_lr
    model.config.vision_encoder_lr = training_args.vision_encoder_lr
    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.config.compressor_lr = training_args.compressor_lr

    llm_trainable = _is_trainable_lr(model.config.llm_lr)
    vision_trainable = _is_trainable_lr(model.config.vision_encoder_lr)
    projector_trainable = _is_trainable_lr(model.config.mm_projector_lr)
    compressor_trainable = _is_trainable_lr(model.config.compressor_lr)

    if training_args.lora_enable:
        # get_peft_model() already froze all base weights and enabled only LoRA params.
        # If llm_lr=0, also freeze the LoRA params themselves.
        if not llm_trainable:
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = False
        _set_module_trainable(model.get_vision_encoder(), vision_trainable)
        _set_module_trainable(model.get_mm_projector(), projector_trainable)
        _set_module_trainable(getattr(model.get_model(), "mm_projector_qbase", None), projector_trainable)
        _set_module_trainable(getattr(model.get_model(), "mm_projector_fold", None), projector_trainable)
        _set_module_trainable(getattr(model.get_model(), "token_compressor", None), compressor_trainable)
    else:
        _set_module_trainable(model.get_model(), llm_trainable)
        _set_module_trainable(model.get_vision_encoder(), vision_trainable)
        _set_module_trainable(model.get_mm_projector(), projector_trainable)
        _set_module_trainable(getattr(model.get_model(), "mm_projector_qbase", None), projector_trainable)
        _set_module_trainable(getattr(model.get_model(), "mm_projector_fold", None), projector_trainable)
        _set_module_trainable(getattr(model.get_model(), "token_compressor", None), compressor_trainable)

    token_compressor = getattr(model.get_model(), "token_compressor", None)
    if on_compressor_built is not None and token_compressor is not None:
        on_compressor_built(token_compressor, model_args, data_args, training_args)

    total_param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_param_count == 0:
        raise RuntimeError(
            "No trainable parameters found. "
            "Please set at least one of llm_lr / vision_encoder_lr / mm_projector_lr / compressor_lr > 0."
        )
    if training_args.local_rank in (0, -1):
        trainable_ratio = 100.0 * trainable_param_count / total_param_count
        rank0_print(
            f"Trainable parameters: {trainable_param_count:,} / {total_param_count:,} "
            f"({trainable_ratio:.4f}%)"
        )

    model.config.max_frames = getattr(data_args, "max_frames", NUM_FRAMES)
    model.config.image_aspect_ratio = data_args.image_aspect_ratio if "avt" not in model_args.vision_encoder else "avt"
    model.config.image_size = data_args.image_size = vision_encoder.image_size
    model.config.image_token_length = data_args.image_token_length = mm_projector.cal_proj_size(
        vision_encoder.num_patches_per_side
    )
    old_vocabulary_size = len(tokenizer)
    new_tokens = tokenizer.add_tokens([COMPRESSION_START_TOKEN, COMPRESSION_END_TOKEN], special_tokens=True)
    if new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
        if not llm_trainable:
            # Only the new rows need to be learned; old rows stay frozen. requires_grad
            # stays True on the whole embed_tokens tensor so ZeRO-2 assigns optimizer
            # state to it (a requires_grad parameter without optimizer state corrupts
            # AllReduce buckets -> NaN at step 2); a backward hook zeros the gradient of
            # the pre-existing rows. create_optimizer routes embed_tokens into the
            # compressor parameter group.
            _old_vocab = old_vocabulary_size

            def _zero_old_embed_rows(grad, _ov=_old_vocab):
                g = grad.clone()
                g[:_ov].zero_()
                return g

            embed = model.get_input_embeddings()
            embed.weight.requires_grad_(True)
            embed.weight.register_hook(_zero_old_embed_rows)

            # lm_head: compression tokens are always masked with IGNORE_INDEX in labels
            # so their rows never receive gradients; freeze it entirely.
            out_embed = model.get_output_embeddings()
            if out_embed is not None and out_embed.weight is not embed.weight:
                _set_module_trainable(out_embed, False)
    model.config.image_token_index = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    model.config.compression_start_token_id = tokenizer.convert_tokens_to_ids(COMPRESSION_START_TOKEN)
    model.config.compression_end_token_id = tokenizer.convert_tokens_to_ids(COMPRESSION_END_TOKEN)
    # Pre-tokenise the "Time:{a}s-{b}s:" pieces onto the config so the arch can build
    # the per-unit range string with no tokenizer at forward time (two-stage fold).
    from videollama3.model.compressor import bake_time_tokens
    bake_time_tokens(model.config, tokenizer,
                     max_seconds=max(2048, int(getattr(data_args, "max_frames", 0) or 0) + 64))

    if data_args.force_image_size is not None:
        vision_encoder.image_processor.force_size = [data_args.force_image_size] * 2
        rank0_print(f"Force set image size to be {data_args.force_image_size}")
    if configure_image_processor is not None:
        configure_image_processor(vision_encoder.image_processor, model_args, data_args)
    vlprocessor = Videollama3Processor(vision_encoder.image_processor, tokenizer)

    assert data_args.use_batch_flattening, "Compressor training currently requires flattening mode (batch size 1 sequence)."
    assert model.config._attn_implementation == "flash_attention_2"
    assert version.parse(transformers.__version__) >= version.parse("4.44.0")

    if model_args.compressor_type == "transformer_decoder_flat":
        if model_args.adaptive_segmentation:
            _out_tokens = (
                f"N*{model_args.num_queries} tokens "
                f"(N = T//{model_args.segment_target_frames}+1 adaptive segments, "
                f"force_every={model_args.segment_force_every}, tau={model_args.segment_sample_tau})"
            )
        else:
            _out_tokens = f"{model_args.num_queries} tokens (flat query bank)"
    else:
        _out_tokens = f"{model_args.compress_image_w * model_args.compress_image_h} tokens (h*w grid)"
    rank0_print(
        f"[INFO] Whole-video compression: 1 window per sample -> {_out_tokens} "
        f"(compressor_type={model_args.compressor_type}, "
        f"max_frames={data_args.max_frames}, "
        f"fixed_frames={data_args.fixed_frames or 'off'}, "
        f"token_prune_ratio={model_args.token_prune_ratio or 'off'})"
    )

    data_module = make_global_compressor_data_module(
        vlprocessor=vlprocessor,
        data_args=data_args,
        output_dir=training_args.output_dir,
        dataset_cls=dataset_cls,
        model_args=model_args,
    )

    trainer = VideoLLaMA3Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        # Single compression mode -> single CE forward.
        use_dual_forward=False,
        partial_loss_weight=1.0,
        full_loss_weight=0.0,
        **data_module,
    )

    # Feed (epoch, training progress) into the dataset so its per-(epoch, index)
    # seeded draws advance and the compression-variance curriculum can anneal.
    # Harmless when the dataset does not use them.
    import transformers as _tf

    class _DatasetProgressCallback(_tf.TrainerCallback):
        def _targets(self, ds):
            subs = getattr(ds, "datasets", None)
            return list(subs) if subs else ([ds] if ds is not None else [])

        def _set(self, ds, **kw):
            for t in self._targets(ds):
                inner = getattr(t, "dataset", t)  # unwrap SubsetWithLengths
                for k, v in kw.items():
                    if hasattr(inner, k):
                        setattr(inner, k, v)

        def on_epoch_begin(self, args, state, control, **kw):
            self._set(trainer.train_dataset, _epoch=int(state.epoch or 0))

        def on_step_begin(self, args, state, control, **kw):
            total = max(1, state.max_steps or 1)
            self._set(trainer.train_dataset, _progress=float(state.global_step) / total)

    trainer.add_callback(_DatasetProgressCallback())

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
            vlprocessor.save_pretrained(training_args.output_dir)
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        if trainer.args.should_save:
            vlprocessor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
