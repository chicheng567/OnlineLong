import bisect
from dataclasses import dataclass, field
import json
import logging
import os
import pathlib
import random
import sys
from typing import Dict, List, Optional, Sequence
import warnings
import torch
import transformers
from packaging import version

import torch.utils.data
sys.path.append("./")

from videollama3.constants import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN,
    NUM_FRAMES,
    STREAM_END_TOKEN,
    STREAM_START_TOKEN,
    COMPRESSION_START_TOKEN,
    COMPRESSION_END_TOKEN
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
from videollama3.train.data.compressor import (  # noqa: E402
    CompressorLazySupervisedDataset,
    DataCollatorWithCompressor,
    SubsetWithLengths,
    make_compressor_data_module,
    select_compression_parts,
    select_full_compression_parts,
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
    # Compressor architecture args
    compressor_type: str = field(default="transformer_decoder")
    compressor_num_layers: int = field(default=8)
    compressor_num_attention_heads: int = field(default=8)
    compressor_intermediate_size: Optional[int] = field(default=None)
    compressor_attention_dropout: float = field(default=0.0)
    compressor_layer_norm_eps: float = field(default=1e-6)
    compress_image_w: int = field(default=16)
    compress_image_h: int = field(default=16)


@dataclass
class DataArguments:
    data_path: List[str] = field(default=None)
    data_folder: Optional[str] = field(default=None)
    fps: Optional[int] = field(default=None)
    max_frames: Optional[int_with_none] = field(default=200)
    multi_dataset: bool = field(default=False)
    image_merge_size: Optional[int] = field(default=1)
    video_merge_size: Optional[int] = field(default=1)
    mm_max_length: Optional[int] = field(default=10240)
    image_aspect_ratio: str = "square"
    use_batch_flattening: bool = field(default=True)
    dataset_cache_dir: Optional[str] = field(default=None)
    force_image_size: Optional[int] = field(default=None)
    # Compressor sampling policy
    compression_ratio: float = field(default=0.3, metadata={"help": "Frame ratio to compress per video."})
    compression_window_size: int = field(default=3, metadata={"help": "Fixed frame window size for each compressed span."})
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
    llm_lr: Optional[float] = field(default=0.0)
    group_by_modality_length: bool = field(default=False)
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
    # Dual-forward loss weights
    use_dual_forward: bool = field(default=True, metadata={"help": "Run both partial and full compression forwards and sum their CE losses."})
    partial_loss_weight: float = field(default=0.5, metadata={"help": "Weight applied to the partial-compression loss term."})
    full_loss_weight: float = field(default=0.5, metadata={"help": "Weight applied to the full-compression loss term (only used when use_dual_forward=True)."})
    step_infer_enabled: bool = field(
        default=True,
        metadata={"help": "Run a random train-sample inference at every step end on rank 0."},
    )
    step_infer_max_new_tokens: int = field(
        default=64,
        metadata={"help": "Max generated tokens for step inference logging."},
    )
    step_infer_do_sample: bool = field(
        default=False,
        metadata={"help": "Use sampling instead of greedy decoding for step inference logging."},
    )


def _build_token_compressor_config(model_config: Videollama3Qwen2Config, model_args: ModelArguments, data_args: Optional[DataArguments]) -> Dict:
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
        "window_size": data_args.compression_window_size,
    }


def train(attn_implementation=None):
    global local_rank
    set_seed(42)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
    vision_encoder = model.get_vision_encoder()
    vision_encoder.to(dtype=compute_dtype, device=training_args.device)

    mm_projector = model.get_mm_projector()
    mm_projector.to(dtype=compute_dtype if training_args.bf16 else torch.float16, device=training_args.device)

    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_hidden_size = vision_encoder.hidden_size
    model.config.token_compressor_config = _build_token_compressor_config(model.config, model_args, data_args)

    # Rebuild compressor with latest config dict.
    from videollama3.model.compressor import build_token_compressor

    model.get_model().token_compressor = build_token_compressor(model.config)
    if model.get_model().token_compressor is None:
        raise RuntimeError("Failed to build token_compressor. Check token_compressor_config.")
    model.get_model().token_compressor.to(dtype=compute_dtype, device=training_args.device)

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
        # vision encoder and compressor use full-parameter training; re-enable them
        # explicitly since get_peft_model() froze everything at call time.
        _set_module_trainable(model.get_vision_encoder(), vision_trainable)
        _set_module_trainable(model.get_mm_projector(), projector_trainable)
        _set_module_trainable(getattr(model.get_model(), "token_compressor", None), compressor_trainable)
    else:
        _set_module_trainable(model.get_model(), llm_trainable)
        _set_module_trainable(model.get_vision_encoder(), vision_trainable)
        _set_module_trainable(model.get_mm_projector(), projector_trainable)
        _set_module_trainable(getattr(model.get_model(), "token_compressor", None), compressor_trainable)

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
    new_vocabulary_size = len(tokenizer)
    if new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
        if not llm_trainable:
            # Only the new rows need to be learned; old rows should stay frozen.
            # Keep requires_grad=True on the whole embed_tokens tensor so that
            # ZeRO-2 can assign optimizer state to it (a requires_grad parameter
            # with no optimizer state corrupts AllReduce buckets → NaN at step 2).
            # A backward hook zeros gradients for pre-existing rows so they are
            # not updated.  create_optimizer routes embed_tokens into the
            # compressor parameter group (see VideoLLaMA3Trainer.create_optimizer).
            # ZeRO-2 does NOT shard parameter or gradient tensors, so the hook
            # operates on the full gradient and works correctly.
            _old_vocab = old_vocabulary_size

            def _zero_old_embed_rows(grad, _ov=_old_vocab):
                g = grad.clone()
                g[:_ov].zero_()
                return g

            embed = model.get_input_embeddings()
            embed.weight.requires_grad_(True)
            embed.weight.register_hook(_zero_old_embed_rows)

            # lm_head: compression tokens are always masked with IGNORE_INDEX in
            # labels so their lm_head rows never receive gradients; freeze entirely
            # to keep it out of the optimizer and avoid any ZeRO state overhead.
            out_embed = model.get_output_embeddings()
            if out_embed is not None and out_embed.weight is not embed.weight:
                _set_module_trainable(out_embed, False)
    model.config.image_token_index = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    model.config.compression_start_token_id = tokenizer.convert_tokens_to_ids(COMPRESSION_START_TOKEN)
    model.config.compression_end_token_id = tokenizer.convert_tokens_to_ids(COMPRESSION_END_TOKEN)

    if data_args.force_image_size is not None:
        vision_encoder.image_processor.force_size = [data_args.force_image_size] * 2
        rank0_print(f"Force set image size to be {data_args.force_image_size}")
    vlprocessor = Videollama3Processor(vision_encoder.image_processor, tokenizer)

    assert data_args.use_batch_flattening, "Compressor training currently requires flattening mode (batch size 1 sequence)."
    assert model.config._attn_implementation == "flash_attention_2"
    assert version.parse(transformers.__version__) >= version.parse("4.44.0")
    data_module = make_compressor_data_module(vlprocessor=vlprocessor, data_args=data_args, output_dir=training_args.output_dir)

    trainer = VideoLLaMA3Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        use_dual_forward=training_args.use_dual_forward,
        partial_loss_weight=training_args.partial_loss_weight,
        full_loss_weight=training_args.full_loss_weight,
        **data_module,
    )


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
