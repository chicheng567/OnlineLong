# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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
from copy import deepcopy
import math
import copy
import json
import os
import pathlib
import random
import re
import sys
import warnings
from packaging import version
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import numpy as np
# torch-related packages
# NOTE: torch must be imported before transformers. Otherwise, `Segmentation fault (core dumped)` will occur.
import torch
import transformers
from packaging import version
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from transformers import TrainerCallback
import logging
from transformers.utils.logging import (
    enable_default_handler,
    enable_explicit_format,
    set_verbosity,
)
sys.path.append('./')

from videollama3.constants import (IGNORE_INDEX,
    NUM_FRAMES, DEFAULT_IMAGE_TOKEN, STREAM_MAX_FRAMES,
    STREAM_START_TOKEN, STREAM_END_TOKEN)
from videollama3.mm_utils import (load_images, load_video, read_frames_decord, process_qa, preprocess_videollama3)
from videollama3.model import *
from videollama3.train.videollama3_trainer import (
    VideoLLaMA3Trainer, find_all_linear_names, get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer)
from videollama3.model.processor import Videollama3Processor
from videollama3.train.data import common as _data_common
from videollama3.train.data.common import (
    int_with_none, rank0_print, set_seed, _is_trainable_lr, _set_module_trainable,
)
from videollama3.train.data.supervised import (
    ConcatDatasetWithLengths, DataCollatorForSupervisedDataset,
    DataCollatorWithFlatteningForSupervisedDataset, LazySupervisedDataset,
    make_flattening_supervised_data_module, make_supervised_data_module,
)

# NOTE: fast tokenizer warning issue: https://github.com/huggingface/transformers/issues/5486
os.environ["TOKENIZERS_PARALLELISM"] = "true"

local_rank = None
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    # LLM Arguments
    model_name_or_path: Optional[str] = field(default="pretrained_models/videollama3_7b_local")
    tokenizer_name_or_path: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v1", metadata={"help": "Version of the conversation template."})
    freeze_llm: bool = field(default=False, metadata={"help": "Deprecated. Use llm_lr=0 to freeze the LLM backbone."})
    freeze_vision_encoder: bool = field(default=False, metadata={"help": "Deprecated. Use vision_encoder_lr=0 to freeze the vision encoder."})
    freeze_mlp: bool = field(default=False, metadata={"help": "Deprecated. Use mm_projector_lr=0 to freeze the multi-modal projector."})
    # Connector Arguments
    mm_projector_type: Optional[str] = field(default='linear')
    # Vision tower Arguments
    vision_encoder: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_attn_implementation: Optional[str] = field(default="flash_attention_2") #always use flash_attention_2
    # Token downsampling Arguments
    use_token_compression: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    # Path Arguments
    data_path: List[str] = field(default=None, metadata={"help": "Path to the training data."})
    # Loading Arguments
    fps: Optional[int] = field(default=None)
    max_frames: Optional[int_with_none] = field(default=200)
    multi_dataset: bool = field(default=False, metadata={"help": "Use meta file to control datasets loading."})
    # Preprocess Arguments
    image_merge_size: Optional[int] = field(default=1)
    video_merge_size: Optional[int] = field(default=1)
    mm_max_length: Optional[int] = field(default=10240)
    image_aspect_ratio: str = 'square'
    use_batch_flattening: bool = field(default=True, metadata={"help": "Whether to flatten the in-batch sequences of variable lengths."})
    dataset_cache_dir: Optional[str] = field(default=None)
    force_image_size: Optional[int] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # shut auto processing (_remove_unused_columns) of transformers Trainer
    remove_unused_columns: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    # Training learning rate Arguments
    vision_encoder_lr: Optional[float] = None
    mm_projector_lr: Optional[float] = None
    compressor_lr: Optional[float] = None
    llm_lr: Optional[float] = None
    # Training Data Arguments
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # Lora or Quant Arguments
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


class LoggingCallback(TrainerCallback):
    """Custom callback to log training metrics to file."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics to file when trainer logs."""
        if state.is_local_process_zero and logs is not None:
            # Format metrics for logging
            metrics_str = f"Step {state.global_step}"
            if "loss" in logs:
                metrics_str += f" | Loss: {logs['loss']:.4f}"
            if "learning_rate" in logs:
                metrics_str += f" | LR: {logs['learning_rate']:.2e}"
            if "grad_norm" in logs:
                metrics_str += f" | Grad Norm: {logs['grad_norm']:.4f}"
            if "epoch" in logs:
                metrics_str += f" | Epoch: {logs['epoch']:.2f}"

            logging.info(metrics_str)


def train(attn_implementation=None):
    global local_rank
    set_seed(42)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    log_file = os.path.join(training_args.output_dir, "training.log")
    error_log_file = os.path.join(training_args.output_dir, "training_errors.log")
    os.makedirs(training_args.output_dir, exist_ok=True)

    log_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(log_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    error_handler = logging.FileHandler(error_log_file, mode='a')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(log_formatter)
    root_logger.addHandler(error_handler)

    logging.captureWarnings(True)
    local_rank = training_args.local_rank
    _data_common.local_rank = local_rank

    if local_rank == 0:
        print('------model args------')
        print(model_args)
        print('------data args------')
        print(data_args)
        print('------training args------')
        print(training_args)

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    model_args.torch_dtype = compute_dtype

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type, # {'fp4', 'nf4'}
                bnb_4bit_quant_storage=compute_dtype,
            )
        ))

    config = Videollama3Qwen2Config.from_pretrained(model_args.model_name_or_path)

    config._attn_implementation = attn_implementation
    config.use_token_compression = model_args.use_token_compression

    config.vision_encoder = model_args.vision_encoder
    model = Videollama3Qwen2ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=compute_dtype,
        do_sample=True,
        **bnb_model_from_pretrained_args
    )
    model.config.use_cache = False
    

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
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
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    # Use local qwen2 tokenizer instead of transformers AutoTokenizer
    try:
        from qwen2 import Qwen2TokenizerFast
        tokenizer = Qwen2TokenizerFast.from_pretrained(
            model_args.tokenizer_name_or_path,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )
    except ImportError:
        from qwen2 import Qwen2Tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(
            model_args.tokenizer_name_or_path,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    if model_args.vision_encoder is not None:
        # initialize vision encoder + multi-modal projector
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_encoder = model.get_vision_encoder()
        vision_encoder.to(dtype=compute_dtype, device=training_args.device)

        mm_max_length = data_args.mm_max_length
        vision_encoder.image_processor.max_tokens = mm_max_length

        mm_projector = model.get_mm_projector()
        mm_projector.to(dtype=compute_dtype if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.is_multimodal = True

        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        # decoupled learning rate
        model.config.llm_lr = training_args.llm_lr
        model.config.vision_encoder_lr = training_args.vision_encoder_lr
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.compressor_lr = training_args.compressor_lr

        llm_trainable = _is_trainable_lr(model.config.llm_lr)
        vision_trainable = _is_trainable_lr(model.config.vision_encoder_lr)
        projector_trainable = _is_trainable_lr(model.config.mm_projector_lr)
        compressor_trainable = _is_trainable_lr(model.config.compressor_lr)

        # Unified control: module with lr == 0 (or lr is not set) will be frozen.
        _set_module_trainable(model.get_model(), llm_trainable)
        _set_module_trainable(model.get_vision_encoder(), vision_trainable)
        _set_module_trainable(model.get_mm_projector(), projector_trainable)
        _set_module_trainable(getattr(model.get_model(), "token_compressor", None), compressor_trainable)

        # Fail fast if all params are frozen by LR settings.
        trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable_param_count == 0:
            raise RuntimeError(
                "No trainable parameters found. "
                "Please set at least one of llm_lr / vision_encoder_lr / mm_projector_lr / compressor_lr > 0."
            )

        model.config.max_frames = getattr(data_args, 'max_frames', NUM_FRAMES)
        model.config.image_aspect_ratio = data_args.image_aspect_ratio if 'avt' not in model_args.vision_encoder else 'avt'

        # NOTE: complement data_args via model hyperparameters
        # 1. acquire image size
        model.config.image_size = data_args.image_size = vision_encoder.image_size
        # 2. calculate the number of tokens in the image
        model.config.image_token_length = data_args.image_token_length = mm_projector.cal_proj_size(vision_encoder.num_patches_per_side)
        # 3. check if alignment
        model.config.is_alignment = training_args.is_alignment = data_args.is_alignment = (
            _is_trainable_lr(model.config.mm_projector_lr) and
            not _is_trainable_lr(model.config.llm_lr) and
            not _is_trainable_lr(model.config.vision_encoder_lr) and
            not _is_trainable_lr(model.config.compressor_lr)
        )
        # 4. set spatial merge size as default
        new_tokens = tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN, STREAM_START_TOKEN, STREAM_END_TOKEN], special_tokens=True)
        model.config.image_token_index = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        if data_args.force_image_size is not None:
            vision_encoder.image_processor.force_size = [data_args.force_image_size] * 2
            rank0_print(f"Force set image size to be {data_args.force_image_size}")
        vlprocessor = Videollama3Processor(vision_encoder.image_processor, tokenizer)
    if model_args.freeze_llm or model_args.freeze_vision_encoder or model_args.freeze_mlp:
        rank0_print("Warning: freeze_* arguments are deprecated. Please use module lr == 0 to freeze.")
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    if local_rank == 0:
        print("Model config:", model.config)
        print("Current model:", model)
        
    if data_args.use_batch_flattening:
        rank0_print('You are using flattening operation to flatten the entire mini batch into a single sequence')
        assert model.config._attn_implementation == 'flash_attention_2'
        assert version.parse(transformers.__version__) >= version.parse("4.44.0")
        data_module = make_flattening_supervised_data_module(vlprocessor=vlprocessor, data_args=data_args)
    else:
        data_module = make_supervised_data_module(vlprocessor=vlprocessor, data_args=data_args)

    # select a Trainer
    trainer = VideoLLaMA3Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[LoggingCallback()],
        **data_module
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
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            vlprocessor.save_pretrained(training_args.output_dir)
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        if trainer.args.should_save:
            vlprocessor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
