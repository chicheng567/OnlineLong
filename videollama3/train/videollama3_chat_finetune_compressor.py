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
from transformers import TrainerCallback

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
from videollama3.train.videollama3_chat_finetune_online import (  # noqa: E402
    ConcatDatasetWithLengths,
    LazySupervisedDataset,
    LoggingCallback,
    _is_trainable_lr,
    _set_module_trainable,
    find_all_linear_names,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    safe_save_model_for_hf_trainer,
    set_seed,
)
from videollama3.train.videollama3_trainer import VideoLLaMA3Trainer  # noqa: E402
from functools import partial
torch.load = partial(torch.load, weights_only=False)
try:
    from deepspeed.runtime.fp16.loss_scaler import LossScaler
    from deepspeed.runtime.zero.config import ZeroStageEnum
    torch.serialization.add_safe_globals([LossScaler, ZeroStageEnum])
except ImportError:
    pass

def rank0_print(*args):
    if local_rank == 0:
        message = " ".join(str(arg) for arg in args)
        print(message)
        if logging.getLogger().hasHandlers():
            logging.info(message)


def int_with_none(value):
    if value == "None":
        return None
    return int(value)


def _select_non_overlapping_windows(
    num_frames: int,
    window_size: int,
    target_frames: int,
) -> List[int]:
    if num_frames < window_size:
        return []
    
    target_windows = max(1, round(target_frames / window_size))
    
    max_possible_windows = num_frames // window_size
    target_windows = min(target_windows, max_possible_windows)
    
    if target_windows == 0:
        return []

    max_start = num_frames - window_size

    if target_windows == 1:
        return [max_start // 2]
    
    step = max_start / (target_windows - 1)
    starts = [int(round(i * step)) for i in range(target_windows)]
    
    return starts


def select_compression_parts(
    total_frames: int,
    total_vision_tokens: int,
    ratio: float,
    window_size: int,
) -> List[List[int]]:
    if total_frames <= 0 or ratio <= 0 or window_size <= 0 or total_frames < window_size:
        return []
    assert total_vision_tokens % total_frames == 0, f"Total vision tokens {total_vision_tokens} should be divisible by total frames {total_frames}."
    tokens_per_frame = total_vision_tokens // total_frames
    target_frames = max(window_size, int(round(total_frames * ratio)))
    starts = _select_non_overlapping_windows(total_frames, window_size, target_frames)
    starts.sort()
    selected_idx = []
    for start in starts:
        assert start + window_size <= total_frames, f"Selected window [{start}, {start + window_size}) exceeds total frames {total_frames}."
        selected_idx.append([start * tokens_per_frame, (start + window_size) * tokens_per_frame])
    
    return selected_idx


def count_video_frames_in_messages(messages: List[Dict]) -> int:
    total_frames = 0
    for message in messages:
        if message.get("role") != "user":
            continue
        for content in message.get("content", []):
            if isinstance(content, dict) and content.get("type") == "video":
                total_frames += int(content.get("num_frames", 0))
    return total_frames

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
    #compression decoder args
    compressor_decoder_layers: int = field(default=0)
    compression_mse_loss_weight: float = field(default=0.0)
    upsample_factor_per_decoder: int = field(default=3)

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


class StepInferenceCallback(TrainerCallback):
    def __init__(
        self,
        train_dataset,
        data_collator,
        tokenizer,
        max_new_tokens: int = 64,
        do_sample: bool = False,
    ):
        self.train_dataset = train_dataset
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

    @staticmethod
    def _trim_text(text: str, max_chars: int = 400) -> str:
        if text is None:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + " ...<truncated>"

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_local_process_zero:
            return control
        if state.global_step <= 0 or len(self.train_dataset) == 0:
            return control

        model = kwargs.get("model", None)
        if model is None:
            return control

        idx = random.randint(0, len(self.train_dataset) - 1)
        sample = self.train_dataset[idx]
        batch = self.data_collator([sample])
        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]

        target_positions = torch.nonzero(labels != -100, as_tuple=False).squeeze(-1)
        if target_positions.numel() == 0:
            logging.info("[step-infer] step=%s sample_idx=%s skipped (no supervised target tokens).", state.global_step, idx)
            return control

        prompt_len = int(target_positions[0].item())
        if prompt_len <= 0:
            logging.info("[step-infer] step=%s sample_idx=%s skipped (invalid prompt length).", state.global_step, idx)
            return control

        try:
            model_param = next(model.parameters())
            device = model_param.device
            model_dtype = model_param.dtype

            prompt_ids = input_ids[:prompt_len].unsqueeze(0).to(device=device)
            attention_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=device)
            position_ids = torch.arange(prompt_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)

            generate_kwargs = dict(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=batch["pixel_values"].to(device=device, dtype=model_dtype),
                grid_sizes=batch["grid_sizes"].to(device=device),
                merge_sizes=batch["merge_sizes"].to(device=device),
                modals=batch["modals"],
                compression_parts=batch.get("compression_parts", []),
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            was_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    generated = model.generate(**generate_kwargs)
            finally:
                if was_training:
                    model.train()

            pred_ids = generated[0, prompt_len:].detach().cpu()
            gt_ids = labels[target_positions].detach().cpu()

            prompt_tail = self.tokenizer.decode(input_ids[max(0, prompt_len - 256):prompt_len], skip_special_tokens=False)
            pred_text = self.tokenizer.decode(pred_ids, skip_special_tokens=False)
            gt_text = self.tokenizer.decode(gt_ids, skip_special_tokens=False)

            logging.info(
                "[step-infer] step=%s sample_idx=%s prompt_tokens=%s pred_tokens=%s\n[prompt-tail]\n%s\n[ground-truth]\n%s\n[prediction]\n%s",
                state.global_step,
                idx,
                prompt_len,
                int(pred_ids.numel()),
                self._trim_text(prompt_tail),
                self._trim_text(gt_text),
                self._trim_text(pred_text),
            )
        except Exception:
            logging.exception("[step-infer] failed at step=%s sample_idx=%s", state.global_step, idx)

        return control


class CompressorLazySupervisedDataset(LazySupervisedDataset):
    def __init__(self, *args, compression_ratio: float = 0.3, compression_window_size: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.compression_ratio = compression_ratio
        self.compression_window_size = compression_window_size
        assert compression_window_size - 2 > 1, "Compression window size cannot be less than 3."
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = self.list_data_dict[i]
        try:
            if self.online_mode:
                modal, images, messages, merge_size = self._convert_online_video(sample)
            else:
                if "stream" in sample and sample["stream"]:
                    raise NotImplementedError("Online stream data is not supported in compressor training yet.")
                    modal, images, messages, merge_size = self._convert_stream(sample)
                else:
                    modal, images, messages, merge_size = self._convert_normal(sample)
            assert modal == "video", "Compressor training currently only supports video data."
            data_dict = self.vlprocessor(
                images=images,
                text=messages,
                merge_size=merge_size,
                return_labels=self.return_label,
                return_tensors="pt",
            )
            data_dict["modals"] = [modal] * len(images)
            if modal == "video":
                image_token_id = self.vlprocessor.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
                # NOTE: data_dict["pixel_values"].shape[0] will be 4 times than total_vision token because of 4x4 merging.
                total_vision_tokens = int((data_dict["input_ids"] == image_token_id).sum().item())
                compression_part = select_compression_parts(
                    total_frames=len(images[0]),
                    total_vision_tokens=total_vision_tokens,
                    ratio=self.compression_ratio,
                    window_size=self.compression_window_size,
                )
            else:
                compression_part = []
            data_dict["compression_parts"] = compression_part
        except Exception:
            backup_idx = random.randint(0, len(self.list_data_dict) - 1)
            logger.exception("Failed to process sample %s. Fallback index: %s.", i, backup_idx)
            return self.__getitem__(backup_idx)
        return data_dict


@dataclass
class DataCollatorWithCompressor:
    vlprocessor: transformers.ProcessorMixin
    compression_ratio: float = 0.3
    compression_window_size: int = 3

    def __call__(self, instances: Sequence[Dict], separator_id=-100) -> Dict[str, torch.Tensor]:
        # input_ids: List[torch.Tensor], labels: List[torch.Tensor]
        input_ids, labels, compression_parts = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "compression_parts"))
        new_input_ids = []
        new_labels = []
        position_ids = []
        new_compression_parts: List[List[int]] = []
        accumulated_length = 0
        image_token_id = self.vlprocessor.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        for sample_idx in range(0, len(input_ids)):
            if input_ids[sample_idx].shape[0] > self.vlprocessor.tokenizer.model_max_length:
                warnings.warn(
                    f"Sample {sample_idx} length {input_ids[sample_idx].shape[0]} exceeds model max length "
                    f"{self.vlprocessor.tokenizer.model_max_length}. It will be truncated."
                )
            capped_ids = input_ids[sample_idx][: self.vlprocessor.tokenizer.model_max_length]
            capped_labels = labels[sample_idx][: self.vlprocessor.tokenizer.model_max_length]
            capped_labels[0] = separator_id
            new_input_ids.append(capped_ids)
            new_labels.append(capped_labels)
            position_ids.append(torch.arange(len(capped_ids), dtype=torch.long))
            new_compression_parts.extend([[startend[0] + accumulated_length, startend[1] + accumulated_length] for startend in compression_parts[sample_idx]])
            image_token_count = int((capped_ids == image_token_id).sum().item())
            accumulated_length += image_token_count
            
        flat_input_ids = torch.cat(new_input_ids)
        flat_labels = torch.cat(new_labels)
        flat_position_ids = torch.cat(position_ids)

        batch = dict(
            input_ids=flat_input_ids.unsqueeze(0),
            labels=flat_labels.unsqueeze(0),
            position_ids=flat_position_ids.unsqueeze(0),
        )
        batch["pixel_values"] = torch.cat([x["pixel_values"] for x in instances])
        batch["grid_sizes"] = torch.cat([x["grid_sizes"] for x in instances])
        batch["merge_sizes"] = torch.cat([x["merge_sizes"] for x in instances])
        batch["modals"] = sum([x["modals"] for x in instances], [])
        batch["compression_parts"] = new_compression_parts
        return batch


def make_compressor_data_module(vlprocessor: transformers.ProcessorMixin, data_args: DataArguments) -> Dict:
    if data_args.multi_dataset:
        rank0_print("Use meta file to control datasets loading. Data path will use as meta path")
        ds_collection = dict()
        meta_path = data_args.data_path[0]
        ds_collection.update(json.loads(open(meta_path).read()))
        collected_datasets = []
        for dataset_name, dataset_cfg in ds_collection.items():
            collected_datasets.append(
                CompressorLazySupervisedDataset(
                    vlprocessor=vlprocessor,
                    data_path=[dataset_cfg["annotation"]],
                    data_args=data_args,
                    dataset_name=dataset_name,
                    dataset_root=dataset_cfg["data_root"],
                    online_mode=dataset_cfg["online_mode"],
                    prefix_captioning=dataset_cfg.get("prefix_captioning", False),
                    compression_ratio=data_args.compression_ratio,
                    compression_window_size=data_args.compression_window_size,
                )
            )
        train_dataset = ConcatDatasetWithLengths(collected_datasets)
    else:
        train_dataset = CompressorLazySupervisedDataset(
            vlprocessor=vlprocessor,
            data_path=data_args.data_path,
            data_args=data_args,
            compression_ratio=data_args.compression_ratio,
            compression_window_size=data_args.compression_window_size,
        )

    data_collator = DataCollatorWithCompressor(
        vlprocessor=vlprocessor,
        compression_ratio=data_args.compression_ratio,
        compression_window_size=data_args.compression_window_size,
    )
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


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
        #compression decoder
        "decoder_layers": model_args.compressor_decoder_layers,
        "upsample_factor_per_decoder": model_args.upsample_factor_per_decoder,
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

    tokenizer_path = model_args.tokenizer_name_or_path or model_args.model_name_or_path
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
    model.config.compression_mse_loss_weight = model_args.compression_mse_loss_weight

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

    new_tokens = tokenizer.add_tokens([COMPRESSION_START_TOKEN, COMPRESSION_END_TOKEN], special_tokens=True)
    if new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
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
    data_module = make_compressor_data_module(vlprocessor=vlprocessor, data_args=data_args)

    callbacks = [LoggingCallback()]
    if training_args.step_infer_enabled:
        callbacks.append(
            StepInferenceCallback(
                train_dataset=data_module["train_dataset"],
                data_collator=data_module["data_collator"],
                tokenizer=tokenizer,
                max_new_tokens=training_args.step_infer_max_new_tokens,
                do_sample=training_args.step_infer_do_sample,
            )
        )
        rank0_print(
            f"Step inference is enabled: max_new_tokens={training_args.step_infer_max_new_tokens}, "
            f"do_sample={training_args.step_infer_do_sample}"
        )

    trainer = VideoLLaMA3Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=callbacks,
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
