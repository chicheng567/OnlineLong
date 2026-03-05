import json
import os
import os.path as osp
import random
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
import transformers
from tqdm import tqdm

from videollama3.constants import DEFAULT_IMAGE_TOKEN
from videollama3.model import Videollama3Qwen2Config, Videollama3Qwen2ForCausalLM
from videollama3.model.processor import Videollama3Processor
from videollama3.train.videollama3_chat_finetune_compressor import (
    select_compression_parts,
)
from videollama3.train.videollama3_chat_finetune_compressor import CompressorLazySupervisedDataset
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="pretrained_models/videollama3_7b_local")
    vision_encoder: Optional[str] = field(default=None)
    mm_attn_implementation: Optional[str] = field(default="flash_attention_2")
    use_token_compression: Optional[bool] = field(default=True)


@dataclass
class InferenceArguments:
    meta_data_path: str = field(default="anno_data/finetune_online.json")
    output_file: str = field(default="captioning_results.jsonl")
    max_frames: int = field(default=200)
    fps: int = field(default=1)
    max_videos: Optional[int] = field(default=None)
    num_inference_repeats: int = field(default=3)
    # Merge size
    video_merge_size: int = field(default=2)
    image_merge_size: int = field(default=1)
    # Compression settings
    compression_ratio: float = field(default=0.5)
    compression_window_size: int = field(default=10)
    dataset_cache_dir: Optional[str] = field(default=None)
    max_new_tokens: int = field(default=300)
    min_length: int = field(default=0)
    do_sample: bool = field(default=False)
    temperature: float = field(default=1.0)
    top_k: int = field(default=50)
    top_p: float = field(default=1.0)
    num_beams: int = field(default=1)
    num_return_sequences: int = field(default=1)
    no_repeat_ngram_size: int = field(default=0)
    repetition_penalty: float = field(default=1.0)
    length_penalty: float = field(default=1.0)
    early_stopping: bool = field(default=False)
    pad_token_id: Optional[int] = field(default=None)
    bos_token_id: Optional[int] = field(default=None)
    eos_token_id: Optional[int] = field(default=None)
    use_cache: bool = field(default=True)
    device: Optional[str] = field(default="cuda")
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    seed: Optional[int] = field(default=42)
    generation_kwargs: Optional[Dict[str, Any]] = field(default=None)

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across PyTorch, NumPy, and Python."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def get_concated_dataset(inference_args: InferenceArguments, vlprocessor: Videollama3Processor):
    meta_data = inference_args.meta_data_path
    ds_collection = dict()
    ds_collection.update(json.load(open(meta_data, "r", encoding="utf-8")))
    collected_datasets = []
    for dataset_name, dataset_cfg in ds_collection.items():
        collected_datasets.append(
            CompressorLazySupervisedDataset(
                vlprocessor=vlprocessor,
                data_path=[dataset_cfg["annotation"]],
                data_args=inference_args,
                dataset_name=dataset_name,
                dataset_root=dataset_cfg["data_root"],
                online_mode=dataset_cfg["online_mode"],
                prefix_captioning=dataset_cfg.get("prefix_captioning", False),
                compression_ratio=inference_args.compression_ratio,
                compression_window_size=inference_args.compression_window_size,
                return_label=False,
            )
        )
    concate_dataset = torch.utils.data.ConcatDataset(collected_datasets)
    return concate_dataset

def captioning() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, InferenceArguments))
    model_args, inference_args = parser.parse_args_into_dataclasses()

    if inference_args.num_inference_repeats < 1:
        raise ValueError("num_inference_repeats must be >= 1.")

    seed = inference_args.seed if inference_args.seed is not None else 42
    set_seed(seed)

    use_cuda = torch.cuda.is_available()
    if not use_cuda and (inference_args.bf16 or inference_args.fp16):
        print("Half precision on CPU is disabled; fallback to fp32.")
        inference_args.bf16 = False
        inference_args.fp16 = False

    if inference_args.bf16:
        compute_dtype = torch.bfloat16
    elif inference_args.fp16:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    attn_impl = model_args.mm_attn_implementation or "flash_attention_2"
    if attn_impl == "flash_attention_2" and not use_cuda:
        print("FlashAttention2 requested but CUDA unavailable. Switch to SDPA.")
        attn_impl = "sdpa"

    print(f"Loading model from {model_args.model_name_or_path}")
    model = Videollama3Qwen2ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=compute_dtype,
    )
    if model.config.use_cache == None:
        model.config.use_cache = inference_args.use_cache
    if getattr(model.get_model(), "token_compressor", None) is None:
        raise RuntimeError("Loaded model has no token_compressor. Please use checkpoint trained by compressor script.")
    device = inference_args.device or ("cuda" if use_cuda else "cpu")
    model.to(device)
    model.eval()
    processor = Videollama3Processor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=False,
        fix_mistral_regex=True,
    )
    
    generation_kwargs = {
        "max_new_tokens": inference_args.max_new_tokens,
        "min_length": inference_args.min_length,
        "do_sample": inference_args.do_sample,
        "temperature": inference_args.temperature,
        "top_k": inference_args.top_k,
        "top_p": inference_args.top_p,
        "num_beams": inference_args.num_beams,
        "num_return_sequences": inference_args.num_return_sequences,
        "no_repeat_ngram_size": inference_args.no_repeat_ngram_size,
        "repetition_penalty": inference_args.repetition_penalty,
        "length_penalty": inference_args.length_penalty,
        "early_stopping": inference_args.early_stopping,
        "pad_token_id": inference_args.pad_token_id or processor.tokenizer.pad_token_id,
        "bos_token_id": inference_args.bos_token_id or processor.tokenizer.bos_token_id,
        "eos_token_id": inference_args.eos_token_id or processor.tokenizer.eos_token_id,
    }
    if inference_args.generation_kwargs:
        generation_kwargs.update(inference_args.generation_kwargs)
    
    inf_set = get_concated_dataset(
        inference_args=inference_args,
        vlprocessor=processor,
    )
    if inference_args.max_videos is not None:
        inf_set = inf_set[: inference_args.max_videos]
    
    print(f"Prepared {len(inf_set)} samples for inference.")
    
    progress = tqdm(inf_set, desc="Compressor inference", unit="sample")

    for sample_idx, sample in enumerate(progress):
        #TODO: modify model.generate to support data input with compression
        sample = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
        sample["pixel_values"] = sample["pixel_values"].to(compute_dtype)
        output = model.generate(
            **sample,
            **generation_kwargs,
        )
        output_text = processor.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
        with open(inference_args.output_file, "a", encoding="utf-8") as f:
            json_line = json.dumps({
                "sample_idx": sample_idx,
                "output": output_text,
            }, ensure_ascii=False)
            f.write(json_line + "\n")

if __name__ == "__main__":
    captioning()
