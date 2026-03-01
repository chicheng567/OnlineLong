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
from videollama3.train.videollama3_chat_finetune_online import LazySupervisedDataset, set_seed


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
    sampling_fps: int = field(default=1)
    video_merge_size: int = field(default=1)
    max_videos: Optional[int] = field(default=None)

    compression_ratio: float = field(default=0.3)
    compression_window_size: int = field(default=5)
    num_inference_repeats: int = field(default=3)

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


def _load_annotation(annotation_path: str) -> List[Dict[str, Any]]:
    if annotation_path.endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with open(annotation_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Annotation must be a list: {annotation_path}")
    return data


def _resolve_video_path(video_field: Any, data_root: str) -> str:
    if isinstance(video_field, list):
        if len(video_field) != 1:
            raise ValueError(f"Only single video item is supported, got: {video_field}")
        video_rel = video_field[0]
    elif isinstance(video_field, str):
        video_rel = video_field
    else:
        raise ValueError(f"Unsupported video field type: {type(video_field)}")

    candidate = osp.join(data_root, video_rel)
    if osp.exists(candidate):
        return candidate
    base, ext = osp.splitext(candidate)
    if ext:
        raise FileNotFoundError(f"Video not found: {candidate}")
    for suffix in VIDEO_EXTENSIONS:
        with_ext = base + suffix
        if osp.exists(with_ext):
            return with_ext
    raise FileNotFoundError(f"Video not found: {candidate} (+ known extensions)")


def _build_dataset_items(
    meta_path: str,
    processor: Videollama3Processor,
    inference_args: InferenceArguments,
) -> List[Dict[str, Any]]:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise ValueError("meta_data_path must contain a dict of dataset configs.")

    data_args = SimpleNamespace(
        fps=inference_args.sampling_fps,
        max_frames=inference_args.max_frames,
        image_merge_size=1,
        video_merge_size=inference_args.video_merge_size,
        mm_max_length=30000,
        use_batch_flattening=True,
        dataset_cache_dir=None,
        data_folder=None,
        force_image_size=None,
    )
    samples: List[Dict[str, Any]] = []
    for dataset_name, dataset_cfg in meta.items():
        annotation_path = dataset_cfg.get("annotation")
        data_root = dataset_cfg.get("data_root")
        if not annotation_path or not data_root:
            raise ValueError(f"Dataset '{dataset_name}' must include annotation and data_root.")
        if not osp.exists(annotation_path):
            raise FileNotFoundError(f"Annotation not found: {annotation_path}")
        dataset = LazySupervisedDataset(
            vlprocessor=processor,
            data_path=[annotation_path],
            data_args=data_args,
            dataset_name=dataset_name,
            dataset_root=data_root,
            online_mode=bool(dataset_cfg.get("online_mode", False)),
            prefix_captioning=bool(dataset_cfg.get("prefix_captioning", False)),
        )
        for row_idx in range(len(dataset)):
            row = dataset.list_data_dict[row_idx]
            samples.append(
                {
                    "dataset": dataset_name,
                    "row_idx": row_idx,
                    "data_root": data_root,
                    "dataset_obj": dataset,
                    "row": row,
                }
            )
    return samples


def _to_model_input_from_dataset(
    sample_tensor_dict: Dict[str, torch.Tensor],
    device: str,
    model_dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    model_inputs = {
        "input_ids": sample_tensor_dict["input_ids"],
        "pixel_values": sample_tensor_dict["pixel_values"],
        "grid_sizes": sample_tensor_dict["grid_sizes"],
        "merge_sizes": sample_tensor_dict["merge_sizes"],
    }
    for key, value in model_inputs.items():
        if torch.is_tensor(value):
            model_inputs[key] = value.to(device)
    model_inputs["pixel_values"] = model_inputs["pixel_values"].to(dtype=model_dtype)
    if model_inputs["input_ids"].ndim == 1:
        model_inputs["input_ids"] = model_inputs["input_ids"].unsqueeze(0)
    return model_inputs


def _generate_with_compression(
    model: Videollama3Qwen2ForCausalLM,
    model_inputs: Dict[str, torch.Tensor],
    compression_parts: List[List[int]],
    generation_kwargs: Dict[str, Any],
) -> torch.LongTensor:
    input_ids = model_inputs["input_ids"]
    attention_mask = input_ids.ne(model.config.pad_token_id if model.config.pad_token_id is not None else -1).long()
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long).unsqueeze(0)

    (
        _,
        prepared_attention_mask,
        prepared_position_ids,
        _,
        inputs_embeds,
        _,
        _,
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        labels=None,
        pixel_values=model_inputs.get("pixel_values"),
        grid_sizes=model_inputs.get("grid_sizes"),
        merge_sizes=model_inputs.get("merge_sizes"),
        modals=["video"] * input_ids.shape[0],
        compression_parts=compression_parts,
    )

    # First-step multimodal compression is already applied above.
    # During autoregressive decode steps, forward() is called without compression_parts,
    # so temporarily disable trainable-mm-compressor checks.
    prev_trainable_mm_compressor = bool(getattr(model.config, "trainable_mm_compressor", False))
    model.config.trainable_mm_compressor = False
    try:
        output_ids = super(Videollama3Qwen2ForCausalLM, model).generate(
            position_ids=prepared_position_ids,
            attention_mask=prepared_attention_mask,
            inputs_embeds=inputs_embeds,
            **generation_kwargs,
        )
    finally:
        model.config.trainable_mm_compressor = prev_trainable_mm_compressor
    prompt_len = inputs_embeds.shape[1]
    if output_ids.shape[1] > prompt_len:
        return output_ids[:, prompt_len:]
    return output_ids


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

    config = Videollama3Qwen2Config.from_pretrained(model_args.model_name_or_path)
    attn_impl = model_args.mm_attn_implementation or "flash_attention_2"
    if attn_impl == "flash_attention_2" and not use_cuda:
        print("FlashAttention2 requested but CUDA unavailable. Switch to SDPA.")
        attn_impl = "sdpa"
    config._attn_implementation = attn_impl
    config.mm_attn_implementation = attn_impl
    config.use_token_compression = model_args.use_token_compression
    if model_args.vision_encoder is not None:
        config.vision_encoder = model_args.vision_encoder

    print(f"Loading model from {model_args.model_name_or_path}")
    model = Videollama3Qwen2ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = inference_args.use_cache
    model.config.trainable_mm_compressor = True
    if getattr(model.get_model(), "token_compressor", None) is None:
        raise RuntimeError("Loaded model has no token_compressor. Please use checkpoint trained by compressor script.")
    compressor = model.get_model().token_compressor
    model_window_size = int(getattr(compressor, "window_size", 0) or 0)
    effective_window_size = inference_args.compression_window_size
    if model_window_size > 0 and effective_window_size != model_window_size:
        print(
            f"[Warning] compression_window_size mismatch: "
            f"arg={effective_window_size}, model={model_window_size}. "
            f"Using model window size."
        )
        effective_window_size = model_window_size

    device = inference_args.device or ("cuda" if use_cuda else "cpu")
    model.to(device)
    model.eval()

    processor = Videollama3Processor.from_pretrained(model_args.model_name_or_path, trust_remote_code=False)
    checkpoint_image_processor = processor.image_processor
    vision_image_processor = model.get_vision_encoder().image_processor
    # Use the model vision image_processor to keep keys compatible with Videollama3Processor
    # (expects grid_sizes/merge_sizes), but preserve checkpoint force_size if provided.
    if getattr(checkpoint_image_processor, "force_size", None) is not None:
        vision_image_processor.force_size = checkpoint_image_processor.force_size
    processor.image_processor = vision_image_processor

    image_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
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

    samples = _build_dataset_items(
        meta_path=inference_args.meta_data_path,
        processor=processor,
        inference_args=inference_args,
    )
    if inference_args.max_videos is not None:
        samples = samples[: inference_args.max_videos]
    if not samples:
        raise RuntimeError("No valid samples found from metadata/annotations.")
    print(f"Prepared {len(samples)} samples for inference.")

    results: List[Dict[str, Any]] = []
    progress = tqdm(samples, desc="Compressor inference", unit="sample")

    for sample_idx, sample in enumerate(progress):
        sample_tensor_dict = sample["dataset_obj"][sample["row_idx"]]
        model_inputs = _to_model_input_from_dataset(
            sample_tensor_dict=sample_tensor_dict,
            device=device,
            model_dtype=model.dtype,
        )

        row = sample["row"]
        first_human = next((x for x in row.get("conversations", []) if x.get("from") == "human"), None)
        prompt = first_human.get("value", "").strip() if first_human else ""
        video_path = _resolve_video_path(row.get("video"), sample["data_root"])
        grid_sizes = model_inputs["grid_sizes"]
        if grid_sizes.ndim != 2 or grid_sizes.shape[1] < 1:
            raise RuntimeError(f"Unexpected grid_sizes shape: {tuple(grid_sizes.shape)}")
        num_frames = int(grid_sizes[:, 0].sum().item())
        if num_frames <= 0:
            raise RuntimeError(f"Invalid num_frames={num_frames} from grid_sizes for sample {sample_idx}.")

        total_vision_tokens = int((model_inputs["input_ids"] == image_token_id).sum().item())
        if total_vision_tokens == 0:
            raise RuntimeError(f"No vision tokens found for sample {sample_idx}: {video_path}")
        if total_vision_tokens % num_frames != 0:
            raise RuntimeError(
                f"Vision tokens/frames mismatch for sample {sample_idx}: "
                f"{total_vision_tokens} tokens vs {num_frames} frames."
            )
        tokens_per_frame = total_vision_tokens // num_frames
        expected_tokens_per_frame = int(getattr(compressor, "compress_image_wh", 0) or 0)
        if expected_tokens_per_frame > 0 and tokens_per_frame != expected_tokens_per_frame:
            raise RuntimeError(
                f"Unexpected tokens_per_frame for sample {sample_idx}: got {tokens_per_frame}, "
                f"expected {expected_tokens_per_frame}. "
                f"video={video_path}, merge_size={inference_args.video_merge_size}, "
                f"force_size={getattr(processor.image_processor, 'force_size', None)}"
            )

        repeat_outputs: List[Dict[str, Any]] = []
        for repeat_idx in range(inference_args.num_inference_repeats):
            repeat_rng = random.Random(seed + sample_idx * 1009 + repeat_idx * 9173)
            compression_parts = select_compression_parts(
                total_frames=num_frames,
                total_vision_tokens=total_vision_tokens,
                ratio=inference_args.compression_ratio,
                window_size=effective_window_size,
                rng=repeat_rng,
            )
            with torch.no_grad():
                new_tokens = _generate_with_compression(
                    model=model,
                    model_inputs=model_inputs,
                    compression_parts=compression_parts,
                    generation_kwargs=generation_kwargs,
                )
            text = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
            repeat_outputs.append(
                {
                    "repeat_idx": repeat_idx,
                    "compression_parts": compression_parts,
                    "caption": text,
                }
            )

        results.append(
            {
                "dataset": sample["dataset"],
                "sample_index": sample["row_idx"],
                "video_path": video_path,
                "prompt": prompt,
                "num_frames": num_frames,
                "total_vision_tokens": total_vision_tokens,
                "repeat_outputs": repeat_outputs,
            }
        )

    os.makedirs(osp.dirname(inference_args.output_file) or ".", exist_ok=True)
    with open(inference_args.output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} items to {inference_args.output_file}")


if __name__ == "__main__":
    captioning()
