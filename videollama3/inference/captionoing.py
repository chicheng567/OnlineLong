import json
import os
import os.path as osp
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import transformers
from torch.utils.data import ConcatDataset, Dataset
from videollama3.model.processor import Videollama3Processor
from tqdm import tqdm

from videollama3.mm_utils import LoadVideoWithClips
from videollama3.model import Videollama3Qwen2Config, Videollama3Qwen2ForCausalLM
from videollama3.train.videollama3_chat_finetune_online import set_seed
@dataclass
class ModelArguments:
    # LLM Arguments
    model_name_or_path: Optional[str] = field(default="pretrained_models/videollama3_7b_local")
    tokenizer_name_or_path: Optional[str] = field(default=None)
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
class InferenceArguments:
    #Data arguments
    meta_data_path: Optional[str] = field(default="anno_data/finetune_online.json")
    #ouptut file
    output_file: Optional[str] = field(default="captioning_results.jsonl")
    # Generation size controls
    max_new_tokens: int = field(default=128)  # preferred over max_length for HF generate
    max_length: Optional[int] = field(default=None)  # legacy; if set, may be used instead of max_new_tokens
    min_length: int = field(default=0)
    # Sampling / search strategy
    do_sample: bool = field(default=False)
    temperature: float = field(default=1.0)
    top_k: int = field(default=50)
    top_p: float = field(default=1.0)
    num_beams: int = field(default=1)
    num_return_sequences: int = field(default=1)
    no_repeat_ngram_size: int = field(default=0)

    # Penalties & length handling
    repetition_penalty: float = field(default=1.0)
    length_penalty: float = field(default=1.0)
    early_stopping: bool = field(default=False)

    # Tokens & caching
    pad_token_id: Optional[int] = field(default=None)
    bos_token_id: Optional[int] = field(default=None)
    eos_token_id: Optional[int] = field(default=None)
    use_cache: bool = field(default=True)

    # Return options from generate()
    return_dict_in_generate: bool = field(default=False)
    output_scores: bool = field(default=False)
    # Misc
    device: Optional[str] = field(default=None)  # e.g., "cuda" or "cpu"; if None, infer from model
    bf16: bool = field(default=True)  # whether to use half precision for generation
    fp16: bool = field(default=False)
    seed: Optional[int] = field(default=None)  # if set, will be used to call set_seed(seed)
    clip_length: int = field(default=300)
    sampling_fps: int = field(default=1)
    max_videos: Optional[int] = field(default=None)
    
    # Extra passthrough dict for any non-explicit HuggingFace generate kwargs
    generation_kwargs: Optional[Dict[str, object]] = field(default=None)

class LazySupervisedDatasetForCaptioning(Dataset):
    def __init__(
        self,
        annotation_path: str,
        data_root: Optional[str],
        fps: int,
        max_video_length: int = 300,
    ):
        self.annotation_path = annotation_path
        self.data_root = data_root
        self.max_video_length = max_video_length
        self.fps = fps
        if data_root is not None:
            if not os.path.exists(data_root):
                raise FileNotFoundError(f"Dataset root {data_root} not exists!")
        with open(annotation_path, 'r') as f:
            datas = json.load(f)
        self.video_list = set()
        for data in datas:
            video_id = data.get("video")
            if not video_id:
                continue
            self.video_list.add(video_id)
        self.video_list = list(self.video_list)
        if not self.video_list:
            raise ValueError(f"No videos found in {annotation_path}")
        print(f"Loaded {len(self.video_list)} videos from {annotation_path} (root={self.data_root or 'cwd'})")
    
    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        rel_path = self.video_list[index]
        full_path = osp.join(self.data_root, rel_path) if self.data_root else rel_path
        return {
            "video": full_path,
            "video_context_window": self.max_video_length,
            "sampling_fps": self.fps,
        }

def Captioning():
    parser = transformers.HfArgumentParser((ModelArguments, InferenceArguments))
    model_args, inference_args = parser.parse_args_into_dataclasses()
    seed = inference_args.seed or 42
    set_seed(seed)
    use_cuda = torch.cuda.is_available()
    if not use_cuda and (inference_args.bf16 or inference_args.fp16):
        print("Requested half precision on CPU; falling back to float32 for compatibility.")
        inference_args.bf16 = False
        inference_args.fp16 = False
    if not use_cuda and inference_args.clip_length > 32:
        print("Large clip length is memory intensive on CPU; capping clip_length to 32 frames.")
        inference_args.clip_length = 32
    if inference_args.bf16:
        compute_dtype = torch.bfloat16
    elif inference_args.fp16:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32
    model_args.torch_dtype = compute_dtype
    print("Loading model...")
    config = Videollama3Qwen2Config.from_pretrained(model_args.model_name_or_path)
    attn_impl = model_args.mm_attn_implementation or "flash_attention_2"
    if attn_impl == "flash_attention_2" and not torch.cuda.is_available():
        print("FlashAttention2 requested but CUDA is unavailable; falling back to SDPA attention.")
        attn_impl = "sdpa"
    config._attn_implementation = attn_impl
    config.mm_attn_implementation = attn_impl
    config.use_token_compression = model_args.use_token_compression
    if model_args.vision_encoder is not None:
        config.vision_encoder = model_args.vision_encoder
    model = Videollama3Qwen2ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=compute_dtype,
        do_sample=True,
    )
    model.config.use_cache = False
    device = inference_args.device or ("cuda" if use_cuda else "cpu")
    model.to(device)
    vl3_processor = Videollama3Processor.from_pretrained(model_args.model_name_or_path, trust_remote_code=False)
    model.eval()
    #Create dataset
    with open(inference_args.meta_data_path, 'r') as f:
        meta_data = json.load(f)
    if not isinstance(meta_data, dict):
        raise ValueError("Metadata file must describe a dict of dataset configs.")
    collected_dataset = []
    for dataset_name, dataset_cfg in meta_data.items():
        annotation_path = dataset_cfg.get("annotation")
        if annotation_path is None:
            raise ValueError(f"Dataset '{dataset_name}' is missing the 'annotation' key.")
        if not osp.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file {annotation_path} not found for dataset '{dataset_name}'.")
        collected_dataset.append(
            LazySupervisedDatasetForCaptioning(
                annotation_path=annotation_path,
                data_root=dataset_cfg.get("data_root"),
                fps=dataset_cfg.get("sampling_fps", inference_args.sampling_fps),
                max_video_length=inference_args.clip_length,
            )
        )
    if not collected_dataset:
        raise ValueError("No datasets were created from the provided metadata.")
    captioning_dataset = ConcatDataset(collected_dataset)
    total_videos = len(captioning_dataset)
    target_videos = min(total_videos, inference_args.max_videos) if inference_args.max_videos else total_videos
    print(f"Prepared {total_videos} videos from {len(collected_dataset)} datasets")
    base_prompt = "<video>\nIdentify all new events that occurred and ended up to the current frame, which have not been reported before. Provide their start times, durations, and descriptions in the format: <start time> - <end time> (duration: <x> seconds), <description>."
    # INFERENCE LOOP
    output_json = []
    processed_videos = 0
    progress_bar = tqdm(total=target_videos, desc="Captioning videos", unit="video") if target_videos > 0 else None
    try:
        for idx, video_obj in enumerate(captioning_dataset):
            if inference_args.max_videos is not None and idx >= inference_args.max_videos:
                print(f"Reached max_videos={inference_args.max_videos}. Stopping early.")
                break
            video_path = video_obj["video"]
            video_context_window = video_obj["video_context_window"]
            print(f"Processing video: {video_path}")
            video_clips, timestamps = LoadVideoWithClips(
                video_path,
                sampling_fps=video_obj["sampling_fps"],
                clip_length=video_context_window,
            )
            captions_before = []
            Prefix_prompt = "There is a streaming video provided. Below are some captions describing the events in the video at different timestamps in ascending order.\n"
            with torch.no_grad():
                for clip_idx, (video_clip, ts_clip) in enumerate(zip(video_clips, timestamps)):
                    if clip_idx == 0:
                        prompt = base_prompt
                    else:
                        prompt = Prefix_prompt + "\n".join(captions_before) + "\n" + base_prompt
                    processor = getattr(vl3_processor, "videollama3_processor", vl3_processor)
                    num_frames = video_clip.shape[0]
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video",
                                    "video": video_clip,
                                    "num_frames": int(num_frames),
                                    "timestamps": ts_clip,
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    inputs = processor(
                        conversation=conversation,
                        return_tensors="pt",
                    ).to(model.device)
                    if "pixel_values" in inputs:
                        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=inference_args.max_new_tokens,
                        do_sample=inference_args.do_sample,
                        temperature=inference_args.temperature,
                        top_k=inference_args.top_k,
                        top_p=inference_args.top_p,
                        num_beams=inference_args.num_beams,
                        num_return_sequences=inference_args.num_return_sequences,
                        no_repeat_ngram_size=inference_args.no_repeat_ngram_size,
                        repetition_penalty=inference_args.repetition_penalty,
                        length_penalty=inference_args.length_penalty,
                        early_stopping=inference_args.early_stopping,
                        pad_token_id=inference_args.pad_token_id or vl3_processor.tokenizer.pad_token_id,
                        bos_token_id=inference_args.bos_token_id or vl3_processor.tokenizer.bos_token_id,
                        eos_token_id=inference_args.eos_token_id or vl3_processor.tokenizer.eos_token_id,
                    )
                    generated_text = vl3_processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    if generated_text.startswith(prompt):
                        new_caption = generated_text[len(prompt):].strip()
                    else:
                        new_caption = generated_text.replace(prompt, "", 1).strip()
                    captions_before.append(new_caption)
            output_json.append(
                {
                    "video_path": video_path,
                    "captions": "\n".join(captions_before),
                }
            )
            processed_videos += 1
            if progress_bar:
                progress_bar.update(1)
    finally:
        if progress_bar:
            progress_bar.close()
    if processed_videos == 0:
        raise RuntimeError("No videos were processed. Check meta_data_path or max_videos setting.")
    with open(inference_args.output_file, 'w') as f:
        for item in output_json:
            f.write(json.dumps(item) + "\n")
    print(f"Captioning results saved to {inference_args.output_file} ({processed_videos} videos)")


if __name__ == "__main__":
    Captioning()
