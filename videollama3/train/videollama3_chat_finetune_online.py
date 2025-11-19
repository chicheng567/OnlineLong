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

# NOTE: fast tokenizer warning issue: https://github.com/huggingface/transformers/issues/5486
os.environ["TOKENIZERS_PARALLELISM"] = "true"

local_rank = None
logger = logging.getLogger(__name__)


def rank0_print(*args):
    if local_rank == 0:
        message = ' '.join(str(arg) for arg in args)
        print(message)
        # Also log to logger if available
        if logging.getLogger().hasHandlers():
            logging.info(message)


def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def int_with_none(value):
    if value == 'None':
        return None
    return int(value)


@dataclass
class ModelArguments:
    # LLM Arguments
    model_name_or_path: Optional[str] = field(default="pretrained_models/videollama3_7b_local")
    tokenizer_name_or_path: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v1", metadata={"help": "Version of the conversation template."})
    freeze_llm: bool = field(default=False, metadata={"help": "Whether to freeze the LLM backbone."})
    freeze_vision_encoder: bool = field(default=True, metadata={"help": "Whether to freeze the vision encoder."})
    freeze_mlp: bool = field(default=False, metadata={"help": "Whether to freeze the multi-modal projector."})
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

from torch.utils.data import ConcatDataset


class ConcatDatasetWithLengths(ConcatDataset):
    """
    Thin wrapper around torch.utils.data.ConcatDataset that preserves the
    length/ modality metadata expected by VideoLLaMA3Trainer when grouping.
    """

    def __init__(self, datasets):
        super().__init__(datasets)
        self._lengths = []
        self._modality_lengths = []
        for dataset in self.datasets:
            if not hasattr(dataset, "lengths") or not hasattr(dataset, "modality_lengths"):
                raise AttributeError(
                    f"{dataset} does not expose `lengths`/`modality_lengths`, "
                    "but they are required for grouped sampling."
                )
            self._lengths.extend(dataset.lengths)
            self._modality_lengths.extend(dataset.modality_lengths)

    @property
    def lengths(self):
        return self._lengths

    @property
    def modality_lengths(self):
        return self._modality_lengths


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


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, vlprocessor, data_args: DataArguments, dataset_name=None, dataset_root=None, online_mode=False, prefix_captioning=False):
        super(LazySupervisedDataset, self).__init__()
        data_objs = []
        self.dataset_name = dataset_name
        self.dataset_root = dataset_root
        self.online_mode = online_mode
        self.prefix_captioning = prefix_captioning
        if dataset_root is not None:
            assert os.path.exists(dataset_root), f"Dataset root {dataset_root} not exists!"
        print(f"Loading data from {data_path}, dataset name: {self.dataset_name}, dataset root: {self.dataset_root}")
        if self.prefix_captioning:
            data = data_path[0]
            with open(data, 'r') as f:
                data_json = json.load(f)
            new_json = []
            for idx in range(len(data_json)):
                data = data_json[idx]
                ori_conversations = data['conversations']
                video = data['video']
                #spliting captions
                if len(ori_conversations) > 2:
                    prefix = "There is a streaming video provided. Below are some captions describing the events in the video at different timestamps in ascending order.\n"
                    suffix = "The following clip contains only the last few seconds of the ongoing stream.\n"
                    events = ""
                    for i in range(0, len(ori_conversations), 2):
                        new_obj = {}
                        new_obj["video"] = video
                        if i == 0:
                            new_obj["conversations"] = ori_conversations[:2]
                            events += new_obj["conversations"][1]["value"] + "\n"
                        else:
                            new_obj["conversations"] = [
                                {
                                    "from": "human",
                                    "start_time": ori_conversations[i - 2]["timestamps"],
                                    "timestamps": ori_conversations[i]["timestamps"],
                                    "value": prefix + events + suffix
                                },
                                ori_conversations[i+1]
                            ]
                            if "<video>" not in ori_conversations[i]["value"]:
                                new_obj["conversations"][0]["value"] += "<video>\n" + ori_conversations[i]["value"]
                            else:
                                new_obj["conversations"][0]["value"] += ori_conversations[i]["value"]
                            events += ori_conversations[i+1]["value"] + "\n"
                        new_json.append(new_obj)
                else:
                    new_json.append(data)
            list_data_dict = new_json
        else:      
            for data in data_path:
                if data.endswith(".json") or data.endswith(".jsonl") and self.prefix_captioning == False:
                    print(f"Loading {data} via `load_dataset`")
                    data_objs.append(load_dataset("json", data_files=data, cache_dir=data_args.dataset_cache_dir)["train"])
                else:
                    raise Exception(f"Unsupported file format (<{data}>)!")
            list_data_dict = concatenate_datasets(data_objs)
        
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.vlprocessor = vlprocessor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        print(f"Loaded {len(self.list_data_dict)} samples")
        

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 576 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def _convert_normal(self, data_dict):
        data_folder = self.data_args.data_folder
        conversation = copy.deepcopy(data_dict["conversations"])

        # data sanity check and repair
        start_idx = 0
        for sentence in conversation:
            if sentence["from"] == "human" or sentence["from"] == "system":
                break
            start_idx += 1
        if start_idx > 0:
            warnings.warn(f"Find {start_idx} non-user sentences at the beginning of the conversation, remove them automatically!")
            conversation = conversation[start_idx:]
        assert len(conversation) > 1, f"Invalid conversation"

        if 'image' in data_dict and data_dict['image'] is not None:
            modal = 'image'
            if all(not "<image>" in sentence["value"] for sentence in conversation):
                warnings.warn(f"Image tag not found in the conversation, add it automatically at the beginning!")
                conversation[0]["value"] = "<image>" + conversation[0]["value"]
            image_file = data_dict['image']
            if isinstance(image_file, list):
                image_file = [os.path.join(data_folder, f) for f in image_file]
            else:
                image_file = os.path.join(data_folder, image_file)
            images = load_images(image_file)
        elif 'video' in data_dict and data_dict['video'] is not None:
            modal = 'video'
            if all(not "<video>" in sentence["value"] for sentence in conversation):
                warnings.warn(f"Video tag not found in the conversation, add it automatically at the beginning!")
                conversation[0]["value"] = "<video>" + conversation[0]["value"]
            video_file = data_dict['video']
            if isinstance(video_file, list) and len(video_file) == 1:
                video_file = os.path.join(data_folder, video_file[0])
                images, timestamps = load_video(video_file, fps=self.data_args.fps, max_frames=self.data_args.max_frames)
                images = [images]
            else:
                raise ValueError(f"Unsupported video format: {video_file}")
        else:
            modal = 'text'
            images = None

        messages = []
        for conv in conversation:
            if conv["from"] == "human":
                # replace video tag to image tag for unified processing
                # conv["value"] = conv["value"].replace("<video>", "<image>" * len(images))
                chunks = conv["value"].split("<image>" if modal == 'image' else "<video>")
                messages.append({
                    "role": "user",
                    "content": []
                })

                for chunk_idx in range(1, 2 * len(chunks)):
                    if chunk_idx % 2 == 1:
                        chunk = chunks[chunk_idx // 2].strip()
                        messages[-1]["content"].append({"type": "text",  "text": chunk}) if chunk else None
                    else:
                        if modal == 'image':
                            messages[-1]["content"].append({"type": "image"})
                        elif modal == 'video':
                            messages[-1]["content"].append({"type": "video", "num_frames": len(images[0]), "timestamps": timestamps})
            else:
                messages.append({
                    "role": "assistant",
                    "content": conv['value']
                })

        if modal == 'video':
            merge_size = self.data_args.video_merge_size
        else:
            # image/text
            merge_size = self.data_args.image_merge_size

        return modal, images, messages, merge_size

    def _convert_stream(self, data_dict):
        video_path = os.path.join(self.data_args.data_folder, data_dict['video'][0])
        frames, timestamps = load_video(
            video_path=video_path,
            start_time=data_dict["start_time"],
            end_time=data_dict["end_time"],
            fps=self.data_args.fps,
            max_frames=self.data_args.max_frames,
        )

        if len(frames) > STREAM_MAX_FRAMES:
            max_time = timestamps[STREAM_MAX_FRAMES]
            frames = frames[:STREAM_MAX_FRAMES]
            timestamps = timestamps[:STREAM_MAX_FRAMES]
        else:
            max_time = float("inf")

        messages = []
        frame_idx = 0

        conversation = copy.deepcopy(data_dict["conversation"])
        for message in conversation:
            if message["time"] >= max_time:
                break

            while frame_idx < len(timestamps) and timestamps[frame_idx] <= message["time"]:
                messages.append({
                    "role": "stream",
                    "content": [{"type": "image", "timestamps": timestamps[frame_idx] - data_dict["start_time"]}],
                })
                frame_idx += 1

            messages.append(message)

        frames = frames[:frame_idx]

        return "video", [frames], messages, self.data_args.video_merge_size
    def _convert_online_video(self, data_dict):
        image_files = data_dict.get("all_image_files", None)
        if image_files is None:
            video_file = data_dict["video"]
            if self.dataset_root is None:
                self.dataset_root = self.data_args.data_folder
            video_path = os.path.join(self.dataset_root, video_file)

            if len(video_path.split(".")) == 1:
                video_formats = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
                for fmt in video_formats:  # Added this line
                    if os.path.exists(f"{video_path}{fmt}"):
                        video_path = f"{video_path}{fmt}"
                        break
            if "start_time" in data_dict["conversations"][0]:
                assert len(data_dict["conversations"]) == 2, "start time only support one query."
                start_time = data_dict["conversations"][0]["start_time"]
                end_time = data_dict["conversations"][0]["timestamps"]
                clip = (start_time, end_time)
            else:
                clip = None
            image_list, timestamps = read_frames_decord(
                video_path,
                sample="fps"+str(self.data_args.fps),
                num_frames=self.data_args.max_frames,
                min_num_frames=1,
                clip=clip,
                return_timestamps=True,
                force_context_length=self.prefix_captioning
            )
            assert len(timestamps) == len(image_list), f"{len(timestamps)} != {len(image_list)}"
            assert len(timestamps) > 0, f"Empty video frames! {video_path}, {clip}"
        else:
            #for object tracking tasks
            fps = data_dict.get("fps", 1)  # Default to 1 fps if not specified
            video_file = data_dict["video"]
            video_root = os.path.join(self.dataset_root, video_file)
            # Uniformly sample to the max_num_frame length
            if len(image_files) > self.data_args.max_frames:
                # Use np.linspace to generate evenly spaced indices
                sampled_indices = np.linspace(
                    0, len(image_files) - 1, self.data_args.max_frames, dtype=int
                )
                image_files = [image_files[i] for i in sampled_indices]
                image_bboxes = [data_dict["image_bboxes"][i] for i in sampled_indices]
            else:
                image_bboxes = data_dict["image_bboxes"]
            # Load all images
            image_list = [
                load_images(os.path.join(video_root, img)) for img in image_files
            ]
            # Generate timestamps
            timestamps = [round(bbox["timestamp"], 1) for bbox in image_bboxes]
            # Get the corresponding bbox
            # Randomly select one image's bbox to replace <bbox> in query_template
            random_index = random.randint(0, len(image_bboxes) - 1)
            selected_bbox = image_bboxes[random_index]
            selected_timestamp = timestamps[random_index]
            # Modify query_template, replace bbox and timestamp
            query_template = data_dict["query_template"]
            human_query = query_template.copy()
            human_query["timestamps"] = selected_timestamp
            human_query["value"] = human_query["value"].replace(
                "<bbox>", str(selected_bbox["bbox"])
            )
            # f"Track the location and actions of the \"person\" at position {selected_bbox['bbox']} over time. Provide start and end timestamps for each instance in seconds with bounding box coordinates."
            # Generate GPT output (timestamps and bbox from 0 to t)
            gpt_output = {
                "from": "gpt",
                "value": "\n".join(
                    [
                        f"At {t}s, {bbox['bbox']}"
                        for t, bbox in zip(
                            timestamps[: random_index + 1],
                            image_bboxes[: random_index + 1],
                        )
                    ]
                ),
            }
            # Generate subsequent human and GPT data (for time after t)
            conversations = [human_query, gpt_output]
            for i, (image_file, timestamp) in enumerate(
                zip(image_files[random_index + 1 :], timestamps[random_index + 1 :])
            ):
                # Human query for video section
                human_query_after = {
                    "from": "human",
                    "timestamps": timestamp,
                    "image_file": image_file,
                    "value": "<video>\n",
                }
                # GPT response for video section
                gpt_response_after = {
                    "from": "gpt",
                    "value": f"At {timestamp}s, {image_bboxes[random_index+1+i]['bbox']}",
                }
                conversations.extend([human_query_after, gpt_response_after])
            data_dict.update({"conversations": conversations})
        if "QA" in data_dict:
            data_dict["conversations"] = process_qa(data_dict["QA"])
        # Ensure the first conversation contains a video placeholder
        for i in range(0, len(data_dict["conversations"]), 2):
            data_dict["conversations"][i]["value"] = data_dict["conversations"][i][
                "value"
            ].replace("<image>", "<video>")
            if "<video>" not in data_dict["conversations"][i]["value"]:
                data_dict["conversations"][i]["value"] = (
                    "<video>\n" + data_dict["conversations"][i]["value"]
                )
        if data_dict.get("need_reset_timestamp", False):
            timestamps = [t - timestamps[0] for t in timestamps]
        
        start_index = 0
        for i in range(0, len(data_dict["conversations"]), 2):
            if image_files is not None:
                image_file = data_dict["conversations"][i].get("image_file", None)
                if image_file is not None and image_file not in image_files:
                    break
            #some query timestamps in the conversation may longer than the video length
            data_dict["conversations"][i]["timestamps"] = min(
                round(timestamps[-1], 1) + 0.1,
                data_dict["conversations"][i]["timestamps"],
            )
            end = data_dict["conversations"][i]["timestamps"]
            #find the end index
            for end_index in range(start_index, len(timestamps)):
                if timestamps[end_index] > end:
                    break
            else:
                end_index = len(timestamps)
            start_index = end_index
    
        image_list = image_list[:end_index]
        timestamps = np.array([round(t, 1) for t in timestamps[:end_index]])
        
        num_frames = len(image_list)
        assert num_frames == len(timestamps), f"{num_frames} != {len(timestamps)}"
        message = preprocess_videollama3(
            deepcopy(data_dict["conversations"]),
            timestamps,
        )
        return "video", image_list, message, self.data_args.video_merge_size
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data_dict = self.list_data_dict[i]
        try:
            if self.online_mode:
                #online video with query timestamps processing
                modal, images, messages, merge_size = self._convert_online_video(data_dict)
            else:
                #from orgin videollama3
                if "stream" in data_dict and data_dict["stream"]:
                    modal, images, messages, merge_size = self._convert_stream(data_dict)
                else:
                    modal, images, messages, merge_size = self._convert_normal(data_dict)
            
            data_dict = self.vlprocessor(
                images=images,
                text=messages,
                merge_size=merge_size,
                return_labels=True,
                return_tensors="pt",
            )

            if modal == 'text':
                unit_size = self.vlprocessor.image_processor.patch_size**2 * 3
                data_dict['pixel_values'] = torch.zeros(self.data_args.image_merge_size**2, unit_size)
                data_dict['grid_sizes'] = torch.as_tensor([[1, self.data_args.image_merge_size, self.data_args.image_merge_size]])
                data_dict['merge_sizes'] = torch.as_tensor([self.data_args.image_merge_size])
            elif modal == 'image' or modal == 'video':
                assert len(data_dict['pixel_values']) > 0 and len(data_dict['grid_sizes']) > 0, f"Invalid image data: {data_dict['images']}, {data_dict['grid_thws']}"
            data_dict['modals'] = [modal] * len(images)

        except Exception:
            backup_idx = random.randint(0, len(self.list_data_dict) - 1)
            logger.exception(
                "Failed to process sample %s (dataset=%s, root=%s). Fallback index: %s. Entry: %s",
                i,
                self.dataset_name or "unknown",
                self.dataset_root or self.data_args.data_folder,
                backup_idx,
                data_dict,
            )
            return self.__getitem__(backup_idx)

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    vlprocessor: transformers.ProcessorMixin

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.vlprocessor.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.vlprocessor.tokenizer.model_max_length]
        labels = labels[:, :self.vlprocessor.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.vlprocessor.tokenizer.pad_token_id),
        )

        # work for 'images' argument in `prepare_inputs_labels_for_multimodal`
        batch["pixel_values"] = torch.cat([x["pixel_values"] for x in instances])
        batch["grid_sizes"] = torch.cat([x["grid_sizes"] for x in instances])
        batch["merge_sizes"] = torch.cat([x["merge_sizes"] for x in instances])
        batch["modals"] = sum([x["modals"] for x in instances], [])

        return batch


def make_supervised_data_module(vlprocessor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        vlprocessor=vlprocessor,
        data_path=data_args.data_path,
        data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(vlprocessor=vlprocessor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


@dataclass
class DataCollatorWithFlatteningForSupervisedDataset(object):
    """Collate examples for batch flattened supervised fine-tuning."""

    vlprocessor: transformers.ProcessorMixin

    def __call__(self, instances: Sequence[Dict], separator_id=-100) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))

        new_input_ids = []
        new_labels = []
        position_ids = []
        for idx in range(0, len(input_ids)):
            new_input_ids.append(input_ids[idx][:self.vlprocessor.tokenizer.model_max_length])
            temp_label = labels[idx][:self.vlprocessor.tokenizer.model_max_length]
            temp_label[0] = separator_id
            new_labels.append(temp_label)
            position_ids.append(torch.tensor(list(range(len(input_ids[idx][:self.vlprocessor.tokenizer.model_max_length])))))

        new_input_ids = torch.cat(new_input_ids)
        new_labels = torch.cat(new_labels)
        position_ids = torch.cat(position_ids)

        batch = dict(
            input_ids=new_input_ids.unsqueeze(0),
            labels=new_labels.unsqueeze(0),
            position_ids=position_ids.unsqueeze(0),
        )

        # work for 'images' argument in `prepare_inputs_labels_for_multimodal`
        batch["pixel_values"] = torch.cat([x["pixel_values"] for x in instances])
        batch["grid_sizes"] = torch.cat([x["grid_sizes"] for x in instances])
        batch["merge_sizes"] = torch.cat([x["merge_sizes"] for x in instances])
        batch["modals"] = sum([x["modals"] for x in instances], [])

        return batch


def make_flattening_supervised_data_module(vlprocessor: transformers.ProcessorMixin, data_args) -> Dict:
    """Make batch flattened dataset and collator for supervised fine-tuning."""
    if data_args.multi_dataset:
        rank0_print("Use meta file to control datasets loading. Data path will use as meta path")
        ds_collection = dict()
        meta_path = data_args.data_path[0]
        ds_collection.update(json.loads(open(meta_path).read()))
        collected_datasets = []
        for k, v in ds_collection.items():
            collected_datasets.append(LazySupervisedDataset(
                vlprocessor=vlprocessor,
                data_path=[v['annotation']],
                data_args=data_args,
                dataset_name=k,
                dataset_root=v['data_root'],
                online_mode=v['online_mode'],
                #captioning task only need previous captions as prefix, previous frames are not needed.
                prefix_captioning=v.get('prefix_captioning', False)
            ))
        train_dataset = ConcatDatasetWithLengths(collected_datasets)
    else:
        train_dataset = LazySupervisedDataset(
            vlprocessor=vlprocessor,
            data_path=data_args.data_path,
            data_args=data_args
        )
    data_collator = DataCollatorWithFlatteningForSupervisedDataset(vlprocessor=vlprocessor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


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
            # device_map={"": training_args.device},
            # BUG: High version transformers report error:
            # ValueError: You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing `quantization_config` argument at the same time
            # load_in_4bit=training_args.bits == 4,
            # load_in_8bit=training_args.bits == 8,
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

        if model.config.llm_lr is None:
            for p in model.get_model().parameters():
                p.requires_grad = False
            for p in model.get_model().vision_encoder.parameters():
                p.requires_grad = True
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        if model.config.vision_encoder_lr is None:
            for p in model.get_model().vision_encoder.parameters():
                p.requires_grad = False

        if model.config.mm_projector_lr is None:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        model.config.max_frames = getattr(data_args, 'max_frames', NUM_FRAMES)
        model.config.image_aspect_ratio = data_args.image_aspect_ratio if 'avt' not in model_args.vision_encoder else 'avt'

        # NOTE: complement data_args via model hyperparameters
        # 1. acquire image size
        model.config.image_size = data_args.image_size = vision_encoder.image_size
        # 2. calculate the number of tokens in the image
        model.config.image_token_length = data_args.image_token_length = mm_projector.cal_proj_size(vision_encoder.num_patches_per_side)
        # 3. check if alignment
        model.config.is_alignment = training_args.is_alignment = data_args.is_alignment = (
            model.config.mm_projector_lr is not None and
            model.config.llm_lr is None and
            model.config.vision_encoder_lr is None
        )
        # 4. set spatial merge size as default
        new_tokens = tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN, STREAM_START_TOKEN, STREAM_END_TOKEN], special_tokens=True)
        assert new_tokens == 0, "Tokenizer already has the special tokens!"
        model.config.image_token_index = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        if data_args.force_image_size is not None:
            vision_encoder.image_processor.force_size = [data_args.force_image_size] * 2
            rank0_print(f"Force set image size to be {data_args.force_image_size}")
        vlprocessor = Videollama3Processor(vision_encoder.image_processor, tokenizer)
    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False
    if model_args.freeze_vision_encoder and model_args.vision_encoder is not None:
        _freeze_params(model.get_vision_encoder())
    if model_args.freeze_mlp and model_args.vision_encoder is not None:
        _freeze_params(model.get_mm_projector())
    if model_args.freeze_llm:
        _freeze_params(model.model.layers)
        _freeze_params(model.model.embed_tokens)
        _freeze_params(model.model.norm)
        _freeze_params(model.lm_head)
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
