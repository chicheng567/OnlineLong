"""Base supervised-finetuning dataset + collators (LLaVA-style conversation data).

Extracted from ``videollama3_chat_finetune_online.py`` so the compressor training
scripts can reuse ``LazySupervisedDataset`` / ``ConcatDatasetWithLengths`` without
importing from another training entrypoint. ``videollama3_chat_finetune_online.py``
now imports these back from here.
"""
import copy
import json
import os
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import ConcatDataset, Dataset

from videollama3.constants import IGNORE_INDEX, STREAM_MAX_FRAMES
from videollama3.mm_utils import (
    load_images,
    load_video,
    preprocess_videollama3,
    process_qa,
    read_frames_decord,
)
from videollama3.train.data.common import logger, rank0_print

__all__ = [
    "ConcatDatasetWithLengths",
    "LazySupervisedDataset",
    "DataCollatorForSupervisedDataset",
    "DataCollatorWithFlatteningForSupervisedDataset",
    "make_supervised_data_module",
    "make_flattening_supervised_data_module",
]

class ConcatDatasetWithLengths(ConcatDataset):
    """
    Thin wrapper around torch.utils.data.ConcatDataset that preserves the
    length/ modality metadata expected by VideoLLaMA3Trainer when grouping.
    """

    def __init__(self, datasets):
        super().__init__(datasets)
        self._lengths = []
        self._modality_lengths = []
        self._compression_depths = []
        _all_depths = True
        for dataset in self.datasets:
            if not hasattr(dataset, "lengths") or not hasattr(dataset, "modality_lengths"):
                raise AttributeError(
                    f"{dataset} does not expose `lengths`/`modality_lengths`, "
                    "but they are required for grouped sampling."
                )
            self._lengths.extend(dataset.lengths)
            self._modality_lengths.extend(dataset.modality_lengths)
            d = getattr(dataset, "compression_depths", None)
            if d is None:
                _all_depths = False
            else:
                self._compression_depths.extend(d)
        if not _all_depths:
            self._compression_depths = None

    @property
    def lengths(self):
        return self._lengths

    @property
    def modality_lengths(self):
        return self._modality_lengths

    @property
    def compression_depths(self):
        return self._compression_depths


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, vlprocessor, data_args, dataset_name=None, dataset_root=None, online_mode=False, prefix_captioning=False, return_label=True):
        super(LazySupervisedDataset, self).__init__()
        data_objs = []
        self.dataset_name = dataset_name
        self.dataset_root = dataset_root
        self.online_mode = online_mode
        self.prefix_captioning = prefix_captioning
        self.return_label = return_label
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
        data_folder = self.dataset_root
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
            if isinstance(video_file, str):
                video_file = os.path.join(data_folder, video_file)
                images, timestamps = load_video(video_file, fps=self.data_args.fps, max_frames=self.data_args.max_frames)
                images = [images]
            elif isinstance(video_file, list) and len(video_file) == 1:
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
                return_labels=self.return_label,
                return_tensors="pt",
            )

            if modal == 'text':
                raise NotImplementedError("Text-only data is not supported so far.")
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
