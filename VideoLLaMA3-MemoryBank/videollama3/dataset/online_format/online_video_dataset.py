"""
Online Video Dataset for VideoChatOnline timestamp-aware format.
Supports both online format with timestamps and standard VideoLLaMA3 format for backward compatibility.
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any, Union
from PIL import Image


class OnlineVideoDataset(Dataset):
    """
    Dataset supporting VideoChatOnline's timestamp format.

    This dataset can handle:
    1. Online format: Videos with timestamp-aware conversations
    2. Standard format: Regular VideoLLaMA3 format for backward compatibility

    Online format structure:
    {
        "video": "path/to/video.mp4",
        "conversations": [
            {
                "from": "human",
                "timestamps": 30.5,
                "value": "<video>\nWhat happens at this moment?"
            },
            {
                "from": "gpt",
                "value": "A person is walking in the scene."
            }
        ]
    }
    """

    def __init__(
        self,
        data_path: str,
        tokenizer=None,
        processor=None,
        **kwargs
    ):
        super().__init__()

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.video_root = kwargs.get('video_root', '')
        self.max_video_frames = kwargs.get('max_video_frames', 768)
        self.default_fps = kwargs.get('fps', 1.0)

        # Load dataset
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # Detect dataset format
        self.format_type = self._detect_format_type()
        print(f"OnlineVideoDataset: Detected format '{self.format_type}' with {len(self.data)} samples")

        # Validate format
        if self.format_type == "unknown":
            raise ValueError(f"Unsupported dataset format in {data_path}")

    def _detect_format_type(self) -> str:
        """Detect whether dataset is online format or standard format."""
        if not self.data or not isinstance(self.data, list):
            return "empty"

        sample = self.data[0]

        # Check for online format characteristics
        if ('video' in sample and
            'conversations' in sample and
            isinstance(sample['conversations'], list) and
            len(sample['conversations']) > 0):

            # Check for timestamps in any conversation
            for conv in sample['conversations']:
                if isinstance(conv, dict) and 'timestamps' in conv:
                    return "online"

        # Check for image sequence format (also considered online)
        if ('video' in sample and
            'all_image_files' in sample and
            isinstance(sample['all_image_files'], list) and
            len(sample['all_image_files']) > 0):
            return "online"

        # Check for standard VideoLLaMA3 format
        if ('image' in sample or 'video' in sample) and 'conversations' in sample:
            return "standard"

        return "unknown"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")

        sample = self.data[idx]

        if self.format_type == "online":
            return self._process_online_sample(sample)
        elif self.format_type == "standard":
            return self._process_standard_sample(sample)
        else:
            raise ValueError(f"Unsupported dataset format: {self.format_type}")

    def _process_online_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process online format with timestamps."""
        video_path = sample.get('video', '')
        conversations = sample.get('conversations', [])

        # Extract all timestamps from human messages
        timestamps = []
        for conv in conversations:
            if (isinstance(conv, dict) and
                conv.get('from') == 'human' and
                'timestamps' in conv):
                timestamps.append(conv['timestamps'])

        # Determine max timestamp for video loading
        max_timestamp = max(timestamps) if timestamps else float('inf')

        # Load video up to max timestamp
        video_frames = self._load_video_until_timestamp(video_path, max_timestamp, sample)

        # Process conversations with temporal context
        processed_conversations = self._process_conversations_with_timestamps(
            conversations, video_frames, timestamps
        )

        return {
            'video': video_frames,
            'conversations': processed_conversations,
            'timestamps': timestamps,
            'video_path': video_path,
            'format_type': 'online',
            'original_sample': sample
        }

    def _resolve_video_path(self, video_path: str) -> str:
        """Resolve video path, handling missing extensions and multiple formats."""
        if not video_path:
            return ""

        base_path = os.path.join(self.video_root, video_path) if self.video_root else video_path

        # If path already has extension and exists, return it
        if os.path.exists(base_path):
            return base_path

        # If no extension, try different video formats
        if '.' not in os.path.basename(video_path):
            video_formats = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
            for fmt in video_formats:
                candidate_path = f"{base_path}{fmt}"
                if os.path.exists(candidate_path):
                    return candidate_path

        return base_path  # Return original path even if doesn't exist

    def _load_image_sequence(
        self,
        sample: Dict[str, Any],
        max_timestamp: float
    ) -> torch.Tensor:
        """Load image sequence from all_image_files."""
        image_files = sample.get("all_image_files", [])
        image_bboxes = sample.get("image_bboxes", [])
        fps = sample.get("fps", 1.0)
        video_path = sample.get("video", "")

        if not image_files:
            return torch.zeros(10, 3, 224, 224, dtype=torch.float32)

        # Base directory for images
        video_root = os.path.join(self.video_root, video_path) if self.video_root else video_path

        # Filter images based on timestamp if available
        if image_bboxes and max_timestamp != float('inf'):
            # Filter by timestamp
            valid_indices = []
            for i, bbox_info in enumerate(image_bboxes):
                if i < len(image_files) and bbox_info.get("timestamp", 0) <= max_timestamp:
                    valid_indices.append(i)

            if valid_indices:
                image_files = [image_files[i] for i in valid_indices]
                image_bboxes = [image_bboxes[i] for i in valid_indices]
            else:
                # Take first image if none match
                image_files = image_files[:1]
                image_bboxes = image_bboxes[:1] if image_bboxes else []

        # Apply max_video_frames limit with uniform sampling
        if len(image_files) > self.max_video_frames:
            sampled_indices = np.linspace(
                0, len(image_files) - 1, self.max_video_frames, dtype=int
            )
            image_files = [image_files[i] for i in sampled_indices]
            if image_bboxes:
                image_bboxes = [image_bboxes[i] for i in sampled_indices]

        # Load images
        image_tensors = []
        for img_file in image_files:
            img_path = os.path.join(video_root, img_file)
            try:
                # Load and convert image
                image = Image.open(img_path).convert('RGB')
                # Convert to tensor [C, H, W] and normalize to [0, 1]
                img_array = np.array(image).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW
                image_tensors.append(img_tensor)
            except Exception as e:
                print(f"Warning: Failed to load image {img_path}: {e}")
                # Create dummy image tensor
                dummy_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)
                image_tensors.append(dummy_tensor)

        if not image_tensors:
            return torch.zeros(10, 3, 224, 224, dtype=torch.float32)

        # Stack into [T, C, H, W] format
        video_tensor = torch.stack(image_tensors, dim=0)
        return video_tensor

    def _load_video_until_timestamp(
        self,
        video_path: str,
        max_timestamp: float,
        sample: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Load video frames up to specified timestamp."""
        if not video_path:
            return torch.zeros(10, 3, 224, 224, dtype=torch.float32)

        # Check if this is an image sequence
        if sample and sample.get("all_image_files"):
            return self._load_image_sequence(sample, max_timestamp)

        # Handle regular video file
        full_video_path = self._resolve_video_path(video_path)

        # Calculate required frames
        if max_timestamp == float('inf'):
            max_frames = self.max_video_frames
        else:
            estimated_frames = int(max_timestamp * self.default_fps) + 10
            max_frames = min(estimated_frames, self.max_video_frames)

        try:
            # Check if file exists
            if not os.path.exists(full_video_path):
                print(f"Warning: Video file not found: {full_video_path}")
                return torch.zeros(max_frames, 3, 224, 224, dtype=torch.float32)

            # Try to use custom video loading function if available
            if hasattr(self, '_load_video_frames'):
                video_frames = self._load_video_frames(
                    full_video_path,
                    max_frames=max_frames,
                    end_time=max_timestamp if max_timestamp != float('inf') else None
                )
            else:
                # Fallback: create dummy frames
                video_frames = torch.randn(max_frames, 3, 224, 224, dtype=torch.float32)
                print(f"Warning: Using dummy frames for {video_path} (video loading not available)")

        except Exception as e:
            print(f"Error loading video {full_video_path}: {e}")
            video_frames = torch.zeros(max_frames, 3, 224, 224, dtype=torch.float32)

        return video_frames

    def _process_conversations_with_timestamps(
        self,
        conversations: List[Dict[str, Any]],
        video_frames: torch.Tensor,
        timestamps: List[float]
    ) -> List[Dict[str, Any]]:
        """Process conversations considering temporal context."""
        processed = []

        for conv in conversations:
            if not isinstance(conv, dict):
                continue

            processed_conv = conv.copy()

            # Add temporal markers and frame index to human messages with timestamps
            if (conv.get('from') == 'human' and 'timestamps' in conv):
                timestamp = conv['timestamps']

                # Add frame index information for memory bank
                frame_idx = self._timestamp_to_frame_index(timestamp, video_frames.size(0))
                processed_conv['frame_index'] = frame_idx

                # Replace <video> with timestamp-aware video token if present
                if '<video>' in conv.get('value', ''):
                    processed_conv['value'] = conv['value'].replace(
                        '<video>',
                        f'<video_t{timestamp}>'
                    )

            processed.append(processed_conv)

        return processed

    def _timestamp_to_frame_index(self, timestamp: float, total_frames: int) -> int:
        """Convert timestamp to frame index."""
        if timestamp <= 0:
            return 0

        # Estimate frame index based on timestamp and fps
        frame_idx = int(timestamp * self.default_fps)
        return min(frame_idx, total_frames - 1)

    def _process_standard_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process standard VideoLLaMA3 format (for backward compatibility)."""
        # Add format identifier and return as-is for backward compatibility
        result = sample.copy()
        result['format_type'] = 'standard'
        result['timestamps'] = []  # No timestamps in standard format

        # If there's a video path, ensure it's properly handled
        if 'video' in sample:
            video_path = sample['video']
            result['video_path'] = video_path
            # For standard format, load entire video (up to max frames)
            result['video'] = self._load_video_until_timestamp(video_path, float('inf'), sample)

        return result

    def get_format_statistics(self) -> Dict[str, Any]:
        """Get statistics about the dataset format."""
        stats = {
            'total_samples': len(self.data),
            'format_type': self.format_type,
            'has_timestamps': False,
            'timestamp_count': 0,
            'avg_timestamps_per_sample': 0.0,
            'has_image_sequences': False,
            'image_sequence_count': 0,
            'has_bboxes': False,
            'bbox_count': 0
        }

        if self.format_type == "online":
            timestamp_counts = []
            image_seq_count = 0
            bbox_count = 0

            for sample in self.data:
                # Count conversation timestamps
                conversations = sample.get('conversations', [])
                sample_timestamps = [
                    conv.get('timestamps') for conv in conversations
                    if isinstance(conv, dict) and 'timestamps' in conv
                ]
                sample_timestamps = [t for t in sample_timestamps if t is not None]
                timestamp_counts.append(len(sample_timestamps))

                # Count image sequences
                if sample.get('all_image_files'):
                    image_seq_count += 1
                    stats['has_image_sequences'] = True

                # Count bboxes
                image_bboxes = sample.get('image_bboxes', [])
                bbox_count += len(image_bboxes)
                if image_bboxes:
                    stats['has_bboxes'] = True

            stats['has_timestamps'] = len([c for c in timestamp_counts if c > 0]) > 0
            stats['timestamp_count'] = sum(timestamp_counts)
            stats['avg_timestamps_per_sample'] = (
                stats['timestamp_count'] / len(self.data) if self.data else 0.0
            )
            stats['max_timestamps_per_sample'] = max(timestamp_counts) if timestamp_counts else 0
            stats['min_timestamps_per_sample'] = min(timestamp_counts) if timestamp_counts else 0
            stats['image_sequence_count'] = image_seq_count
            stats['bbox_count'] = bbox_count

        return stats

    def set_video_loader(self, loader_func):
        """Set custom video loading function."""
        self._load_video_frames = loader_func

    def __repr__(self) -> str:
        stats = self.get_format_statistics()
        return (
            f"OnlineVideoDataset("
            f"samples={stats['total_samples']}, "
            f"format='{stats['format_type']}', "
            f"timestamps={stats['timestamp_count']})"
        )