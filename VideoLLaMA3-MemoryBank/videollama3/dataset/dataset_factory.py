"""
Dataset Factory for creating appropriate dataset instances based on format detection.
Supports both online timestamp-aware format and standard VideoLLaMA3 format.
"""

import json
import os
from typing import Any, Dict, Optional

from .online_format import OnlineVideoDataset


def create_dataset(
    data_path: str,
    tokenizer=None,
    processor=None,
    **kwargs
) -> OnlineVideoDataset:
    """
    Factory function to create appropriate dataset based on format.

    This function automatically detects the dataset format and creates the
    appropriate dataset instance. Currently supports:
    1. Online format (VideoChatOnline with timestamps)
    2. Standard format (VideoLLaMA3 format)

    Args:
        data_path: Path to the dataset JSON file
        tokenizer: Tokenizer for text processing
        processor: Processor for multimodal data
        **kwargs: Additional arguments passed to dataset constructor

    Returns:
        OnlineVideoDataset: Dataset instance that handles the detected format

    Raises:
        FileNotFoundError: If data_path doesn't exist
        ValueError: If dataset format is not supported
        json.JSONDecodeError: If JSON file is malformed
    """

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    # Try to detect format by reading first few samples
    try:
        format_type = detect_dataset_format(data_path)
        print(f"DatasetFactory: Detected format '{format_type}' in {data_path}")

        # Create OnlineVideoDataset which handles both formats
        return OnlineVideoDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            processor=processor,
            **kwargs
        )

    except Exception as e:
        print(f"Error creating dataset: {e}")
        raise


def detect_dataset_format(data_path: str) -> str:
    """
    Detect dataset format by examining the structure.

    Args:
        data_path: Path to the dataset JSON file

    Returns:
        str: Format type ('online', 'standard', 'unknown', or 'empty')

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data or not isinstance(data, list):
            return "empty"

        # Take first few samples for format detection
        samples_to_check = min(3, len(data))
        online_format_count = 0
        standard_format_count = 0

        for i in range(samples_to_check):
            sample = data[i]
            if not isinstance(sample, dict):
                continue

            # Check for online format characteristics
            if _is_online_format(sample):
                online_format_count += 1
            elif _is_standard_format(sample):
                standard_format_count += 1

        # Determine format based on majority
        if online_format_count > 0:
            return "online"
        elif standard_format_count > 0:
            return "standard"
        else:
            return "unknown"

    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in {data_path}: {e}")
    except Exception as e:
        print(f"Warning: Error detecting format in {data_path}: {e}")
        return "unknown"


def _is_online_format(sample: Dict[str, Any]) -> bool:
    """Check if sample follows online format."""
    if not isinstance(sample, dict):
        return False

    # Must have video
    if not 'video' in sample:
        return False

    # Check for image sequence format
    if ('all_image_files' in sample and
        isinstance(sample['all_image_files'], list) and
        len(sample['all_image_files']) > 0):
        return True

    # Check for conversation-based online format
    if not 'conversations' in sample:
        return False

    conversations = sample['conversations']
    if not isinstance(conversations, list) or len(conversations) == 0:
        return False

    # Check for timestamps in any conversation
    for conv in conversations:
        if isinstance(conv, dict) and 'timestamps' in conv:
            return True

    return False


def _is_standard_format(sample: Dict[str, Any]) -> bool:
    """Check if sample follows standard VideoLLaMA3 format."""
    if not isinstance(sample, dict):
        return False

    # Must have either image or video, and conversations
    if not (('image' in sample or 'video' in sample) and 'conversations' in sample):
        return False

    conversations = sample['conversations']
    if not isinstance(conversations, list):
        return False

    # Standard format typically doesn't have timestamps
    for conv in conversations:
        if isinstance(conv, dict) and 'timestamps' in conv:
            return False

    return True


def get_dataset_info(data_path: str) -> Dict[str, Any]:
    """
    Get detailed information about a dataset without loading it fully.

    Args:
        data_path: Path to the dataset JSON file

    Returns:
        Dict containing dataset information
    """

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        info = {
            'path': data_path,
            'total_samples': len(data) if isinstance(data, list) else 0,
            'format_type': detect_dataset_format(data_path),
            'file_size_mb': os.path.getsize(data_path) / (1024 * 1024),
            'has_video': False,
            'has_image': False,
            'has_timestamps': False,
            'sample_fields': set()
        }

        # Analyze first few samples
        if isinstance(data, list) and len(data) > 0:
            samples_to_check = min(5, len(data))

            for i in range(samples_to_check):
                sample = data[i]
                if isinstance(sample, dict):
                    # Collect field names
                    info['sample_fields'].update(sample.keys())

                    # Check for media types
                    if 'video' in sample:
                        info['has_video'] = True
                    if 'image' in sample:
                        info['has_image'] = True

                    # Check for timestamps
                    conversations = sample.get('conversations', [])
                    for conv in conversations:
                        if isinstance(conv, dict) and 'timestamps' in conv:
                            info['has_timestamps'] = True
                            break

        # Convert set to list for JSON serialization
        info['sample_fields'] = list(info['sample_fields'])

        return info

    except Exception as e:
        return {
            'path': data_path,
            'error': str(e),
            'total_samples': 0,
            'format_type': 'error'
        }


# Backward compatibility alias
def create_online_dataset(*args, **kwargs):
    """Deprecated: Use create_dataset() instead."""
    import warnings
    warnings.warn(
        "create_online_dataset() is deprecated. Use create_dataset() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_dataset(*args, **kwargs)