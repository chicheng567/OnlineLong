"""Dataset / collator machinery shared by the training entrypoints.

Training entrypoints (``videollama3_chat_finetune_online.py``,
``videollama3_chat_finetune_compressor.py``, ``compressor_pretrain_with_videollama3.py``,
``stage2a_pretrain_compressor_fold.py``) import from here instead of from one another.

Layers, imported one direction only:
    common            -> tiny helpers (rank0_print, set_seed, ...)
    supervised        -> LazySupervisedDataset + base collators
    compressor        -> windowed-compression SFT dataset/collator + window helpers
    global_compressor -> whole-video ("global") compression dataset + frame helpers
"""
from videollama3.train.data.common import (
    int_with_none,
    rank0_print,
    set_seed,
    _is_trainable_lr,
    _set_module_trainable,
)
from videollama3.train.data.supervised import (
    ConcatDatasetWithLengths,
    DataCollatorForSupervisedDataset,
    DataCollatorWithFlatteningForSupervisedDataset,
    LazySupervisedDataset,
    make_flattening_supervised_data_module,
    make_supervised_data_module,
)
from videollama3.train.data.compressor import (
    CompressorLazySupervisedDataset,
    DataCollatorWithCompressor,
    SubsetWithLengths,
    count_video_frames_in_messages,
    make_compressor_data_module,
    select_compression_parts,
    select_full_compression_parts,
)
from videollama3.train.data.global_compressor import (
    GlobalCompressorLazySupervisedDataset,
    build_range_ts_info,
    get_video_content,
    make_global_compressor_data_module,
    resample_indices,
    resample_video_frames,
)

__all__ = [
    "int_with_none",
    "rank0_print",
    "set_seed",
    "_is_trainable_lr",
    "_set_module_trainable",
    "ConcatDatasetWithLengths",
    "DataCollatorForSupervisedDataset",
    "DataCollatorWithFlatteningForSupervisedDataset",
    "LazySupervisedDataset",
    "make_flattening_supervised_data_module",
    "make_supervised_data_module",
    "CompressorLazySupervisedDataset",
    "DataCollatorWithCompressor",
    "SubsetWithLengths",
    "count_video_frames_in_messages",
    "make_compressor_data_module",
    "select_compression_parts",
    "select_full_compression_parts",
    "GlobalCompressorLazySupervisedDataset",
    "build_range_ts_info",
    "get_video_content",
    "make_global_compressor_data_module",
    "resample_indices",
    "resample_video_frames",
]
