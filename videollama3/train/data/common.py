"""Small shared helpers for the training entrypoints and the dataset modules.

``local_rank`` is a module global that every training entrypoint's ``train()``
sets right after parsing args (``videollama3.train.data.common.local_rank =
training_args.local_rank``); ``rank0_print`` and the data modules read it from
here so there is a single source of truth.
"""
import logging
from typing import Optional

import torch

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


def _is_trainable_lr(lr: Optional[float]) -> bool:
    return lr is not None and lr > 0


def _set_module_trainable(module: Optional[torch.nn.Module], trainable: bool):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = trainable


_PIXEL_VALUES_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def cast_pixel_values_(data_dict, spec: Optional[str]) -> None:
    """Down-cast ``data_dict["pixel_values"]`` in place, in the dataloader worker.

    The image processor always emits fp32 patches (`flatten_patches` in
    `image_processing_videollama3.py`), and that tensor is what crosses the
    worker -> main-process boundary through ``/dev/shm``. At Phase-3 geometry
    (``--vision_max_tokens 65536``) it is ~588 MiB per video, so
    ``num_workers * prefetch_factor * per_device_batch`` copies per rank blow past
    a 256 GiB ``/dev/shm``. DeepSpeed casts the encoder's float inputs to bf16 on
    entry anyway, so emitting bf16 here halves the shm/IPC traffic at no cost.
    """
    if not spec or spec == "float32":
        return
    dtype = _PIXEL_VALUES_DTYPES.get(spec)
    if dtype is None:
        raise ValueError(
            f"Unknown pixel_values_dtype {spec!r}; expected one of {sorted(_PIXEL_VALUES_DTYPES)}"
        )
    pv = data_dict.get("pixel_values", None)
    if pv is not None and pv.is_floating_point() and pv.dtype != dtype:
        data_dict["pixel_values"] = pv.to(dtype)
