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
