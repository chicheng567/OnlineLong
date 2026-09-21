# Adopted from: https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py
import os
import logging
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    logger,
    TRAINER_STATE_NAME,
)
from qwen2.modeling_qwen2 import Qwen2RMSNorm
ALL_LAYERNORM_LAYERS = (nn.LayerNorm, Qwen2RMSNorm)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_encoder', 'vision_resampler', 'token_compressor']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "is_alignment", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        # if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if torch.distributed.get_rank() == 0:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class VideoLLaMA3Trainer(Trainer):

    def __init__(self, *args, use_dual_forward=True, partial_loss_weight=1.0, full_loss_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_dual_forward = use_dual_forward
        self.partial_loss_weight = partial_loss_weight
        self.full_loss_weight = full_loss_weight

    def get_train_dataloader(self):
        """Keep prefetched batches on the HOST until the micro-step that uses them.

        ``Trainer.get_batch_samples`` pulls ``gradient_accumulation_steps`` batches
        into a list in one go, and accelerate's ``DataLoaderShard.__iter__`` has
        already ``send_to_device``'d each one, so the WHOLE accumulation window's
        inputs sit in VRAM simultaneously. On a video model that is the single
        largest allocation in the step: one Phase-3 batch carries a
        ``(~250k, 588)`` ``pixel_values`` (~0.5 GiB at --vision_max_tokens 65536),
        so ``GLOBAL_BATCH 512`` on 8 GPUs (grad_acc = 64) parks **~32 GiB per rank**
        before the first forward runs. Measured: `live` = 15.8 GiB right after
        deepspeed init, 48.6 GiB at the first forward, flat across all 64
        micro-steps, and the OOM lands in `get_batch_samples` -> `send_to_device`
        fetching the next window.

        ``DataLoaderShard`` skips the transfer when ``device is None``, and
        ``Trainer.training_step`` calls ``_prepare_inputs`` anyway, which moves (and,
        under DeepSpeed, casts) each batch as its micro-step runs. So clearing the
        attribute makes the window cost ONE batch instead of ``grad_acc`` of them,
        with no change to what the model sees. Set VL3_KEEP_BATCHES_ON_GPU=1 to
        restore the stock behaviour.
        """
        dataloader = super().get_train_dataloader()
        if os.environ.get("VL3_KEEP_BATCHES_ON_GPU") == "1":
            return dataloader
        if getattr(dataloader, "device", None) is not None:
            dataloader.device = None
            if self.args.local_rank in (0, -1):
                print("[trainer] dataloader device placement OFF -- batches move to GPU "
                      "per micro-step in _prepare_inputs, not grad_acc at a time")
        return dataloader

    def training_step(self, *args, **kwargs):
        """VL3_NAN_PROBE=1: tick the per-step probe counter and, after backward,
        name the first parameter GROUP whose grad (or value) went non-finite.

        Phase 3 reports grad_norm=nan several steps before the forward loss goes
        nan, so the parameter scan is what localises the blow-up; the per-stage
        activation probes in compressor.py say which forward stage fed it."""
        import os as _os
        probe = _os.environ.get("VL3_NAN_PROBE", "")
        if not probe:
            return super().training_step(*args, **kwargs)

        from videollama3.model import compressor as _comp
        _comp._nan_probe_new_step()
        out = super().training_step(*args, **kwargs)

        if self.args.local_rank not in (0, -1):
            return out
        try:
            model = self.accelerator.unwrap_model(self.model)
        except Exception:
            model = self.model
        groups = {}
        for name, prm in model.named_parameters():
            if not prm.requires_grad:
                continue
            # token_compressor.stage1.* -> "token_compressor.stage1", etc.
            key = ".".join(name.split(".")[:2])
            g = groups.setdefault(key, {"n": 0, "bad_grad": 0, "bad_val": 0,
                                        "gmax": 0.0, "no_grad": 0})
            g["n"] += 1
            fin_v = torch.isfinite(prm.detach())
            if not bool(fin_v.all()):
                g["bad_val"] += 1
                if prm.dim() == 2 and g.get("rows") is None:
                    bad = (~fin_v).any(1).nonzero().flatten()
                    g["rows"] = (name, int(bad.numel()), int(prm.shape[0]),
                                 bad[:4].tolist(), bad[-4:].tolist())
            gr = prm.grad
            if gr is None:
                g["no_grad"] += 1
                continue
            gd = gr.detach().float()
            fin = torch.isfinite(gd)
            if not bool(fin.all()):
                g["bad_grad"] += 1
            if bool(fin.any()):
                g["gmax"] = max(g["gmax"], float(gd[fin].abs().max()))
        rows = [(k, v) for k, v in groups.items()
                if v["bad_grad"] or v["bad_val"] or probe == "verbose"]
        if rows:
            from tqdm import tqdm as _tqdm
            for k, v in sorted(rows):
                _tqdm.write(
                    f"[NAN_PROBE step {_comp._nan_probe_step[0]:>4}] param {k:<28} "
                    f"n={v['n']:<4} bad_grad={v['bad_grad']:<4} bad_value={v['bad_val']:<4} "
                    f"no_grad={v['no_grad']:<4} max|grad|={v['gmax']:.4g}"
                )
                if v.get("rows"):
                    nm, nbad, nrows, first, last = v["rows"]
                    _tqdm.write(
                        f"[NAN_PROBE step {_comp._nan_probe_step[0]:>4}]   -> {nm}: "
                        f"{nbad}/{nrows} rows non-finite, first={first} last={last}"
                    )
        return out

    def _ce_forward(self, model, base_inputs, compression_parts, compression_ts_info, label="", num_items_in_batch=None):
        """Run one student forward (CE mode) with the given compression_parts.

        Returns (ce_loss, outputs).
        """
        fwd = {
            k: v for k, v in base_inputs.items()
            if k not in (
                "compression_parts", "compression_ts_info",
                "compression_parts_full", "compression_ts_info_full",
            )
        }
        fwd["compression_parts"] = compression_parts
        fwd["compression_ts_info"] = compression_ts_info
        if label:
            fwd["_debug_label"] = label
        # Pass num_items_in_batch so the model uses sum/N_total reduction instead of mean,
        # which is required when model_accepts_loss_kwargs=True (HF won't divide by GAS).
        if num_items_in_batch is not None:
            fwd["num_items_in_batch"] = num_items_in_batch
        outputs = model(**fwd)
        return outputs.loss, outputs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # HF passes num_items_in_batch (total non-IGNORE tokens across all GAS micro-batches)
        # when model_accepts_loss_kwargs=True. Forward it to _ce_forward so the model uses
        # sum/N_total reduction instead of mean — critical for correct GAS normalisation.
        num_items_in_batch = kwargs.get("num_items_in_batch")
        parts = inputs.get("compression_parts", [])
        ts = inputs.get("compression_ts_info", [])
        parts_full = inputs.get("compression_parts_full", [])
        ts_full = inputs.get("compression_ts_info_full", [])

        if not parts and not parts_full:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        loss = None
        outputs = None
        l_partial = None
        l_full = None

        if parts:
            l_partial, outs_partial = self._ce_forward(model, inputs, parts, ts, label="partial", num_items_in_batch=num_items_in_batch)
            loss = self.partial_loss_weight * l_partial
            outputs = outs_partial

        if self.use_dual_forward and parts_full:
            l_full, outs_full = self._ce_forward(model, inputs, parts_full, ts_full, label="full", num_items_in_batch=num_items_in_batch)
            weighted_full = self.full_loss_weight * l_full
            loss = weighted_full if loss is None else loss + weighted_full
            if outputs is None:
                outputs = outs_full

        log_dict = {}
        if l_partial is not None:
            log_dict["loss_partial"] = (self.partial_loss_weight * l_partial).detach().item()
        if l_full is not None:
            log_dict["loss_full"] = (self.full_loss_weight * l_full).detach().item()

        # Option B: distribution-match aux loss stashed on the token compressor
        # during the CE forward (compressor._distribution_match_loss). Added here so
        # it shows up as its own logged term. No-op unless
        # --compressor_distr_loss_weight > 0.
        #
        # compute_loss() runs once per gradient-accumulation micro-batch. The CE
        # term above is already correctly batch-averaged via num_items_in_batch
        # (sum-reduced over the whole accumulated batch, so summing GAS micro-batch
        # contributions reproduces the true mean). `aux` has no such built-in
        # normalization -- each call adds this micro-batch's own raw distr loss at
        # full weight -- so without dividing by GAS it is effectively counted GAS
        # times (empirically confirmed: reported step loss == sum of per-micro-batch
        # loss_partial + loss_distr, so a GAS=16 run applies distr_loss_weight as if
        # it were 16x larger than configured).
        aux = self._compressor_distr_loss(model)
        if aux is not None and loss is not None:
            aux = aux / self.args.gradient_accumulation_steps
            loss = loss + aux
            log_dict["loss_distr"] = aux.detach().item()

        if log_dict:
            self.log(log_dict)

        return (loss, outputs) if return_outputs else loss

    def _compressor_distr_loss(self, model):
        """Return the already-weighted Option-B distribution-match loss stashed on
        the token compressor by the most recent CE forward, or None when disabled /
        absent. With use_dual_forward the stash reflects the LAST (full) window
        only; the recipes that enable this weight run a single forward."""
        comp = None
        try:
            comp = self.accelerator.unwrap_model(model).get_token_compressor()
        except Exception:
            for attr in ("module", "base_model"):
                inner = getattr(model, attr, None)
                if inner is not None and hasattr(inner, "get_token_compressor"):
                    comp = inner.get_token_compressor()
                    break
        if comp is None:
            return None
        w = float(getattr(comp, "distr_loss_weight", 0.0) or 0.0)
        raw = getattr(comp, "_last_distr_loss", None)
        if w <= 0.0 or raw is None:
            return None
        return w * raw

    def _get_train_sampler(self, dataset: Optional[torch.utils.data.Dataset] = None) -> Optional[torch.utils.data.Sampler]:
        dataset = dataset if dataset is not None else self.train_dataset
        if dataset is None or not has_length(dataset):
            return None

        if getattr(self.args, "group_by_compression_depth", False):
            depths = getattr(dataset, "compression_depths", None)
            if depths is not None and len(depths) == len(dataset):
                return LengthGroupedSampler(
                    self.args.train_batch_size,
                    world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                    lengths=list(depths),
                    group_by_modality=False,
                )
            if self.args.local_rank in (-1, 0):
                print("[trainer] group_by_compression_depth set but dataset has no usable "
                      "`compression_depths` (durations_json?); falling back to default sampler.")

        if self.args.group_by_modality_length:
            lengths = dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            compressor_lr = getattr(self.args, "compressor_lr", None)
            llm_lr = getattr(self.args, "llm_lr", None)
            mm_projector_lr = getattr(self.args, "mm_projector_lr", None)
            vision_encoder_lr = getattr(self.args, "vision_encoder_lr", None)
            optimized_parameters = [(n, p) for n, p in opt_model.named_parameters() if p.requires_grad]
            optimizer_grouped_parameters = []

            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            if llm_lr is not None and llm_lr > 0:
                compressor_name = "token_compressor"
                lm_parameters = [
                    name
                    for name, _ in optimized_parameters
                    if "vision_encoder" not in name
                    and "mm_projector" not in name
                    and (compressor_lr is None or compressor_lr <= 0 or compressor_name not in name)
                ]
                decay_lm_parameters = [name for name in lm_parameters if name in decay_parameters]
                nodecay_lm_parameters = [name for name in lm_parameters if name not in decay_parameters]
                optimizer_grouped_parameters.extend([
                    {
                        "params": [p for n, p in optimized_parameters if n in decay_lm_parameters],
                        "weight_decay": self.args.weight_decay,
                        "lr": llm_lr,
                    },
                    {
                        "params": [p for n, p in optimized_parameters if n in nodecay_lm_parameters],
                        "weight_decay": 0.0,
                        "lr": llm_lr,
                    }
                ])

            if compressor_lr is not None and compressor_lr > 0:
                qbase_lr = getattr(self.args, "qbase_lr", None)
                mamba_lr = getattr(self.args, "mamba_lr", None)
                # mamba_lr overrides compressor_lr as the fold/embed rate when set;
                # single-stage compressors (no "+mamba") never set it, so this is a
                # no-op fallback to compressor_lr for them.
                fold_lr = mamba_lr if (mamba_lr is not None and mamba_lr > 0) else compressor_lr
                compressor_parameters = [name for name, _ in optimized_parameters if "token_compressor" in name]
                # When the LLM is frozen (llm_lr=0) but embed_tokens was extended
                # for new compression tokens, route embed_tokens into this group so
                # ZeRO-2 always has optimizer state for it (avoids AllReduce NaN).
                # lm_head is excluded because it is frozen (new token rows are masked
                # in labels and never produce gradients).
                if llm_lr is None or llm_lr <= 0:
                    compressor_parameters += [
                        name for name, _ in optimized_parameters
                        if "embed_tokens" in name and name not in compressor_parameters
                    ]
                # Optional qbase/mamba LR split: the qbase (token_compressor.stage1.*)
                # gets its own group at qbase_lr; everything else (the Mamba-2/SSD
                # fold + embed_tokens) stays at fold_lr. qbase_lr <= 0 -> single group.
                use_qbase_group = qbase_lr is not None and qbase_lr > 0 and any(
                    "token_compressor.stage1" in n for n in compressor_parameters
                )
                if use_qbase_group:
                    qbase_parameters = [n for n in compressor_parameters if "token_compressor.stage1" in n]
                    rest_parameters = [n for n in compressor_parameters if n not in qbase_parameters]
                    for grp_names, grp_lr in ((rest_parameters, fold_lr), (qbase_parameters, qbase_lr)):
                        decay_grp = [n for n in grp_names if n in decay_parameters]
                        nodecay_grp = [n for n in grp_names if n not in decay_parameters]
                        optimizer_grouped_parameters.extend([
                            {
                                "params": [p for n, p in optimized_parameters if n in decay_grp],
                                "weight_decay": self.args.weight_decay,
                                "lr": grp_lr,
                            },
                            {
                                "params": [p for n, p in optimized_parameters if n in nodecay_grp],
                                "weight_decay": 0.0,
                                "lr": grp_lr,
                            },
                        ])
                    print(f"[create_optimizer] qbase/mamba LR split: {len(qbase_parameters)} qbase params @ "
                          f"{qbase_lr}, {len(rest_parameters)} fold/embed params @ {fold_lr}")
                else:
                    decay_compressor_parameters = [name for name in compressor_parameters if name in decay_parameters]
                    nodecay_compressor_parameters = [name for name in compressor_parameters if name not in decay_parameters]
                    optimizer_grouped_parameters.extend([
                        {
                            "params": [p for n, p in optimized_parameters if n in decay_compressor_parameters],
                            "weight_decay": self.args.weight_decay,
                            "lr": fold_lr,
                        },
                        {
                            "params": [p for n, p in optimized_parameters if n in nodecay_compressor_parameters],
                            "weight_decay": 0.0,
                            "lr": fold_lr,
                        }
                    ])

            if mm_projector_lr is not None and mm_projector_lr > 0:
                projector_parameters = [name for name, _ in optimized_parameters if "mm_projector" in name]
                decay_projector_parameters = [name for name in projector_parameters if name in decay_parameters]
                nodecay_projector_parameters = [name for name in projector_parameters if name not in decay_parameters]
                optimizer_grouped_parameters.extend([
                    {
                        "params": [p for n, p in optimized_parameters if n in decay_projector_parameters], 
                        "weight_decay": self.args.weight_decay,
                        "lr": mm_projector_lr,
                    },
                    {
                        "params": [p for n, p in optimized_parameters if n in nodecay_projector_parameters],
                        "weight_decay": 0.0,
                        "lr": mm_projector_lr,
                    }
                ])

            if vision_encoder_lr is not None and vision_encoder_lr > 0:
                vision_encoder_parameters = [name for name, _ in optimized_parameters if "vision_encoder" in name]
                decay_vision_encoder_parameters = [name for name in vision_encoder_parameters if name in decay_parameters]
                nodecay_vision_encoder_parameters = [name for name in vision_encoder_parameters if name not in decay_parameters]
                optimizer_grouped_parameters.extend([
                    {
                        "params": [p for n, p in optimized_parameters if n in decay_vision_encoder_parameters], 
                        "weight_decay": self.args.weight_decay,
                        "lr": vision_encoder_lr,
                    },
                    {
                        "params": [p for n, p in optimized_parameters if n in nodecay_vision_encoder_parameters],
                        "weight_decay": 0.0,
                        "lr": vision_encoder_lr,
                    }
                ])

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            # Remember the per-group LR THIS run's CLI args actually asked for, keyed by
            # group order -- see _load_optimizer_and_scheduler for why.
            self._configured_group_lrs = [g["lr"] for g in optimizer_grouped_parameters]

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _load_optimizer_and_scheduler(self, checkpoint):
        """Resuming (DeepSpeed's `deepspeed_load_checkpoint` -> `engine.load_checkpoint(...,
        load_optimizer_states=True, load_lr_scheduler_states=True)`, and -- on the deepspeed
        branch -- this base method's own extra `self.lr_scheduler.load_state_dict(...)`) both
        restore each optimizer param group's `lr`/`initial_lr` (and the scheduler's `base_lrs`)
        from the CHECKPOINTED values. That silently overrides whatever `create_optimizer()`
        just built from THIS run's CLI args a few lines earlier in `_inner_training_loop` --
        e.g. a deliberate `--qbase_lr` change across a cold-start -> resume launch pair (see
        `shell/pretrain_phase2_internvid.sh`'s documented staggered-start recipe). Confirmed:
        without this, a `QBASE_LR=1e-12` cold run followed by a same-`OUTPUT_DIR` resume at
        `QBASE_LR=1e-5` silently keeps training the qbase group at ~1e-12 for the entire
        resumed run -- no error, no warning, `token_compressor.stage1.*` ends up bit-identical
        to the warm start.

        Re-apply the CLI-configured per-group LR here, right after both restores have run, so
        a resume always trains at the LR this run was actually launched with. For a genuine
        crash-recovery resume (CLI unchanged from the interrupted run) this is a no-op --
        `_configured_group_lrs` already matches what got checkpointed."""
        super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint is None or self.optimizer is None:
            return
        lrs = getattr(self, "_configured_group_lrs", None)
        if not lrs:
            return
        groups = self.optimizer.param_groups
        if len(groups) != len(lrs):
            print(f"[post-resume LR reapply] optimizer has {len(groups)} param groups but "
                  f"{len(lrs)} configured LRs were recorded -- skipping (mismatched group "
                  f"layout vs the checkpoint?).")
            return
        for g, lr in zip(groups, lrs):
            g["lr"] = lr
            g["initial_lr"] = lr
        if hasattr(self.lr_scheduler, "base_lrs"):
            self.lr_scheduler.base_lrs = list(lrs)
        print(f"[post-resume LR reapply] reapplied this run's configured per-group LRs {lrs} "
              f"(overriding whatever the checkpoint's optimizer/scheduler state restored).")

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'is_alignment', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
            self.args.distributed_state.wait_for_everyone()
        else:
            # NOTE: Supporting save complete lora checkpoint during training.
            if self.args.lora_enable:
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)

                state_dict = get_peft_state_maybe_zero_3(self.model.named_parameters(), self.args.lora_bias)
                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters())
                if self.args.local_rank == 0 or self.args.local_rank == -1:
                    # save for acquring `config.json`
                    self.model.config.save_pretrained(output_dir)
                    # save for acquring `adapter_config.json`, `adapter_model.bin`
                    # self.model.save_pretrained(output_dir, state_dict=state_dict)
                    torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))

                # save for acquring lora adapter parameters & trainer states: `adapter_config.json`, `adapter_model.safetensors`
                super(VideoLLaMA3Trainer, self)._save_checkpoint(trial, metrics)
            else:
                super(VideoLLaMA3Trainer, self)._save_checkpoint(trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'is_alignment', False):
            pass
        else:
            super(VideoLLaMA3Trainer, self)._save(output_dir, state_dict)
