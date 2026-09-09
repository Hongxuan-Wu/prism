"""Frozen checkpoint inference; no training or optimizer code.

Adapted from the held-out prototype feature extraction used in PRISM.
Checkpoint-defined Python code runs only after explicit caller approval.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

ORIGINAL_TRANSFORMER_TOKEN_LIMIT = 100
ORIGINAL_EVO_CHARACTER_LIMIT = 400


def load_base_backbone(checkpoint):
    options = {"trust_remote_code": True, "local_files_only": True}
    try:
        return AutoModel.from_pretrained(checkpoint, **options), "AutoModel"
    except ValueError as error:
        if "Unrecognized configuration class" not in str(error):
            raise
    masked_lm = AutoModelForMaskedLM.from_pretrained(checkpoint, **options)
    backbone = getattr(masked_lm, "base_model", None)
    if not isinstance(backbone, torch.nn.Module) or backbone is masked_lm:
        raise ValueError("Masked-LM checkpoint lacks a distinct base model")
    return backbone, "AutoModelForMaskedLM.base_model"


def load_feature_model(model_id: str, checkpoint: Path, device: torch.device, *, trust_checkpoint_code: bool = False) -> tuple[torch.nn.Module, object | None, dict[str, object]]:
    if not trust_checkpoint_code:
        raise ValueError("Review the local checkpoint code, then explicitly trust it")
    if model_id == "evo8k":
        package_name = f"_prism_evo_{hashlib.sha256(str(checkpoint).encode()).hexdigest()[:12]}"
        package = types.ModuleType(package_name)
        package.__path__ = [str(checkpoint)]
        sys.modules[package_name] = package
        config_class = importlib.import_module(f"{package_name}.configuration_hyena").StripedHyenaConfig
        model_class = importlib.import_module(f"{package_name}.modeling_hyena").StripedHyenaModelForCausalLM
        config = config_class.from_pretrained(checkpoint, local_files_only=True)
        config.use_cache = False
        wrapper = model_class.from_pretrained(checkpoint, config=config, local_files_only=True)
        backbone = getattr(wrapper, "backbone", None)
        unembed = getattr(backbone, "unembed", None)
        if not isinstance(backbone, torch.nn.Module) or not callable(getattr(unembed, "unembed", None)):
            raise ValueError("Evo checkpoint does not expose the expected backbone/unembed")
        unembed.unembed = lambda hidden: hidden
        model, tokenizer, loader = backbone, None, "AutoModelForCausalLM.backbone_without_unembed"
    elif model_id == "ntv2_500m":
        from transformers import modeling_utils, pytorch_utils

        for name in ("find_pruneable_heads_and_indices", "prune_linear_layer"):
            if not hasattr(modeling_utils, name):
                setattr(modeling_utils, name, getattr(pytorch_utils, name))
        masked_lm = AutoModelForMaskedLM.from_pretrained(
            checkpoint, trust_remote_code=True, local_files_only=True
        )
        model = getattr(masked_lm, "base_model", None)
        if not isinstance(model, torch.nn.Module) or model is masked_lm:
            raise ValueError("NT-v2 masked-LM checkpoint does not expose a distinct base model")
        loader = "AutoModelForMaskedLM.base_model"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    else:
        model, loader = load_base_backbone(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
        module = sys.modules.get(model.__class__.__module__)
        if module is not None and hasattr(module, "flash_attn_qkvpacked_func"):
            setattr(module, "flash_attn_qkvpacked_func", None)
    model.requires_grad_(False).eval().to(device)
    return model, tokenizer, {
        "loader": loader,
        "pooling": "hidden_states[:,0,:]",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def extract_features(
    model_id: str,
    model: torch.nn.Module,
    tokenizer: object | None,
    sequences: list[str],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if not sequences or batch_size < 1:
        raise ValueError("feature extraction requires sequences and a positive batch size")
    parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            if model_id == "evo8k":
                input_ids = torch.tensor(
                    [[ord(character) for character in sequence[:ORIGINAL_EVO_CHARACTER_LIMIT]] for sequence in batch],
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.ones_like(input_ids)
                with (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else contextlib.nullcontext()
                ):
                    outputs = model(input_ids, padding_mask=attention_mask)
            else:
                if tokenizer is None:
                    raise ValueError("Transformer tokenizer is missing")
                encoded = tokenizer(
                    text=batch,
                    return_tensors="pt",
                    max_length=ORIGINAL_TRANSFORMER_TOKEN_LIMIT,
                    padding=True,
                    truncation=True,
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                with (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if device.type == "cuda"
                    else contextlib.nullcontext()
                ):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None and isinstance(outputs, (tuple, list)) and outputs:
                hidden = outputs[0]
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise ValueError(f"{model_id} output lacks token hidden states")
            parts.append(hidden[:, 0, :].float().cpu().numpy())
    features = np.concatenate(parts)
    if features.shape[0] != len(sequences) or not np.all(np.isfinite(features)):
        raise ValueError("feature extraction produced invalid embeddings")
    return features
