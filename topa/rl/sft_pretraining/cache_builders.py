import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm.auto import tqdm
import torch
from transformers import PreTrainedTokenizerBase

from .feature_cache import CacheKey, cache_or_build, fingerprint_dict
from .cached_datasets import CachedFeatures
from .data_loading import DataRoot, load_dataset_json, get_default_prefix
from .sft_datasets import ConvStateDataset, MacroActionDataset, MicroActionDataset, FlatMicroActionDataset, _manual_truncate_by_utterance

def _truncate_one(text, tokenizer, max_length, protect_prefix):
    return _manual_truncate_by_utterance(tokenizer, text, max_length, protect_prefix)

def _tokenize(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int,
    manual_truncate_utterances: bool = True,
    protect_prefix: str | None = None,
    parallel_tokenization: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if manual_truncate_utterances:
        if parallel_tokenization:
            truncate_fn = partial(
                _truncate_one, tokenizer=tokenizer, max_length=max_length, protect_prefix=protect_prefix
            )
            n_workers = os.cpu_count() // 2
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                texts = list(
                    tqdm(
                        ex.map(truncate_fn, texts, chunksize=32),
                        total=len(texts),
                        desc=f"[cache] truncate-by-utterance using {n_workers} cores.",
                        unit=" ex",
                    )
                )
        else:
            texts = [
                _manual_truncate_by_utterance(tokenizer, t, max_length, protect_prefix=protect_prefix)
                for t in tqdm(texts, desc="[cache] truncate-by-utterance", unit=" ex")
            ]

        # Assert manual truncation is sufficient: tokenizer truncation should NOT change the sequence.
        enc = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        enc_trunc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

        ids_a = enc["input_ids"]
        mask_a = enc["attention_mask"]
        ids_b = enc_trunc["input_ids"]
        mask_b = enc_trunc["attention_mask"]

        for i in range(ids_a.size(0)):
            la = int(mask_a[i].sum().item())
            lb = int(mask_b[i].sum().item())
            assert la == lb, (f"Manual truncation failed: len(no_trunc)={la} len(trunc)={lb} max_length={max_length}")
            assert torch.equal(ids_a[i, :la], ids_b[i, :lb]), ("Manual truncation mismatch vs tokenizer truncation.")

        return ids_a.cpu().numpy(), mask_a.cpu().numpy()

    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    return enc["input_ids"], enc["attention_mask"]

def build_cached_conv_state(
    cache_dir: Path,
    root: DataRoot,
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    max_turns: int,
    force_rebuild: bool = False,
) -> CachedFeatures:
    data_json = load_dataset_json(root)

    meta = {
        "kind": "conv_state",
        "cache_version": 2,
        "n": len(df),
        "max_length": max_length,
        "max_turns": max_turns,
        "df_cols": ",".join([str(c) for c in df.columns]),
        "tokenizer": getattr(tokenizer, "name_or_path", "tok"),
        "csv": "updated_annotations_conv_state.csv",
    }
    key = CacheKey("conv_state", fingerprint_dict(meta))

    def _build() -> CachedFeatures:
        ds = ConvStateDataset(root=root, data_json=data_json, df=df, max_turns=max_turns)
        texts = []
        labels = []
        for i in tqdm(range(len(ds)), desc="[cache] build conv_state", unit=" ex"):
            item = ds[i]
            texts.append(item["text"])
            labels.append(item["labels"])
        input_ids, attention_mask = _tokenize(tokenizer, texts, max_length=max_length, manual_truncate_utterances=True, protect_prefix=None)
        labels_np = np.asarray(labels, dtype=np.int64)
        feats = CachedFeatures(input_ids=input_ids, attention_mask=attention_mask, labels=labels_np, extra={})
        # Persist conv-state label metadata so training can build multi-head classifier without re-parsing spec.
        feats.dim_names = list(ds.dim_names)
        feats.dim_to_label2id = ds.dim_to_label2id
        feats.dim_num_labels = ds.dim_num_labels
        feats.spec_dim_names = list(ds.spec_dim_names)
        return feats

    return cache_or_build(cache_dir, key, _build, force_rebuild=force_rebuild)

def build_cached_macro(
    cache_dir: Path,
    root: DataRoot,
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    max_turns: int,
    use_conv_states: bool,
    conv_state_encoding: str = "one_hot",
    force_rebuild: bool = False,
) -> CachedFeatures:
    data_json = load_dataset_json(root)

    meta = {
        "kind": "macro",
        "n": len(df),
        "max_length": max_length,
        "max_turns": max_turns,
        "use_conv_states": int(use_conv_states),
        "conv_state_encoding": str(conv_state_encoding),
        "tokenizer": getattr(tokenizer, "name_or_path", "tok"),
        "csv": "annotations_macro_actions.csv",
    }
    key = CacheKey("macro", fingerprint_dict(meta))

    def _build() -> CachedFeatures:
        ds = MacroActionDataset(
            root=root,
            data_json=data_json,
            df=df,
            max_turns=max_turns,
            use_conv_states=use_conv_states,
            conv_state_encoding=str(conv_state_encoding),
        )
        texts = []
        labels = []
        extra: Dict[str, Any] = {}
        conv_vecs = []
        for i in tqdm(range(len(ds)), desc="[cache] build macro", unit=" ex"):
            item = ds[i]
            texts.append(item["text"])
            labels.append(item["labels"])
            if "conv_state_vec" in item:
                conv_vecs.append(item["conv_state_vec"].numpy())
        input_ids, attention_mask = _tokenize(tokenizer, texts, max_length=max_length, manual_truncate_utterances=True, protect_prefix=get_default_prefix())
        labels_np = np.asarray(labels, dtype=np.int64)
        if conv_vecs:
            extra["conv_state_vec"] = np.asarray(conv_vecs, dtype=np.float32)
        return CachedFeatures(input_ids=input_ids, attention_mask=attention_mask, labels=labels_np, extra=extra)

    return cache_or_build(cache_dir, key, _build, force_rebuild=force_rebuild)

def build_cached_micro_independent(
    cache_dir: Path,
    root: DataRoot,
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    context_turns: int,
    force_rebuild: bool = False,
) -> CachedFeatures:
    """Cache micro features for independent micro policy training.

    IMPORTANT: df must already be filtered to a single macro action.
    """
    data_json = load_dataset_json(root)

    macro_name = str(df["selected_macro_action"].iloc[0]) if len(df) else "EMPTY"
    meta = {
        "kind": "micro_independent",
        "macro": macro_name,
        "n": len(df),
        "max_length": max_length,
        "context_turns": context_turns,
        "tokenizer": getattr(tokenizer, "name_or_path", "tok"),
        "csv": "annotations_micro_actions_refined.csv",
    }
    key = CacheKey("micro_independent", fingerprint_dict(meta))

    def _build() -> CachedFeatures:
        ds = MicroActionDataset(root=root, data_json=data_json, df=df, context_turns=context_turns, return_macro_action_id=False)
        texts = []
        labels = []
        for i in tqdm(range(len(ds)), desc=f"[cache] build micro_indep ({macro_name})", unit=" ex"):
            item = ds[i]
            texts.append(item["text"])
            labels.append(item["labels"])
        input_ids, attention_mask = _tokenize(tokenizer, texts, max_length=max_length, manual_truncate_utterances=True, protect_prefix=get_default_prefix())
        labels_np = np.asarray(labels, dtype=np.int64)
        return CachedFeatures(input_ids=input_ids, attention_mask=attention_mask, labels=labels_np, extra={})

    return cache_or_build(cache_dir, key, _build, force_rebuild=force_rebuild)

def build_cached_micro_flat(
    cache_dir: Path,
    df: pd.DataFrame,
    data_json: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    label2id: Dict[str, int],
    context_turns: int,
    force_rebuild: bool = False,
) -> CachedFeatures:
    meta = {
        "kind": "micro_flat",
        "n": len(df),
        "max_length": max_length,
        "context_turns": context_turns,
        "tokenizer": getattr(tokenizer, "name_or_path", "tok"),
        "labels": len(label2id),
    }
    key = CacheKey("micro_flat", fingerprint_dict(meta))

    def _build() -> CachedFeatures:
        ds = FlatMicroActionDataset(data_json=data_json, df=df, label2id=label2id, context_turns=context_turns)
        texts = []
        labels = []
        for i in tqdm(range(len(ds)), desc="[cache] build micro_flat", unit=" ex"):
            item = ds[i]
            texts.append(item["text"])
            labels.append(item["labels"])
        input_ids, attention_mask = _tokenize(tokenizer, texts, max_length=max_length, manual_truncate_utterances=True, protect_prefix=get_default_prefix())
        labels_np = np.asarray(labels, dtype=np.int64)
        return CachedFeatures(input_ids=input_ids, attention_mask=attention_mask, labels=labels_np, extra={})

    return cache_or_build(cache_dir, key, _build, force_rebuild=force_rebuild)
