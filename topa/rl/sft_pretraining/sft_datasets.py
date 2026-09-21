from dataclasses import dataclass
from typing import Dict, List, Any
from contextlib import contextmanager

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .data_loading import (
    DataRoot,
    IGNORE_CONV_STATE_COLS,
    build_conv_state_label_maps,
    build_macro_action_map,
    build_micro_action_maps,
    build_conversation_text,
    build_micro_action_text,
    load_action_space,
    load_conversation_state_spec,
    build_macro_policy_context_text
)
from .utils import is_none_like, norm_str

@contextmanager
def _no_model_max_len_warning(tokenizer: Any):
    """Temporarily set tokenizer.model_max_length high to avoid warning spam during length-estimation tokenizations."""
    old = getattr(tokenizer, "model_max_length", None)
    try:
        # Large enough that we won't trigger the tokenizer's 'sequence length > model_max_length' warning
        tokenizer.model_max_length = 10**9
        yield
    finally:
        if old is not None:
            tokenizer.model_max_length = old

def _token_len(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    with _no_model_max_len_warning(tokenizer):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

def _manual_truncate_by_utterance(tokenizer: PreTrainedTokenizerBase, text: str, max_length: int, 
                                  protect_prefix: str | None = None) -> str:
    """Truncate by dropping whole utterance-lines from the *left* (keep most recent).

    If `protect_prefix` is provided and `text` starts with it, we never truncate that prefix.

    We first drop whole utterance-lines (fast approximation), then do a final exact length check
    with the tokenizer. If the remaining single utterance still doesn't fit, we truncate that
    utterance at token-level (keeping the tail), while still protecting the prefix.
    """
    special = tokenizer.num_special_tokens_to_add(pair=False)
    budget = max(1, max_length - special)

    def _enc_len(s: str) -> int:
        with _no_model_max_len_warning(tokenizer):
            return len(tokenizer(s, truncation=False, add_special_tokens=True, return_attention_mask=False)["input_ids"])

    def _finalize(prefix: str, kept_lines: list[str]) -> str:
        def build(prefix_: str, lines_: list[str]) -> str:
            if prefix_:
                if lines_:
                    return prefix_ + "".join(["\n" + ln for ln in lines_])
                return prefix_
            return "\n".join(lines_) if lines_ else ""

        out = build(prefix, kept_lines)

        # If tokenization ended up slightly longer than our approximation, drop more whole lines.
        while kept_lines and _enc_len(out) > max_length and len(kept_lines) > 1:
            kept_lines = kept_lines[1:]
            out = build(prefix, kept_lines)

        # If still too long and only one utterance remains -> token-truncate that utterance.
        if kept_lines and _enc_len(out) > max_length and len(kept_lines) == 1:
            prefix_sep = (prefix + "\n") if prefix else ""
            with _no_model_max_len_warning(tokenizer):
                prefix_ids = tokenizer(
                    prefix_sep, add_special_tokens=False, return_attention_mask=False
                )["input_ids"]
                combined_ids = tokenizer(
                    prefix_sep + kept_lines[0], add_special_tokens=False, return_attention_mask=False
                )["input_ids"]

            # Slice at the exact token boundary of the prefix
            conv_ids = combined_ids[len(prefix_ids) :]
            allowed = max_length - special - len(prefix_ids)
            if allowed <= 0:
                return prefix
            if len(conv_ids) > allowed:
                conv_ids = conv_ids[-allowed:]
            truncated = tokenizer.decode(conv_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).lstrip()
            if prefix:
                out = prefix_sep + ("… " + truncated if truncated else "")
            else:
                out = ("… " + truncated) if truncated else ""
        return out

    if protect_prefix is not None and text.startswith(protect_prefix):
        prefix = protect_prefix
        rest = text[len(prefix) :].lstrip("\n ")
        lines = [ln for ln in rest.splitlines() if ln.strip()]

        # Fast approximate length accounting (no specials)
        total = _token_len(tokenizer, prefix)
        lens = [_token_len(tokenizer, "\n" + ln) for ln in lines]
        total += sum(lens)

        i = 0
        while i < len(lines) and total > budget:
            total -= lens[i]
            i += 1
        kept = lines[i:]
        return _finalize(prefix, kept)

    # Generic path (no protected prefix)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text

    lens = [_token_len(tokenizer, ln + "\n") for ln in lines]
    total = sum(lens)

    i = 0
    while i < len(lines) and total > budget:
        total -= lens[i]
        i += 1
    kept = lines[i:]
    return _finalize("", kept)

def get_dim_names_from_conv_state_df(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in IGNORE_CONV_STATE_COLS]

class ConvStateDataset(Dataset):
    def __init__(self, root: DataRoot, data_json: list[dict], df: pd.DataFrame, max_turns: int = 64):
        self.root = root
        self.data_json = data_json
        self.df = df.reset_index(drop=True)
        self.max_turns = max_turns

        spec = load_conversation_state_spec(root)
        self.spec_dim_names, self.dim_to_label2id = build_conv_state_label_maps(spec)

        # We'll use only dims present in the CSV (in that order).
        self.dim_names = [d for d in self.spec_dim_names if d in df.columns]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx].to_dict()
        text = build_conversation_text(self.data_json, row, max_turns=self.max_turns)

        labels = []
        for dim in self.dim_names:
            v = row[dim]
            if is_none_like(v):
                v = "none"
            v = norm_str(v).lower()
            label2id = self.dim_to_label2id[dim]
            labels.append(label2id[v])

        return {"text": text, "labels": labels}

    @property
    def dim_num_labels(self) -> Dict[str, int]:
        return {d: len(self.dim_to_label2id[d]) for d in self.dim_names}

class MacroActionDataset(Dataset):
    def __init__(
        self,
        root: DataRoot,
        data_json: list[dict],
        df: pd.DataFrame,
        max_turns: int = 64,
        use_conv_states: bool = True,
        conv_state_encoding: str = "one_hot",
    ):
        self.root = root
        self.data_json = data_json
        self.df = df.reset_index(drop=True)
        self.max_turns = max_turns
        self.use_conv_states = use_conv_states
        self.conv_state_encoding = str(conv_state_encoding).strip().lower()

        self.action_space = load_action_space(root)
        self.macro2id = build_macro_action_map(self.action_space)

        # Conversation-state spec for building the vector representation.
        spec = load_conversation_state_spec(root)
        self.spec_dim_names, self.dim_to_label2id = build_conv_state_label_maps(spec)
        self.dim_names = [d for d in self.spec_dim_names if d in df.columns]

        # Precompute offsets for a single concatenated one-hot vector.
        self.dim_offsets: Dict[str, int] = {}
        off = 0
        for dim in self.dim_names:
            self.dim_offsets[dim] = off
            off += len(self.dim_to_label2id[dim])
        self.conv_state_dim = off if self.conv_state_encoding == "one_hot" else len(self.dim_names)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx].to_dict()
        text = build_macro_policy_context_text(self.data_json, row, max_turns=self.max_turns)

        out: Dict[str, Any] = {"text": text}
        if self.use_conv_states:
            vec = torch.zeros(self.conv_state_dim, dtype=torch.float32)
            for j, dim in enumerate(self.dim_names):
                v = row[dim]
                if is_none_like(v):
                    v = "none"
                v = norm_str(v).lower()
                label2id = self.dim_to_label2id[dim]
                cid = label2id[v]
                if self.conv_state_encoding == "one_hot":
                    vec[self.dim_offsets[dim] + cid] = 1.0
                else:
                    vec[j] = float(cid)
            out["conv_state_vec"] = vec

        label_name = row["selected_macro_action"]
        y = self.macro2id[label_name]
        out["labels"] = y
        return out

class MicroActionDataset(Dataset):
    def __init__(self, root: DataRoot, data_json: list[dict], df: pd.DataFrame, context_turns: int = 16, return_macro_action_id: bool = True):
        self.root = root
        self.data_json = data_json
        self.df = df.reset_index(drop=True)
        self.context_turns = context_turns
        self.return_macro_action_id = return_macro_action_id

        self.action_space = load_action_space(root)
        self.macro2id, self.macro_id_to_micro_label2id = build_micro_action_maps(self.action_space)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx].to_dict()
        text = build_micro_action_text(self.data_json, row, context_turns=self.context_turns)

        macro_name = row["selected_macro_action"]
        macro_id = self.macro2id[macro_name]

        micro_name = row["selected_micro_action"]
        micro2id = self.macro_id_to_micro_label2id[macro_id]
        micro_id = micro2id[micro_name]

        # # Append macro action name to condition the head selection and help encoder.
        # text = ctx + f"\n\nMACRO_ACTION: {macro_name}"

        item = {"text": text, "labels": micro_id}
        if self.return_macro_action_id:
            item["macro_action_id"] = macro_id
        return item

    @property
    def macro_id_to_num_micro(self) -> Dict[int, int]:
        return {mid: len(micro2id) for mid, micro2id in self.macro_id_to_micro_label2id.items()}

class FlatMicroActionDataset:
    """Dataset for flat micro policy."""

    def __init__(self, data_json: list[dict], df: pd.DataFrame, label2id: Dict[str, int], context_turns: int = 16):
        self.data_json = data_json
        self.df = df.reset_index(drop=True)
        self.label2id = label2id
        self.context_turns = context_turns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx].to_dict()
        text = build_micro_action_text(self.data_json, row, context_turns=self.context_turns)

        macro_name = row["selected_macro_action"]
        micro_name = row["selected_micro_action"]
        label = f"{macro_name}::{micro_name}"
        y = self.label2id[label]

        # # Keep the macro in text so the flat encoder has access to hierarchy context.
        # text = ctx + f"\n\nMACRO_ACTION: {macro_name}"

        return {"text": text, "labels": int(y)}

@dataclass
class SimpleCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 512
    include_macro_action_id: bool = True
    manual_truncate_utterances: bool = False
    protect_prefix: str | None = None

    def __call__(self, batch: List[dict]) -> dict:
        texts = [b["text"] for b in batch]
        if self.manual_truncate_utterances:
            texts = [
                _manual_truncate_by_utterance(
                    self.tokenizer, t, self.max_length, protect_prefix=self.protect_prefix
                )
                for t in texts
            ]

            # Assert manual truncation is sufficient: tokenizer truncation should NOT change the sequence.
            enc = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
            enc_trunc = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

            ids_a = enc["input_ids"]
            mask_a = enc["attention_mask"]
            ids_b = enc_trunc["input_ids"]
            mask_b = enc_trunc["attention_mask"]

            for i in range(ids_a.size(0)):
                la = int(mask_a[i].sum().item())
                lb = int(mask_b[i].sum().item())
                assert la == lb, (f"Manual truncation failed: len(no_trunc)={la} len(trunc)={lb} max_length={self.max_length}")
                assert torch.equal(ids_a[i, :la], ids_b[i, :lb]), ("Manual truncation mismatch vs tokenizer truncation.")
        else:
            enc = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

        out = dict(enc)
        if "labels" in batch[0]:
            labels = [b["labels"] for b in batch]
            out["labels"] = torch.tensor(labels, dtype=torch.long)

        if self.include_macro_action_id and "macro_action_id" in batch[0]:
            out["macro_action_id"] = torch.tensor([b["macro_action_id"] for b in batch], dtype=torch.long)

        if "conv_state_vec" in batch[0]:
            out["conv_state_vec"] = torch.stack([b["conv_state_vec"] for b in batch], dim=0)

        return out
