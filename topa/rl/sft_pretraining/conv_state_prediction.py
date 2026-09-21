from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer

from .data_loading import DataRoot, build_conversation_text, load_dataset_json
from .models import RobertaForConvStateMultiHead
from .utils import load_json


def _row_for_conv_state(row: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(row)
    if "to_idx" not in row:
        for col in ["activation_turn_idx", "turn_idx", "utterance_idx", "from_idx"]:
            if col in row:
                row["to_idx"] = int(row[col])
                break
    if "from_idx" not in row:
        row["from_idx"] = int(row["to_idx"])
    return row


def predict_conv_state_vectors_encoder(
    *,
    root: DataRoot,
    df: pd.DataFrame,
    model_name: str,
    adapter_dir: str | Path,
    meta_path: str | Path,
    max_length: int = 512,
    max_turns: int | None = None,
    batch_size: int = 64,
    encoding: str = "one_hot",
) -> Tuple[np.ndarray, List[str]]:
    encoding = str(encoding).strip().lower()
    meta = load_json(meta_path)
    dim_names = list(meta["dim_names"])
    dim_id2label = meta["dim_id2label"]
    dim_num_labels = {dim: len(dim_id2label[dim]) for dim in dim_names}

    offsets: Dict[str, int] = {}
    off = 0
    for dim in dim_names:
        offsets[dim] = off
        off += int(dim_num_labels[dim])

    data_json = load_dataset_json(root)
    texts = [
        build_conversation_text(data_json, _row_for_conv_state(row.to_dict()), max_turns=max_turns)
        for _, row in df.iterrows()
    ]

    tokenizer = AutoTokenizer.from_pretrained(str(model_name), use_fast=True)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"
    enc = tokenizer(texts, padding=True, truncation=True, max_length=int(max_length), return_tensors="pt")

    config = AutoConfig.from_pretrained(str(model_name))
    base = RobertaForConvStateMultiHead.from_pretrained(
        str(model_name),
        config=config,
        dim_num_labels=dim_num_labels,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    out_dim = off if encoding == "one_hot" else len(dim_names)
    vec = np.zeros((len(df), out_dim), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(df), int(batch_size)):
            end = start + int(batch_size)
            out = model(
                input_ids=enc["input_ids"][start:end].to(device),
                attention_mask=enc["attention_mask"][start:end].to(device),
            )
            logits_list = out["logits"] if isinstance(out, dict) else out.logits
            for j, dim in enumerate(dim_names):
                pred = torch.argmax(logits_list[j], dim=-1).detach().cpu().numpy()
                rows = np.arange(start, min(end, len(df)))
                if encoding == "one_hot":
                    vec[rows, offsets[dim] + pred] = 1.0
                else:
                    vec[rows, j] = pred.astype(np.float32)
    return vec, dim_names
