import os
import re
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from transformers import Trainer
from transformers.trainer_callback import PrinterCallback, ProgressCallback

def set_seed(seed: int = 42) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_torch_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g

def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def norm_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    return " ".join(s.strip().split())

def safe_name(s: str) -> str:
    s = norm_str(str(s))
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    return s.strip("_")

def is_none_like(x: Any) -> bool:
    if x is None:
        return True
    s = str(x).strip().lower()
    return s == "" or s == "none" or s == "nan"

def parse_turn_id(x: Any) -> int | None:
    if x is None or pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return int(float(s))
    except ValueError:
        pass
    m = re.search(r"(\d+)(?:\.0+)?$", s)
    if m:
        return int(m.group(1))
    return None

def add_preferred_turn_index(
    df: pd.DataFrame,
    out_col: str,
    id_cols: Sequence[str],
    fallback_cols: Sequence[str],
) -> pd.DataFrame:
    for col in id_cols:
        if col in df.columns:
            parsed = df[col].map(parse_turn_id)
            if parsed.notna().any():
                df[out_col] = pd.to_numeric(parsed, errors="coerce")
                return df
    for col in fallback_cols:
        if col in df.columns:
            df[out_col] = pd.to_numeric(df[col], errors="coerce")
            return df
    raise ValueError(f"Cannot build {out_col}; missing id columns {list(id_cols)} and fallback columns {list(fallback_cols)}")

def termination_labels_from_options(option_ids: Sequence[Any]) -> list[int]:
    return [
        1 if i == len(option_ids) - 1 else int(option_ids[i + 1] != option_ids[i])
        for i in range(len(option_ids))
    ]

_SYSTEM_LABEL = "system"
_USER_LABEL = "user"

def set_role_labels(system_label: str | None, user_label: str | None) -> None:
    global _SYSTEM_LABEL, _USER_LABEL
    if system_label:
        _SYSTEM_LABEL = str(system_label).strip() or _SYSTEM_LABEL
    if user_label:
        _USER_LABEL = str(user_label).strip() or _USER_LABEL

def role_labels() -> tuple[str, str]:
    return _SYSTEM_LABEL, _USER_LABEL

def speaker_role(raw: str) -> str:
    # Dataset uses 'system' and 'user' speakers; allow configured labels too.
    s = raw.strip().lower()
    system_label, user_label = role_labels()
    sys_l = system_label.strip().lower()
    usr_l = user_label.strip().lower()
    if s in {sys_l, "system", "assistant"}:
        return system_label
    if s in {usr_l, "user", "human"}:
        return user_label
    raise Exception(f"Unknown speaker {s}")

def format_dialogue(dialogue: List[Dict[str, str]], start_idx: int, end_idx: int, max_turns: int | None = None) -> str:
    # inclusive indices
    start_idx = max(0, start_idx)
    end_idx = min(len(dialogue) - 1, end_idx)
    turns = dialogue[start_idx : end_idx + 1]
    if max_turns is not None and len(turns) > max_turns:
        turns = turns[-max_turns:]
    lines = []
    for t in turns:
        role = speaker_role(t["speaker"])
        text = norm_str(t["text"])
        if text:
            lines.append(f"{role.upper()}: {text}")
    return "\n".join(lines)

def topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    """Compute top-k accuracy for multiclass logits.

    If num_classes <= k, we return standard accuracy (top-1) to avoid a trivial 1.0.
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected logits shape (N,C), got {logits.shape}")
    n, c = logits.shape
    k_eff = int(min(k, c))
    # Avoid trivial top-k when c <= k (e.g., binary / 3-way)
    if c <= k:
        preds = logits.argmax(axis=-1)
        return float((preds == labels).mean())
    topk = np.argpartition(logits, -k_eff, axis=-1)[:, -k_eff:]
    ok = (topk == labels.reshape(-1, 1)).any(axis=1)
    return float(ok.mean())

def remove_logging(trainer: Trainer, show_eval: bool = True):
    # Silence terminal logging (keep TensorBoard)
    trainer.remove_callback(PrinterCallback)
    trainer.remove_callback(ProgressCallback)
    if show_eval:
        class EvalOnlyPrinterCallback(PrinterCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs or not state.is_world_process_zero:
                    return control

                # Only print eval metrics (keys like eval_loss, eval_accuracy, ...)
                eval_logs = {k: v for k, v in logs.items() if k.startswith("eval_")}
                if eval_logs:
                    return super().on_log(args, state, control, logs=eval_logs, **kwargs)
                return control
        class ProgressBarNoTrainPostfixCallback(ProgressCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                # Don’t put training loss/metrics in the tqdm postfix.
                # Still allow eval logs (optional) if you want them in the bar.
                if logs and any(k.startswith("eval_") for k in logs):
                    return super().on_log(args, state, control, logs=logs, **kwargs)
                return control

        # Remove default terminal logging + default progress callback
        trainer.add_callback(ProgressBarNoTrainPostfixCallback)
        trainer.add_callback(EvalOnlyPrinterCallback)
    else:
        class ProgressBarNoMetricsCallback(ProgressCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                # Keep the bar, but don't show any metrics/loss in the bar postfix.
                # (Progress updates still happen via on_step_end in ProgressCallback.)
                return control
        # Replace progress callback to avoid postfix (loss/metrics)
        trainer.add_callback(ProgressBarNoMetricsCallback)

def resolve_mixed_precision(amp: str) -> tuple[bool, bool, str]:
    """Resolve mixed precision flags for Hugging Face Trainer.
    Args:
        amp: one of {none, fp16, bf16, auto}
    Returns:
        (fp16, bf16, resolved_mode)
    Notes:
        - Tesla T4 does not support BF16 -> auto resolves to FP16.
        - A6000 (Ampere) supports BF16 -> auto resolves to BF16.
        - If BF16 is requested but not supported, we fall back to FP16 (if CUDA) or none.
    """
    req = (amp or 'auto').strip().lower()
    if req not in {'none', 'fp16', 'bf16', 'auto'}:
        raise ValueError(f'Invalid amp={amp!r}. Expected none|fp16|bf16|auto.')

    if not torch.cuda.is_available():
        return (False, False, 'none')
    bf16_ok = bool(getattr(torch.cuda, 'is_bf16_supported', lambda: False)())
    if req == 'auto':
        return (False, True, 'bf16') if bf16_ok else (True, False, 'fp16')
    if req == 'bf16':
        if bf16_ok:
            return (False, True, 'bf16')
        # fallback
        return (True, False, 'fp16')
    if req == 'fp16':
        return (True, False, 'fp16')
    return (False, False, 'none')

def split_df_user(df: pd.DataFrame, user_split: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_users = set(user_split["train_user_idxs"])
    val_users = set(user_split["val_user_idxs"])
    train_df = df[df["user_idx"].isin(train_users)].copy()
    val_df = df[df["user_idx"].isin(val_users)].copy()
    return train_df, val_df

def split_df_stratify(df: pd.DataFrame, label_col: str, val_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].astype(str).to_numpy())
    class_names = list(le.classes_)

    idx = np.arange(len(df))

    # # If any class is too small, fall back to non-stratified split.
    # counts = np.bincount(y)
    # can_stratify = (counts.min() >= 2) and (len(idx) * val_frac >= len(class_names))

    # rare_actions = df[label_col].astype(str).value_counts()
    # rare_action_names = rare_actions[rare_actions < 2].index.tolist()

    # print("Actions with < 2 samples:", rare_action_names)

    # if can_stratify:
    train_idx, val_idx = train_test_split(idx, test_size=val_frac, random_state=seed, stratify=y)
    # else:
    #     train_idx, val_idx = train_test_split(idx, test_size=val_frac, random_state=seed, shuffle=True)

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    return train_df, val_df, class_names
