from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, PreTrainedTokenizerBase

from ..sft_pretraining.activation_sft_utils import build_conv_state_vectors
from ..sft_pretraining.cache_builders import _tokenize
from ..sft_pretraining.conv_state_prediction import predict_conv_state_vectors_encoder
from ..sft_pretraining.data_loading import (
    DataRoot,
    build_flat_micro_label_map,
    build_macro_action_map,
    build_micro_action_maps,
    build_micro_action_text,
    get_default_prefix,
    load_action_space,
    load_dataset_json,
)
from ..sft_pretraining.feature_cache import CacheKey, cache_or_build, fingerprint_dict
from ..sft_pretraining.utils import add_preferred_turn_index, termination_labels_from_options
from .activation_rollouts import OfflineRolloutFeatures, _compute_returns
from .reward_loading import ExtrinsicConfig, load_extrinsic_reward_csv, load_intrinsic_reward_csv


class _CLSModel(torch.nn.Module):
    def __init__(self, encoder_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(str(encoder_name))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return out.last_hidden_state[:, 0, :]


def _encode_texts(
    texts: List[str],
    tokenizer: PreTrainedTokenizerBase,
    encoder_name: str,
    max_length: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_ids, attention_mask = _tokenize(
        tokenizer,
        texts,
        max_length=int(max_length),
        manual_truncate_utterances=True,
        protect_prefix=get_default_prefix(),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    model = _CLSModel(str(encoder_name)).to(device).eval()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    ids = torch.tensor(input_ids, dtype=torch.long)
    mask = torch.tensor(attention_mask, dtype=torch.long)
    vecs: List[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), int(batch_size)), desc=f"[cache] encode {encoder_name}", unit="batch"):
            end = start + int(batch_size)
            cls = model(input_ids=ids[start:end].to(device), attention_mask=mask[start:end].to(device))
            vecs.append(cls.detach().float().cpu().numpy())
    return input_ids, attention_mask, np.concatenate(vecs, axis=0).astype(np.float32)


def _with_turn_idx(df: pd.DataFrame) -> pd.DataFrame:
    df["user_idx"] = df["user_idx"].astype(int)
    df["session_idx"] = df["session_idx"].astype(int)
    df = add_preferred_turn_index(df, "turn_idx", ["utterance_id", "turn_id", "from_id", "to_id"], ["utterance_idx"])
    df = df.dropna(subset=["turn_idx"]).reset_index(drop=True)
    df["turn_idx"] = df["turn_idx"].astype(int)
    df["utterance_idx"] = df["turn_idx"]
    df["row_i"] = np.arange(len(df), dtype=np.int64)
    return df


def build_cached_option_utterance_rollout_encoder(
    cache_dir: Path,
    root: DataRoot,
    tokenizer: PreTrainedTokenizerBase,
    encoder_name: str,
    max_length: int,
    context_turns: int | None,
    gamma: float,
    use_conv_state: bool,
    extr_cfg: ExtrinsicConfig,
    encode_batch_size: int,
    conv_state_adapter_dir: str = "",
    conv_state_meta_path: str = "",
    conv_state_encoding: str = "one_hot",
    force_rebuild: bool = False,
) -> OfflineRolloutFeatures:
    meta = {
        "kind": "option_utterance_offline_encoder",
        "encoder_name": str(encoder_name),
        "max_length": int(max_length),
        "context_turns": context_turns,
        "gamma": float(gamma),
        "use_conv_state": int(bool(use_conv_state)),
        "conv_state_source": str(conv_state_adapter_dir),
        "threshold": float(extr_cfg.threshold),
        "high_threshold": float(extr_cfg.high_threshold),
        "reward_mapping": "0_below_threshold_1_below_high_2_at_high",
        "extr_reward_key": str(extr_cfg.reward_key or ""),
        "indexing": "preferred_id_turn_index",
        "termination_labeling": "next_option_change_only",
        "conv_state_encoding": str(conv_state_encoding),
    }
    key = CacheKey("option_utterance_offline_encoder", fingerprint_dict(meta))

    def _build() -> OfflineRolloutFeatures:
        data_json = load_dataset_json(root)
        df = pd.read_csv(root.annotations_dir / "annotations_micro_actions_refined.csv")
        df = _with_turn_idx(df)
        df = df.sort_values(["user_idx", "session_idx", "turn_idx"]).reset_index(drop=True)
        df["row_i"] = np.arange(len(df), dtype=np.int64)

        action_space = load_action_space(root)
        macro2id = build_macro_action_map(action_space)
        num_options = int(max(macro2id.values()) + 1)

        extra_vecs = None
        conv_state_dim = 0
        if use_conv_state:
            if str(conv_state_adapter_dir).strip() and Path(str(conv_state_adapter_dir)).exists():
                extra_vecs, _ = predict_conv_state_vectors_encoder(
                    root=root,
                    df=df,
                    model_name=str(encoder_name),
                    adapter_dir=str(conv_state_adapter_dir),
                    meta_path=str(conv_state_meta_path),
                    max_length=int(max_length),
                    max_turns=context_turns,
                    batch_size=int(encode_batch_size),
                    encoding=str(conv_state_encoding),
                )
            else:
                extra_vecs, _ = build_conv_state_vectors(root, df, encoding=str(conv_state_encoding))
            conv_state_dim = int(extra_vecs.shape[1])

        df_extr = load_extrinsic_reward_csv(root.annotations_dir / "annotations_extrinsic_reward.csv", cfg=extr_cfg)
        extr_map = {
            (int(r["user_idx"]), int(r["session_idx"])): float(r["extrinsic_reward"])
            for r in df_extr.to_dict(orient="records")
        }

        texts: List[str] = []
        extra_rows: List[np.ndarray] = []
        actions: List[int] = []
        rewards: List[float] = []
        dones: List[int] = []
        term_labels: List[int] = []
        episode_ids: List[int] = []

        ep = 0
        for (u, s), sub in tqdm(df.groupby(["user_idx", "session_idx"], sort=False), desc="[cache] build option offline (encoder)", unit="sess"):
            u = int(u)
            s = int(s)
            local_texts: List[str] = []
            local_extra: List[np.ndarray] = []
            local_actions: List[int] = []
            for _, row in sub.iterrows():
                macro_name = str(row["selected_macro_action"])
                if macro_name not in macro2id:
                    continue
                local_texts.append(build_micro_action_text(data_json, row.to_dict(), context_turns=context_turns))
                if extra_vecs is not None:
                    local_extra.append(extra_vecs[int(row["row_i"])])
                local_actions.append(int(macro2id[macro_name]))

            if not local_actions:
                continue

            local_terms = termination_labels_from_options(local_actions)
            for i, action_id in enumerate(local_actions):
                done = int(i == len(local_actions) - 1)
                texts.append(local_texts[i])
                if extra_vecs is not None:
                    extra_rows.append(local_extra[i])
                actions.append(action_id)
                rewards.append(float(extr_map[(u, s)]) if done else 0.0)
                dones.append(done)
                term_labels.append(local_terms[i])
                episode_ids.append(ep)
            ep += 1

        input_ids, attention_mask, input_embeds = _encode_texts(texts, tokenizer, encoder_name, max_length, encode_batch_size)
        latent_dim = int(input_embeds.shape[1])
        if extra_vecs is not None:
            input_embeds = np.concatenate([input_embeds, np.asarray(extra_rows, dtype=np.float32)], axis=1)

        actions_np = np.asarray(actions, dtype=np.int64)
        rewards_np = np.asarray(rewards, dtype=np.float32)
        dones_np = np.asarray(dones, dtype=np.int64)
        extra = {
            "input_embeds": input_embeds.astype(np.float32),
            "latent_dim": int(latent_dim),
            "conv_state_dim": int(conv_state_dim),
            "actions": actions_np,
            "rewards": rewards_np,
            "dones": dones_np,
            "returns": _compute_returns(rewards_np, dones_np, gamma=float(gamma)),
            "term_labels": np.asarray(term_labels, dtype=np.float32),
            "episode_id": np.asarray(episode_ids, dtype=np.int64),
            "label2id": macro2id,
            "num_labels": num_options,
        }
        return OfflineRolloutFeatures(input_ids=input_ids, attention_mask=attention_mask, labels=actions_np.copy(), extra=extra)

    feat = cache_or_build(cache_dir, key, _build, force_rebuild=force_rebuild)
    feat.extra["_cache_key"] = key.filename()
    feat.extra["_cache_dir"] = str(Path(cache_dir).resolve())
    feat.extra["_cache_meta"] = dict(meta)
    return feat


def build_cached_intra_option_rollout_encoder(
    cache_dir: Path,
    root: DataRoot,
    tokenizer: PreTrainedTokenizerBase,
    encoder_name: str,
    max_length: int,
    context_turns: int | None,
    macro_action: str,
    gamma: float,
    use_conv_state: bool = False,
    conv_state_encoding: str = "one_hot",
    intrinsic_center: bool = True,
    encode_batch_size: int = 16,
    force_rebuild: bool = False,
) -> OfflineRolloutFeatures:
    meta = {
        "kind": "intra_option_offline_encoder",
        "macro_action": str(macro_action),
        "encoder_name": str(encoder_name),
        "max_length": int(max_length),
        "context_turns": context_turns,
        "gamma": float(gamma),
        "use_conv_state": int(bool(use_conv_state)),
        "intrinsic_center": int(bool(intrinsic_center)),
        "indexing": "preferred_id_turn_index",
        "conv_state_encoding": str(conv_state_encoding),
    }
    key = CacheKey("intra_option_offline_encoder", fingerprint_dict(meta))

    def _build() -> OfflineRolloutFeatures:
        data_json = load_dataset_json(root)
        df_micro = pd.read_csv(root.annotations_dir / "annotations_micro_actions_refined.csv")
        df_micro = _with_turn_idx(df_micro)
        df_micro = df_micro[df_micro["selected_macro_action"].astype(str) == str(macro_action)].reset_index(drop=True)
        df_micro["row_i"] = np.arange(len(df_micro), dtype=np.int64)

        extra_vecs = None
        conv_state_dim = 0
        if use_conv_state:
            extra_vecs, _ = build_conv_state_vectors(root, df_micro, encoding=str(conv_state_encoding))
            conv_state_dim = int(extra_vecs.shape[1])

        action_space = load_action_space(root)
        macro2id, macro_id_to_micro2id = build_micro_action_maps(action_space)
        mid = macro2id[str(macro_action)]
        micro2id = macro_id_to_micro2id[mid]

        df_intr = load_intrinsic_reward_csv(root.annotations_dir / "annotations_intrinsic_reward.csv")
        df_intr = df_intr[df_intr["selected_macro_action"].astype(str) == str(macro_action)].copy()
        df_intr = add_preferred_turn_index(df_intr, "from_turn_idx", ["from_id"], ["from_idx"])
        df_intr = add_preferred_turn_index(df_intr, "to_turn_idx", ["to_id"], ["to_idx"])
        df_intr = df_intr.dropna(subset=["from_turn_idx", "to_turn_idx"]).reset_index(drop=True)
        df_intr["from_turn_idx"] = df_intr["from_turn_idx"].astype(int)
        df_intr["to_turn_idx"] = df_intr["to_turn_idx"].astype(int)
        df_intr = df_intr.sort_values(["user_idx", "session_idx", "from_turn_idx", "to_turn_idx"]).reset_index(drop=True)

        texts: List[str] = []
        extra_rows: List[np.ndarray] = []
        actions: List[int] = []
        rewards: List[float] = []
        dones: List[int] = []
        episode_ids: List[int] = []

        ep = 0
        for pr in tqdm(df_intr.to_dict(orient="records"), desc=f"[cache] build intra-option offline (encoder) ({macro_action})", unit="phase"):
            u = int(pr["user_idx"])
            s = int(pr["session_idx"])
            from_i = int(pr["from_turn_idx"])
            to_i = int(pr["to_turn_idx"])
            sub = df_micro[
                (df_micro["user_idx"] == u)
                & (df_micro["session_idx"] == s)
                & (df_micro["turn_idx"] >= from_i)
                & (df_micro["turn_idx"] <= to_i)
            ].copy()
            if len(sub) == 0:
                continue
            sub = sub.sort_values("turn_idx").reset_index(drop=True)
            r_phase = float(pr["reward"])
            if intrinsic_center:
                r_phase = r_phase - 1.0

            local_texts: List[str] = []
            local_extra: List[np.ndarray] = []
            local_actions: List[int] = []
            for _, row in sub.iterrows():
                micro_name = str(row["selected_micro_action"])
                if micro_name not in micro2id:
                    continue
                local_texts.append(build_micro_action_text(data_json, row.to_dict(), context_turns=context_turns))
                if extra_vecs is not None:
                    local_extra.append(extra_vecs[int(row["row_i"])])
                local_actions.append(int(micro2id[micro_name]))

            for i, action_id in enumerate(local_actions):
                is_last = i == len(local_actions) - 1
                texts.append(local_texts[i])
                if extra_vecs is not None:
                    extra_rows.append(local_extra[i])
                actions.append(action_id)
                rewards.append(float(r_phase) if is_last else 0.0)
                dones.append(1 if is_last else 0)
                episode_ids.append(ep)
            ep += 1

        input_ids, attention_mask, input_embeds = _encode_texts(texts, tokenizer, encoder_name, max_length, encode_batch_size)
        latent_dim = int(input_embeds.shape[1])
        if extra_vecs is not None:
            input_embeds = np.concatenate([input_embeds, np.asarray(extra_rows, dtype=np.float32)], axis=1)

        actions_np = np.asarray(actions, dtype=np.int64)
        rewards_np = np.asarray(rewards, dtype=np.float32)
        dones_np = np.asarray(dones, dtype=np.int64)
        extra = {
            "input_embeds": input_embeds.astype(np.float32),
            "latent_dim": int(latent_dim),
            "conv_state_dim": int(conv_state_dim),
            "actions": actions_np,
            "rewards": rewards_np,
            "dones": dones_np,
            "returns": _compute_returns(rewards_np, dones_np, gamma=float(gamma)),
            "episode_id": np.asarray(episode_ids, dtype=np.int64),
            "label2id": micro2id,
            "num_labels": int(len(micro2id)),
        }
        return OfflineRolloutFeatures(input_ids=input_ids, attention_mask=attention_mask, labels=actions_np.copy(), extra=extra)

    feat = cache_or_build(cache_dir, key, _build, force_rebuild=force_rebuild)
    feat.extra["_cache_key"] = key.filename()
    feat.extra["_cache_dir"] = str(Path(cache_dir).resolve())
    feat.extra["_cache_meta"] = dict(meta)
    return feat


def build_cached_flat_rl_rollout_encoder(
    cache_dir: Path,
    root: DataRoot,
    tokenizer: PreTrainedTokenizerBase,
    encoder_name: str,
    max_length: int,
    context_turns: int | None,
    gamma: float,
    use_conv_state: bool = False,
    conv_state_encoding: str = "one_hot",
    extr_cfg: ExtrinsicConfig = ExtrinsicConfig(),
    encode_batch_size: int = 16,
    force_rebuild: bool = False,
) -> OfflineRolloutFeatures:
    meta = {
        "kind": "flat_rl_offline_encoder",
        "encoder_name": str(encoder_name),
        "max_length": int(max_length),
        "context_turns": context_turns,
        "gamma": float(gamma),
        "use_conv_state": int(bool(use_conv_state)),
        "extr_threshold": float(extr_cfg.threshold),
        "extr_high_threshold": float(extr_cfg.high_threshold),
        "reward_mapping": "0_below_threshold_1_below_high_2_at_high",
        "extr_reward_key": str(extr_cfg.reward_key or ""),
        "indexing": "preferred_id_turn_index",
        "conv_state_encoding": str(conv_state_encoding),
    }
    key = CacheKey("flat_rl_offline_encoder", fingerprint_dict(meta))

    def _build() -> OfflineRolloutFeatures:
        data_json = load_dataset_json(root)
        df_micro = pd.read_csv(root.annotations_dir / "annotations_micro_actions_refined.csv")
        df_micro = _with_turn_idx(df_micro)
        df_micro = df_micro.sort_values(["user_idx", "session_idx", "turn_idx"]).reset_index(drop=True)
        df_micro["row_i"] = np.arange(len(df_micro), dtype=np.int64)

        extra_vecs = None
        conv_state_dim = 0
        if use_conv_state:
            extra_vecs, _ = build_conv_state_vectors(root, df_micro, encoding=str(conv_state_encoding))
            conv_state_dim = int(extra_vecs.shape[1])

        action_space = load_action_space(root)
        class_names, label2id = build_flat_micro_label_map(action_space)
        df_extr = load_extrinsic_reward_csv(root.annotations_dir / "annotations_extrinsic_reward.csv", cfg=extr_cfg)
        extr_map: Dict[Tuple[int, int], float] = {
            (int(r["user_idx"]), int(r["session_idx"])): float(r["extrinsic_reward"])
            for r in df_extr.to_dict(orient="records")
        }

        texts: List[str] = []
        extra_rows: List[np.ndarray] = []
        actions: List[int] = []
        rewards: List[float] = []
        dones: List[int] = []
        episode_ids: List[int] = []

        ep = 0
        for (u, s), sub in tqdm(df_micro.groupby(["user_idx", "session_idx"], sort=False), desc="[cache] build flat RL offline (encoder)", unit="sess"):
            u = int(u)
            s = int(s)
            r_sess = float(extr_map[(u, s)])
            local_texts: List[str] = []
            local_extra: List[np.ndarray] = []
            local_actions: List[int] = []
            for _, row in sub.sort_values("turn_idx").iterrows():
                label = f"{str(row['selected_macro_action'])}::{str(row['selected_micro_action'])}"
                if label not in label2id:
                    continue
                local_texts.append(build_micro_action_text(data_json, row.to_dict(), context_turns=context_turns))
                if extra_vecs is not None:
                    local_extra.append(extra_vecs[int(row["row_i"])])
                local_actions.append(int(label2id[label]))

            if not local_actions:
                continue
            for i, action_id in enumerate(local_actions):
                is_last = i == len(local_actions) - 1
                texts.append(local_texts[i])
                if extra_vecs is not None:
                    extra_rows.append(local_extra[i])
                actions.append(action_id)
                rewards.append(float(r_sess) if is_last else 0.0)
                dones.append(1 if is_last else 0)
                episode_ids.append(ep)
            ep += 1

        input_ids, attention_mask, input_embeds = _encode_texts(texts, tokenizer, encoder_name, max_length, encode_batch_size)
        latent_dim = int(input_embeds.shape[1])
        if extra_vecs is not None:
            input_embeds = np.concatenate([input_embeds, np.asarray(extra_rows, dtype=np.float32)], axis=1)

        actions_np = np.asarray(actions, dtype=np.int64)
        rewards_np = np.asarray(rewards, dtype=np.float32)
        dones_np = np.asarray(dones, dtype=np.int64)
        extra = {
            "input_embeds": input_embeds.astype(np.float32),
            "latent_dim": int(latent_dim),
            "conv_state_dim": int(conv_state_dim),
            "actions": actions_np,
            "rewards": rewards_np,
            "dones": dones_np,
            "returns": _compute_returns(rewards_np, dones_np, gamma=float(gamma)),
            "episode_id": np.asarray(episode_ids, dtype=np.int64),
            "label2id": label2id,
            "num_labels": int(len(class_names)),
        }
        return OfflineRolloutFeatures(input_ids=input_ids, attention_mask=attention_mask, labels=actions_np.copy(), extra=extra)

    feat = cache_or_build(cache_dir, key, _build, force_rebuild=force_rebuild)
    feat.extra["_cache_key"] = key.filename()
    feat.extra["_cache_dir"] = str(Path(cache_dir).resolve())
    feat.extra["_cache_meta"] = dict(meta)
    return feat
