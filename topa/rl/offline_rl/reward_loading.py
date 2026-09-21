from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ExtrinsicConfig:
    threshold: float = 4.0
    high_threshold: float = 5.0
    reward_key: str | None = None


def load_intrinsic_reward_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["user_idx"] = df["user_idx"].astype(int)
    df["session_idx"] = df["session_idx"].astype(int)
    return df


def load_extrinsic_reward_csv(path: Path, cfg: ExtrinsicConfig = ExtrinsicConfig()) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["user_idx"] = df["user_idx"].astype(int)
    df["session_idx"] = df["session_idx"].astype(int)

    reward_key = "" if cfg.reward_key is None else str(cfg.reward_key).strip()
    if not reward_key:
        for candidate in ["overall_mean", "reward", "score", "value", "success"]:
            if candidate in df.columns:
                reward_key = candidate
                break

    df["reward_value"] = df[reward_key].astype(float)
    if "overall_mean" not in df.columns:
        df["overall_mean"] = df["reward_value"]
    df["extrinsic_bin"] = (df["reward_value"] >= float(cfg.threshold)).astype(int)
    # df["extrinsic_reward"] = 0.0
    # df.loc[df["reward_value"] >= float(cfg.threshold), "extrinsic_reward"] = 1.0
    # df.loc[df["reward_value"] >= float(cfg.high_threshold), "extrinsic_reward"] = 2.0
    
    # Continuous CTRS reward in [0, 1].
    # Assumes reward_value / overall_mean is on the original 0–6 CTRS scale.
    df["extrinsic_reward"] = df["reward_value"].clip(lower=0.0, upper=6.0) / 6.0
    return df[["user_idx", "session_idx", "overall_mean", "reward_value", "extrinsic_reward", "extrinsic_bin"]]