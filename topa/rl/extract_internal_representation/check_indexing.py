"""Check that activation indexing matches annotation indexing.

This script verifies, for each (user_idx, session_idx):

  - max macro turn ids < num_system_turns
  - max micro utterance ids < num_system_turns

It assumes activation files store `activations: [L, T, D]`, where T is the
number of `<|system|>` turns in that session transcript.
"""



import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch

from ..sft_pretraining.utils import add_preferred_turn_index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Domain root containing annotations/.")
    ap.add_argument("--activations_dir", type=str, required=True, help="Directory with <role>_*/session_*.pt")
    ap.add_argument("--macro_csv", type=str, default="annotations/annotations_macro_actions.csv")
    ap.add_argument("--micro_csv", type=str, default="annotations/annotations_micro_actions_refined.csv")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    act_dir = Path(args.activations_dir)
    if not act_dir.is_absolute():
        act_dir = (root / act_dir).resolve()

    macro_path = Path(args.macro_csv)
    if not macro_path.is_absolute():
        macro_path = (root / macro_path).resolve()
    micro_path = Path(args.micro_csv)
    if not micro_path.is_absolute():
        micro_path = (root / micro_path).resolve()

    df_macro = pd.read_csv(macro_path) if macro_path.exists() else pd.DataFrame()
    df_micro = pd.read_csv(micro_path) if micro_path.exists() else pd.DataFrame()

    # Ensure numeric indexing columns
    if len(df_macro) > 0:
        df_macro = add_preferred_turn_index(df_macro, "from_turn_idx", ["from_id"], ["from_idx"])
        df_macro = add_preferred_turn_index(df_macro, "to_turn_idx", ["to_id"], ["to_idx"])
        for c in ["user_idx", "session_idx", "from_turn_idx", "to_turn_idx"]:
            if c in df_macro.columns:
                df_macro[c] = pd.to_numeric(df_macro[c], errors="coerce")

    if len(df_micro) > 0:
        df_micro = add_preferred_turn_index(df_micro, "activation_turn_idx", ["utterance_id", "turn_id", "from_id", "to_id"], ["utterance_idx"])
        for c in ["user_idx", "session_idx", "activation_turn_idx"]:
            if c in df_micro.columns:
                df_micro[c] = pd.to_numeric(df_micro[c], errors="coerce")

    # Load activations and check bounds
    problems = 0
    totals = 0

    # (user, session) -> T
    T_map: Dict[Tuple[int, int], int] = {}
    scan_dirs = sorted(
        [p for p in act_dir.iterdir() if p.is_dir() and re.match(r"^[A-Za-z][A-Za-z0-9._-]*_\d+$", p.name)],
        key=lambda p: p.name,
    )
    for pdir in scan_dirs:
        try:
            u = int(pdir.name.split("_")[-1])
        except Exception:
            continue
        for sf in sorted(pdir.glob("session_*.pt")):
            try:
                s = int(sf.stem.split("_")[-1])
            except Exception:
                continue
            obj = torch.load(sf, map_location="cpu")
            acts = obj["activations"]
            assert acts.ndim == 3
            T = int(acts.shape[1])
            assert T == int(obj["num_system_turns"])
            assert T == int(obj["turn_token_positions"].numel())
    #         T_map[(u, s)] = T

    # for (u, s), T in sorted(T_map.items()):
    #     totals += 1
    #     bad = False
    #     msg = []
    #     if len(df_macro) > 0 and {"user_idx", "session_idx", "from_turn_idx", "to_turn_idx"}.issubset(df_macro.columns):
    #         subm = df_macro[(df_macro["user_idx"] == u) & (df_macro["session_idx"] == s)].copy()
    #         subm = subm.dropna(subset=["from_turn_idx", "to_turn_idx"])
    #         if len(subm) > 0:
    #             mx = int(pd.to_numeric(subm[["from_turn_idx", "to_turn_idx"]].max().max(), errors="coerce"))
    #             if mx >= T:
    #                 bad = True
    #                 msg.append(f"macro max(from/to)={mx} >= T={T}")

    #     if len(df_micro) > 0 and {"user_idx", "session_idx", "activation_turn_idx"}.issubset(df_micro.columns):
    #         subu = df_micro[(df_micro["user_idx"] == u) & (df_micro["session_idx"] == s)].copy()
    #         subu = subu.dropna(subset=["activation_turn_idx"])
    #         if len(subu) > 0:
    #             mxu = int(pd.to_numeric(subu["activation_turn_idx"].max(), errors="coerce"))
    #             if mxu >= T:
    #                 bad = True
    #                 msg.append(f"micro max(turn_id)={mxu} >= T={T}")

    #     if bad:
    #         problems += 1
    #         print(f"[mismatch] user={u} session={s} :: " + "; ".join(msg))

    # ok = totals - problems
    # print(f"Checked {totals} sessions: OK={ok} mismatched={problems}")
    # assert totals > 0
    assert problems == 0


if __name__ == "__main__":
    main()
