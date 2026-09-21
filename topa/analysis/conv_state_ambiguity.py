"""Conversation-state ambiguity mining.

Workflow:
  1) normalize conv-state categories to the canonical list.
  2) Compare two annotation runs and extract disagreements.
  3) Export a compact expert review CSV with context.
"""

from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from .generate_review_set import update_conversation_states_categories, build_session_index, format_utterances

def run_conv_state_ambiguity_review(
    conversation_states_json: str,
    ann_csv_1: str,
    ann_csv_2: str,
    data: List[Dict[str, Any]],
    domain_params: Dict[str, Any],
    out_dir: str,
    none_value: str = "none",
    n_review: int = 200,
    context_length: int = 10,
) -> str:
    """Generate an expert review set from disagreements between 2 conv_state runs.

    Returns the path of the generated CSV.
    """
    out_dir = str(Path(out_dir))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: category normalization (optional but recommended)
    a1_u = str(Path(out_dir) / "updated_annotations_conv_state.csv")
    a2_u = str(Path(out_dir) / "updated_annotations_conv_state_v2.csv")
    report_path = str(Path(out_dir) / "conv_state_categories_update_report.json")
    try:
        update_conversation_states_categories(
            conversation_states_json=conversation_states_json,
            ann_csv_1=ann_csv_1,
            ann_csv_2=ann_csv_2,
            out_csv_1=a1_u,
            out_csv_2=a2_u,
            out_report_path=report_path,
            none_value=none_value,
        )
        ann1 = a1_u
        ann2 = a2_u
    except Exception:
        # fall back to raw annotations
        ann1 = ann_csv_1
        ann2 = ann_csv_2

    df1 = pd.read_csv(ann1)
    df2 = pd.read_csv(ann2)

    required_cols = {"user_idx", "session_idx", "phase_idx"}
    if not required_cols.issubset(df1.columns) or not required_cols.issubset(df2.columns):
        raise ValueError("conv_state CSVs must contain user_idx/session_idx/phase_idx")

    # Identify conv-state label columns. We exclude obvious meta columns.
    idx_cols = ["user_idx", "session_idx", "phase_idx"]
    meta_cols = {"from_id", "to_id", "from_idx", "to_idx", "selected_macro_action"}
    label_cols = [c for c in df1.columns if c not in idx_cols and c not in meta_cols]
    if not label_cols:
        raise ValueError("No conv-state label columns found in annotations")

    # Merge on indices and compare label columns
    merged = df1.merge(df2, on=idx_cols, suffixes=("_run1", "_run2"))

    diffs = []
    for col in label_cols:
        c1 = f"{col}_run1"
        c2 = f"{col}_run2"
        if c1 not in merged.columns or c2 not in merged.columns:
            continue
        # handle NaNs consistently
        neq = merged[c1].astype(str) != merged[c2].astype(str)
        if neq.any():
            sub = merged.loc[neq, idx_cols + [c1, c2]].copy()
            sub.rename(columns={c1: f"{col}_run1", c2: f"{col}_run2"}, inplace=True)
            diffs.append(sub)

    if not diffs:
        out_path = str(Path(out_dir) / "expert_review_conversation_states.csv")
        pd.DataFrame(columns=idx_cols + ["context"]).to_csv(out_path, index=False)
        return out_path

    # Combine differences across columns (outer join on indices)
    df_diff = diffs[0]
    for d in diffs[1:]:
        df_diff = df_diff.merge(d, on=idx_cols, how="outer")

    available = len(df_diff)
    if available == 0:
        out_path = str(Path(out_dir) / "expert_review_conversation_states.csv")
        pd.DataFrame(columns=idx_cols + ["context"]).to_csv(out_path, index=False)
        return out_path

    n_review_eff = min(int(n_review), available)
    df_diff = df_diff.sample(n=n_review_eff, random_state=42).reset_index(drop=True)

    # Attach meta columns (stable across runs) for expert context
    meta_available = [c for c in meta_cols if c in df1.columns]
    if meta_available:
        df_meta = df1[idx_cols + meta_available].drop_duplicates(subset=idx_cols)
        df_diff = df_diff.merge(df_meta, on=idx_cols, how="left")

    # Add context
    sessions, pos = build_session_index(data, domain_params)
    contexts = []
    for r in df_diff.itertuples(index=False):
        utts = sessions[(r.user_idx, r.session_idx)]
        # Prefer from_id (macro-phase anchor) if available; fallback to system_{phase_idx}
        anchor = getattr(r, "from_id", None)
        if isinstance(anchor, str) and anchor in pos[(r.user_idx, r.session_idx)]:
            p = pos[(r.user_idx, r.session_idx)][anchor]
        else:
            p = pos[(r.user_idx, r.session_idx)].get(f"system_{r.phase_idx}", 0)
        ctx = utts[max(0, p - context_length) : p]
        contexts.append(format_utterances(ctx))

    df_diff.insert(0, "context", contexts)
    df_diff["expert_label"] = ""
    df_diff["expert_notes"] = ""

    out_path = str(Path(out_dir) / "expert_review_conversation_states.csv")
    df_diff.to_csv(out_path, index=False)
    return out_path