import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd

IGNORE_INDEX_COLS = ["user_idx", "session_idx", "phase_idx", "from_id", "to_id", "from_idx", "to_idx", "selected_macro_action"]
MERGE_KEYS = ["user_idx", "session_idx", "from_id", "to_id", "from_idx", "to_idx"]
NA_SENT = "__NA__"

def norm_cell(x):
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"none", "nan"}:
        return None
    return s

def normalize_series(s: pd.Series) -> pd.Series:
    ss = s.astype("string").str.strip()
    ss = ss.replace({"": pd.NA})
    ss = ss.mask(ss.str.lower().isin(["none", "nan"]), pd.NA)
    return ss

def build_allowed_maps(conv_states):
    """
    Returns:
      allowed_exact[var] = set of allowed values (trimmed, excluding none-like)
      allowed_lower_to_canon[var] = {lower_value: canonical_value} for case-insensitive matching
    """
    allowed_exact = {}
    allowed_lower_to_canon = {}

    for v in conv_states:
        var = v.get("Variable Name")
        cats = v.get("Categorical values") or []

        cleaned = [norm_cell(c) for c in cats]
        cleaned = [c for c in cleaned if c is not None]  # exclude none-like from allowed set

        allowed_exact[var] = set(cleaned)

        # case-insensitive map (only safe if no collisions)
        lower_map = {}
        collisions = set()
        for c in cleaned:
            key = c.lower()
            if key in lower_map and lower_map[key] != c:
                collisions.add(key)
            else:
                lower_map[key] = c
        for k in collisions:
            lower_map.pop(k, None)

        allowed_lower_to_canon[var] = lower_map

    return allowed_exact, allowed_lower_to_canon

def update_conversation_states_categories(conversation_states_json: str, ann_csv_1: str, ann_csv_2: str, out_csv_1: str, 
                                          out_csv_2: str, out_report_csv: str, none_value: str = "none"):
    """
    Behavior:
    1) Do NOT add missing categories to conversation_states_json.
    2) Coerce any non-empty value not in allowed categories -> `none_value` (in BOTH files).
    3) If one annotator has a valid (non-none) label and the other has None/none, copy the valid label over the none.
       (This happens after coercion, so we only propagate allowed labels.)
    4) Writes cleaned copies of both annotation CSVs + a report.
    """
    with open(conversation_states_json, "r", encoding="utf-8") as f:
        conv_states = json.load(f)

    a1 = pd.read_csv(ann_csv_1)
    a2 = pd.read_csv(ann_csv_2)

    allowed_exact, allowed_lower_to_canon = build_allowed_maps(conv_states)

    # Only variables that exist in the annotations
    vars_in_ann = [c for c in a1.columns if c in allowed_exact and c not in IGNORE_INDEX_COLS]

    report = []

    def coerce_df(df: pd.DataFrame, which: str) -> pd.DataFrame:
        nonlocal report
        df = df.copy()

        for var in vars_in_ann:
            if var not in df.columns:
                continue

            col = df[var].map(norm_cell)  # None or trimmed string
            if col.isna().all():
                continue

            allowed = allowed_exact.get(var, set())
            lower_map = allowed_lower_to_canon.get(var, {})

            # case-insensitive canonicalization where possible
            canon = col.map(lambda x: lower_map.get(x.lower(), x) if x is not None else None)

            # unknown (non-empty) -> none_value
            unknown_mask = canon.notna() & (~canon.isin(list(allowed)))
            if unknown_mask.any():
                unknown_vals = canon[unknown_mask].value_counts().to_dict()
                report.append({
                    "file": which,
                    "Variable Name": var,
                    "n_overwritten_unknown": int(unknown_mask.sum()),
                    "unknown_values_counts_json": json.dumps(unknown_vals, ensure_ascii=False),
                })

            df.loc[unknown_mask, var] = none_value

            # also write back canonical (trimmed + canonical case) for known values
            known_mask = canon.notna() & (~unknown_mask)
            df.loc[known_mask, var] = canon[known_mask]

        return df

    # 1) First, coerce each file independently (unknown -> none, canonicalize known)
    a1_clean = coerce_df(a1, "ann1")
    a2_clean = coerce_df(a2, "ann2")

    # 2) Then, propagate non-none labels across annotators when the other is none
    #    (row-wise, column-wise; only for vars_in_ann)
    merged_keys = [c for c in ["user_idx", "session_idx", "from_id", "to_id", "from_idx", "to_idx"] if c in a1_clean.columns and c in a2_clean.columns]
    if not merged_keys:
        # fallback: align by row index if keys missing (not ideal, but prevents crash)
        a1_aligned = a1_clean.copy()
        a2_aligned = a2_clean.copy()
        align_mode = "index"
    else:
        a1_aligned = a1_clean.set_index(merged_keys, drop=False)
        a2_aligned = a2_clean.set_index(merged_keys, drop=False)
        # keep only intersecting rows to avoid creating new rows
        common_idx = a1_aligned.index.intersection(a2_aligned.index)
        a1_aligned = a1_aligned.loc[common_idx].copy()
        a2_aligned = a2_aligned.loc[common_idx].copy()
        align_mode = "keys"

    def is_none_series(s: pd.Series) -> pd.Series:
        ss = s.astype("string").str.strip().str.lower()
        return s.isna() | ss.eq("") | ss.eq("none") | ss.eq(none_value.lower())

    propagated_total = 0

    for var in vars_in_ann:
        if var not in a1_aligned.columns or var not in a2_aligned.columns:
            continue

        a1_none = is_none_series(a1_aligned[var])
        a2_none = is_none_series(a2_aligned[var])

        # propagate A2 -> A1
        mask_21 = a1_none & (~a2_none)
        # propagate A1 -> A2
        mask_12 = a2_none & (~a1_none)

        n_prop = int(mask_21.sum() + mask_12.sum())
        if n_prop > 0:
            report.append({
                "file": "both",
                "Variable Name": var,
                "n_propagated_none_to_label": n_prop,
                "alignment": align_mode,
                "note": f"Copied non-{none_value} label to the side that was None/{none_value}.",
            })
            propagated_total += n_prop

        a1_aligned.loc[mask_21, var] = a2_aligned.loc[mask_21, var]
        a2_aligned.loc[mask_12, var] = a1_aligned.loc[mask_12, var]

    # write back to full frames
    if align_mode == "keys":
        # update only common rows; keep original non-common rows as-is
        a1_out = a1_clean.set_index(merged_keys, drop=False)
        a2_out = a2_clean.set_index(merged_keys, drop=False)
        a1_out.update(a1_aligned)
        a2_out.update(a2_aligned)
        a1_out = a1_out.reset_index(drop=True)
        a2_out = a2_out.reset_index(drop=True)
    else:
        a1_out = a1_aligned
        a2_out = a2_aligned

    # Save
    Path(out_csv_1).parent.mkdir(parents=True, exist_ok=True)
    a1_out.to_csv(out_csv_1, index=False, encoding="utf-8")
    a2_out.to_csv(out_csv_2, index=False, encoding="utf-8")
    pd.DataFrame(report).to_csv(out_report_csv, index=False, encoding="utf-8")

    return out_csv_1, out_csv_2, out_report_csv

def process_dialogue(dialogue: List[dict], domain_params: dict) -> List[dict]:
    system = domain_params["system"]
    user = domain_params["user"]
    utterances = []
    sys_c = 0
    usr_c = 0
    for turn in dialogue:
        sp = turn.get("speaker")
        txt = turn.get("text", "")
        if sp == "system":
            utt_id = f"{system}_{sys_c}"
            sys_c += 1
            speaker = system
        elif sp == "user":
            utt_id = f"{user}_{usr_c}"
            usr_c += 1
            speaker = user
        else:
            utt_id = f"{sp}_{len(utterances)}"
            speaker = str(sp)
        utterances.append({"utterance_id": utt_id, "speaker": speaker, "utterance": txt})
    return utterances

def build_session_index(data: list, domain_params: dict):
    sessions = {}
    pos = {}
    for user_idx, user in enumerate(data):
        for session_idx, session in enumerate(user.get("sessions", [])):
            utts = process_dialogue(session.get("dialogue", []), domain_params)
            sessions[(user_idx, session_idx)] = utts
            pos[(user_idx, session_idx)] = {u["utterance_id"]: i for i, u in enumerate(utts)}
    return sessions, pos

def format_utterances(utterances: List[Dict[str, Any]]) -> str:
    return "\n".join([f'{u["speaker"]} ({u["utterance_id"]}): {u["utterance"]}' for u in utterances])

def _get_user_and_session_meta(data: list, user_idx: int, session_idx: int) -> Tuple[str, str]:
    user_obj = data[user_idx] if 0 <= user_idx < len(data) else {}
    sess_list = user_obj.get("sessions") or []
    sess_obj = sess_list[session_idx] if 0 <= session_idx < len(sess_list) else {}
    sm = (sess_obj.get("session_metadata") or {})

    summary = sm.get("summary", "") or ""

    conds = sm.get("presenting conditions", None)
    if conds is None:
        conds = sm.get("presenting_conditions", "")

    if isinstance(conds, list):
        presenting_conditions = "; ".join(str(x) for x in conds)
    else:
        presenting_conditions = str(conds) if conds is not None else ""
    return summary, presenting_conditions

def build_uncertain_review_set_conv_state(
    ann_csv_1: str,
    ann_csv_2: str,
    data: list,
    domain_params: dict,
    out_csv: str,
    n_review: int = 200,
    context_length: int = 10,
):
    import json
    import pandas as pd

    a1 = pd.read_csv(ann_csv_1)
    a2 = pd.read_csv(ann_csv_2)

    dim_cols = [c for c in a1.columns if c not in IGNORE_INDEX_COLS]
    MERGE_KEYS = ["user_idx", "session_idx", "from_id", "to_id"]

    # avoid many-to-many merges if duplicates exist
    if a1.duplicated(MERGE_KEYS).any() and "phase_idx" in a1.columns:
        a1 = a1.sort_values("phase_idx", kind="stable").drop_duplicates(MERGE_KEYS, keep="first")
    if a2.duplicated(MERGE_KEYS).any() and "phase_idx" in a2.columns:
        a2 = a2.sort_values("phase_idx", kind="stable").drop_duplicates(MERGE_KEYS, keep="first")

    merged = a1.merge(a2, on=MERGE_KEYS, how="inner", suffixes=("_a1", "_a2"))

    # compute disagreements
    diff_map = {}
    for col in dim_cols:
        s1 = normalize_series(merged[f"{col}_a1"])
        s2 = normalize_series(merged[f"{col}_a2"])
        eq = s1.fillna(NA_SENT).eq(s2.fillna(NA_SENT))
        diff_map[col] = ~eq

    diff_df = pd.DataFrame(diff_map, index=merged.index)
    merged["diff_count"] = diff_df.sum(axis=1).astype(int)

    # build session index
    sessions, pos = build_session_index(data, domain_params)

    # --- select EXACTLY n_review rows, *considering* sequence_len>=3 ---
    candidates = merged[merged["diff_count"] > 0].sort_values("diff_count", ascending=False, kind="stable").copy()

    selected_idx = []
    from_idxs = []
    to_idxs = []
    seq_lens = []

    for r in candidates.itertuples(index=True):
        key = (int(r.user_idx), int(r.session_idx))
        pos_map = pos.get(key, {})

        p_from = pos_map.get(getattr(r, "from_id", None))
        p_to = pos_map.get(getattr(r, "to_id", None))

        seq_len = (p_to - p_from + 1)

        if seq_len >= 3:
            selected_idx.append(r.Index)
            from_idxs.append(p_from)
            to_idxs.append(p_to)
            seq_lens.append(seq_len)

            if len(selected_idx) >= n_review:
                break

    if len(selected_idx) < n_review:
        raise ValueError(
            f"Only {len(selected_idx)} rows satisfy diff_count>0 AND sequence_len>=3; "
            f"cannot return exactly n_review={n_review}."
        )

    review = candidates.loc[selected_idx].copy()
    review["from_idx"] = pd.array(from_idxs, dtype="Int64")
    review["to_idx"] = pd.array(to_idxs, dtype="Int64")
    review["sequence_len"] = seq_lens

    # disagreements (on final set)
    diff_sub = diff_df.loc[review.index]
    disagree_lists = diff_sub.apply(lambda row: [c for c, v in row.items() if v], axis=1)

    def disagreements_json(idx):
        cols = disagree_lists.loc[idx]
        items = []
        for c in cols:
            items.append(
                {
                    "dim": c,
                    "a1": norm_cell(review.loc[idx, f"{c}_a1"]),
                    "a2": norm_cell(review.loc[idx, f"{c}_a2"]),
                }
            )
        return json.dumps(items, ensure_ascii=False)

    review["disagree_dims"] = disagree_lists.apply(lambda xs: "; ".join(xs))
    review["disagreements_json"] = [disagreements_json(i) for i in review.index]

    # build context + sequence + metadata
    contexts = []
    sequences = []
    session_summaries = []
    presenting_conditions_list = []

    for r in review.itertuples(index=False):
        key = (int(r.user_idx), int(r.session_idx))
        utts = sessions.get(key)

        p_from = int(r.from_idx)
        p_to = int(r.to_idx)

        ctx_start = max(0, p_from - context_length)
        ctx_block = utts[ctx_start:p_from]
        contexts.append(format_utterances(ctx_block))
        seq_block = utts[p_from : p_to + 1]
        sequences.append(format_utterances(seq_block))

        summary, conds = _get_user_and_session_meta(data, int(r.user_idx), int(r.session_idx))
        session_summaries.append(summary)
        presenting_conditions_list.append(conds)

    review["context"] = contexts
    review["sequence"] = sequences
    review["session_summary"] = session_summaries
    review["presenting_conditions"] = presenting_conditions_list

    out_cols = [
        "user_idx", "session_idx",
        "from_id", "to_id",
        "from_idx", "to_idx",
        "diff_count", "disagree_dims", "disagreements_json",
        "context",
        "sequence",
        "session_summary", "presenting_conditions",
    ]

    review_out = review[out_cols].copy()
    review_out.to_csv(out_csv, index=False, encoding="utf-8")
    return out_csv