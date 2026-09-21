import json
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ..annotation.utils import format_utterances
from ..utils import OpenAIEmbedder, JinaLocalEmbedder, build_session_index

def build_expert_review_set(
    micro_actions_csv_path: str,
    data: list,
    domain_params: dict,
    out_dir: str,
    conf_threshold: float = 0.65,
    n_review: int = 200,
    context_length: int = 10,
    random_state: int = 42,
) -> str:
    """Create a diverse set of low-confidence utterances for a domain expert.

    - Filter low-confidence system utterances.
    - Cluster them using OpenAI embeddings + KMeans (diversity).
    - Select a diverse subset (1 per cluster, then fill the rest by lowest confidence).

    The expert is expected to fill the `expert_label` column.
    """
    out_dir = str(Path(out_dir))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(micro_actions_csv_path)
    mask = (df["selected_micro_action"] == "None") | (df["confidence_score"] < conf_threshold)
    low = df[mask].copy()
    print(f"Low confidence utterances: {len(low)}/{len(df)}")
    if len(low) == 0:
        return ""

    sessions, pos = build_session_index(data, domain_params)

    def _get_user_and_session_meta(user_idx: int, session_idx: int):
        """Extract user id/name, session summary, and presenting conditions."""
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

    # Attach utterance text + short context for expert readability + session metadata.
    texts = []
    contexts = []
    session_summaries = []
    presenting_conditions_list = []

    for r in low.itertuples(index=False):
        utts = sessions[(r.user_idx, r.session_idx)]
        p = pos[(r.user_idx, r.session_idx)][r.utterance_id]

        texts.append(utts[p]["utterance"])
        ctx = utts[max(0, p - context_length): p]
        contexts.append(format_utterances(ctx))

        summary, conds = _get_user_and_session_meta(r.user_idx, r.session_idx)
        session_summaries.append(summary)
        presenting_conditions_list.append(conds)

    low["utterance_text"] = texts
    low["context"] = contexts
    low["session_summary"] = session_summaries
    low["presenting_conditions"] = presenting_conditions_list

    low["word_count"] = low["utterance_text"].fillna("").astype(str).str.split().str.len()
    low = low[low["word_count"] >= 5].copy()
    print(f"Low confidence utterances (post-length-filter): {len(low)}/{len(df)}")
    if len(low) == 0:
        return ""

    # Clustering (diversity)
    # embedder = OpenAIEmbedder(model="text-embedding-3-small")
    embedder = JinaLocalEmbedder()
    X = embedder.embed_many(low["utterance_text"].tolist())
    k = max(1, min(n_review, X.shape[0]))
    if k == 1:
        low["cluster_id"] = 0
    else:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        low["cluster_id"] = km.fit_predict(X)

    # Representative per cluster = lowest confidence (most uncertain)
    review = (
        low.sort_values("confidence_score", ascending=True)
        .groupby("cluster_id", as_index=False)
        .head(1)
    )

    review["expert_label"] = "" # to be filled by expert
    review["expert_notes"] = ""

    out_path = str(Path(out_dir) / "expert_review_micro_actions.csv")
    cols = [
        "user_idx",
        "session_idx",
        "utterance_id",
        "context",
        "selected_macro_action",
        "selected_micro_action",
        "confidence_score",
        "session_summary",
        "presenting_conditions",
        "cluster_id",
        "expert_label",
        "expert_notes",
    ]
    # keep only existing columns
    cols = [c for c in cols if c in review.columns]
    review[cols].to_csv(out_path, index=False, encoding="utf-8")
    return out_path

def propagate_expert_labels(
    expert_review_csv_path: str,
    micro_actions_csv_path: str,
    data: list,
    domain_params: dict,
    out_dir: str,
    conf_threshold: float = 0.65,
    similarity_delta: float = 0.85,
    context_length: int = 5
) -> Tuple[str, str]:
    """Propagate expert-provided labels to similar low-confidence utterances.

    Similarity uses OpenAI embeddings cosine similarity (on utterance text or context).
    Only updates rows with confidence < conf_threshold.

    Returns:
      (updated_csv_path, report_json_path)
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(micro_actions_csv_path)
    low_mask = (df["selected_micro_action"] == "None") | (df["confidence_score"] < conf_threshold)

    exp = pd.read_csv(expert_review_csv_path)
    exp = exp[exp["expert_label"].notna()].copy()
    exp["expert_label"] = exp["expert_label"].astype(str).str.strip()
    exp = exp[exp["expert_label"] != ""]
    if len(exp) == 0:
        return "", ""

    sessions, pos = build_session_index(data, domain_params)

    # Build text arrays for low-confidence utterances and expert-labeled utterances.
    low_df = df[low_mask].copy()
    low_texts = []
    for r in low_df.itertuples(index=False):
        utts = sessions[(r.user_idx, r.session_idx)]
        p = pos[(r.user_idx, r.session_idx)][r.utterance_id]
        if context_length > 0:
            ctx = utts[max(0, p-context_length): p+1]
            low_texts.append(format_utterances(ctx))
        else:
            low_texts.append(utts[p]["utterance"])

    exp_texts = []
    for r in exp.itertuples(index=False):
        utts = sessions[(r.user_idx, r.session_idx)]
        p = pos[(r.user_idx, r.session_idx)][r.utterance_id]
        if context_length > 0:
            ctx = utts[max(0, p - context_length) : p + 1]
            exp_texts.append(format_utterances(ctx))
        else:
            exp_texts.append(utts[p]["utterance"])

    # embedder = OpenAIEmbedder(model="text-embedding-3-small")
    embedder = JinaLocalEmbedder()
    X_low = embedder.embed_many(low_texts)
    X_exp = embedder.embed_many(exp_texts)

    # cosine sim = (A/||A||) @ (B/||B||)^T
    X_low = X_low / (np.linalg.norm(X_low, axis=1, keepdims=True) + 1e-12)
    X_exp = X_exp / (np.linalg.norm(X_exp, axis=1, keepdims=True) + 1e-12)
    sim = X_exp @ X_low.T

    updated = 0
    updated_by_label = {}

    low_indices = low_df.index.tolist()
    for i, r in enumerate(exp.itertuples(index=False)):
        label = str(r.expert_label).strip()
        hits = (sim[i] >= similarity_delta).nonzero()[0].tolist()
        for j in hits:
            idx = low_indices[j]
            df.at[idx, "selected_micro_action"] = label
            df.at[idx, "confidence_score"] = float(sim[i][j])
            updated += 1
            updated_by_label[label] = updated_by_label.get(label, 0) + 1

    out_csv = str(Path(out_dir) / "annotations_micro_actions_propagated.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8")

    report = {
        "expert_review_csv_path": expert_review_csv_path,
        "micro_actions_csv_path": micro_actions_csv_path,
        "conf_threshold": conf_threshold,
        "similarity_delta": similarity_delta,
        "num_low_conf": int(low_mask.sum()),
        "num_expert_labeled": len(exp),
        "num_updated": updated,
        "updated_by_label": updated_by_label
    }

    out_report = str(Path(out_dir) / "label_propagation_report.json")
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return out_csv, out_report