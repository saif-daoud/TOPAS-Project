import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from .prompts import build_refine_action_space_prompt
from ..utils import build_logger, parse_json, run_llm_query, build_session_index, OpenAIEmbedder, JinaLocalEmbedder
from ..annotation.utils import format_utterances

logger = build_logger()

def refine_micro_action_space(
    api_key: str,
    domain_params: dict,
    micro_actions_json_path: str,
    micro_actions_csv_path: str,
    data: list,
    out_dir: str,
    context_length: int = 5,
    conf_threshold: float = 0.65,
    max_examples_per_macro: int = 20,
    max_new_actions_per_macro: int = 6,
    propagation_delta: float = 0.87
) -> Dict[str, Any]:
    """Refine the micro action space by discovering missing micro actions (from 'None'/low-conf cases).

    - Adds new micro actions into micro_actions.json.
    - Writes a refined micro-actions annotation CSV (only updates low-conf rows returned by the LLM).
    - Writes a short report JSON.
    """
    with open(micro_actions_json_path, "r", encoding="utf-8") as f:
        action_space = json.load(f)

    df = pd.read_csv(micro_actions_csv_path, encoding="utf-8")
    cand = df[(df["selected_micro_action"].astype(str) == "None") | (df["confidence_score"] < float(conf_threshold))].copy()
    cand = cand[cand["selected_macro_action"].astype(str).str.lower() != "uncertain"]

    # logger.info(f"Low confidence utterances: {len(cand)}/{len(df)}")
    print(f"Low confidence utterances: {len(cand)}/{len(df)}")

    # Build per-session utterances for quick context retrieval
    sessions, sys_pos = build_session_index(data, domain_params)

    usage = {"inp_t": 0.0, "out_t": 0.0}
    report = {
        "added_micro_actions": [],
        "updated_rows": 0,
        "propagated_rows": 0,
        "usage": usage
    }

    macro_by_name = {m["name"]: m for m in action_space}
    # api_key_emb = os.environ["OPENAI_KEY"]
    # embedder = OpenAIEmbedder(api_key_emb, model="text-embedding-3-small")
    embedder = JinaLocalEmbedder()

    for macro_name, g in tqdm(cand.groupby("selected_macro_action", sort=False), total=cand["selected_macro_action"].nunique()):
        macro = macro_by_name[macro_name]

        # Collect examples (row_key + context)
        examples = []
        for idx, r in g.head(max_examples_per_macro).iterrows():
            utts = sessions[(r.user_idx, r.session_idx)]
            p = sys_pos[(r.user_idx, r.session_idx)][r.utterance_id]
            last_ctx = utts[max(0, p-context_length): p]
            ex = {
                "row_key": f"u{r.user_idx}_s{r.session_idx}_{r.utterance_id}",
                "utterance_id": r.utterance_id,
                "last_utterances": format_utterances(last_ctx),
                "utterance": utts[p]["utterance"],
                "current_label": r.selected_micro_action,
                "current_confidence": r.confidence_score
            }
            examples.append(ex)

        if len(examples) == 0:
            continue

        prompt = build_refine_action_space_prompt(domain_params, macro, examples, max_new_actions=max_new_actions_per_macro)
        raw, inp_t, out_t = run_llm_query(prompt, api_key)
        usage["inp_t"] += inp_t / 1_000_000
        usage["out_t"] += out_t / 1_000_000

        try:
            parsed = parse_json(raw)
        except Exception:
            try:
                parsed = parse_json(raw.replace("'", '"'))
            except Exception:
                continue

        new_actions = parsed["new_micro_actions"]
        mapping = parsed["utterance_to_action"]

        # Add new micro actions
        existing_names = {m["name"] for m in macro["micro_actions"]}
        added_here = []
        for a in new_actions:
            name = a["name"]
            if name in existing_names:
                continue
            micro_obj = {
                "name": name,
                "description": a["description"],
                "states": a["states"],
                "confidence_score": a["confidence_score"]
            }
            macro["micro_actions"].append(micro_obj)
            existing_names.add(name)
            added_here.append(
                {
                    "name": name,
                    "description": micro_obj["description"],
                    "states": micro_obj["states"],
                    "reason": a["reason"],
                    "confidence_score": a["confidence_score"]
                }
            )

        if len(added_here) > 0:
            report["added_micro_actions"].append({"macro_action": macro_name, "new_micro_actions": added_here})

        # Update rows in the micro-actions CSV
        new_action_names = {a["name"] for a in added_here}
        rowkey_to_label = {}
        for m in mapping:
            row_key = m["row_key"]
            selected = m["selected_micro_action"]
            conf = m["confidence_score"]
            reason = m["reason"]

            if selected not in new_action_names.union(existing_names):
                continue

            # Decode row key: uX_sY_system_Z
            try:
                # split on '_': u{u}_s{s}_{utt_id}
                parts = row_key.split("_", 2)
                u = int(parts[0][1:])
                s = int(parts[1][1:])
                utt_id = parts[2]
            except Exception:
                logger.info(f"[Warning] Can't decode raw_key {row_key}")
                continue

            mask = (df["user_idx"] == u) & (df["session_idx"] == s) & (df["utterance_id"] == utt_id)
            if mask.sum() == 0:
                logger.info(f"[Warning] Can't find {u} - {s} - {utt_id}")
                continue
            if mask.sum() > 1:
                logger.info(f"[Warning] Many rows for {u} - {s} - {utt_id}")

            df.loc[mask, "selected_micro_action"] = selected
            df.loc[mask, "confidence_score"] = conf
            df.loc[mask, "refinement_reason"] = reason
            report["updated_rows"] += int(mask.sum())

            rowkey_to_label[row_key] = selected

        # Propagate ONLY newly added micro actions to other low-conf / None utterances
        # Seeds = examples that were assigned to a newly added action.
        if len(new_action_names) > 0:
            # All candidate rows for this macro (same group g)
            group_mask = df["selected_macro_action"] == macro_name
            group_rows = df[group_mask].copy()
            group_cand_mask = (group_rows["selected_micro_action"] == "None") | (group_rows["confidence_score"] < conf_threshold)
            group_cand = group_rows[group_cand_mask].copy()

            # Build embeddings for candidate texts (we use short context for better semantics)
            cand_texts = []
            cand_index = []
            for ridx, r in group_cand.iterrows():
                utts = sessions[(r.user_idx, r.session_idx)]
                p = sys_pos[(r.user_idx, r.session_idx)][r.utterance_id]
                ctx = utts[max(0, p-context_length): p+1]
                cand_texts.append(format_utterances(ctx))
                cand_index.append(ridx)

            if len(cand_texts) == 0:
                continue

            # Seeds per new action = the subset of mapped examples that the LLM assigned to that new action.
            # We embed short context (same formatting as candidates) to improve semantic matching.
            seeds_by_action = {a: [] for a in new_action_names}
            for rk, lab in rowkey_to_label.items():
                if lab not in new_action_names:
                    continue
                try:
                    parts = rk.split("_", 2)
                    u = int(parts[0][1:])
                    s = int(parts[1][1:])
                    utt_id = parts[2]
                except Exception:
                    logger.info(f"[Warning] Can't decode raw_key {rk}")
                    continue

                utts = sessions[(u, s)]
                p = sys_pos[(u, s)][utt_id]
                ctx = utts[max(0, p-context_length) : p+1]
                seeds_by_action[lab].append(format_utterances(ctx))

            # If LLM didn't label any seed example with the new action, skip propagation for it.
            seeds_by_action = {k: v for k, v in seeds_by_action.items() if len(v) > 0}
            if len(seeds_by_action) == 0:
                continue

            X_cand = embedder.embed_many(cand_texts)
            X_cand = X_cand / (np.linalg.norm(X_cand, axis=1, keepdims=True) + 1e-12)

            for act_name, seed_texts in seeds_by_action.items():
                X_seed = embedder.embed_many(seed_texts)
                X_seed = X_seed / (np.linalg.norm(X_seed, axis=1, keepdims=True) + 1e-12)

                # For each candidate, take the max sim over seeds
                sim = X_seed @ X_cand.T
                best = sim.max(axis=0)
                hits = (best >= propagation_delta).nonzero()[0].tolist()

                for j in hits:
                    idx = cand_index[j]
                    df.at[idx, "selected_micro_action"] = act_name
                    df.at[idx, "confidence_score"] = float(best[j])
                    df.at[idx, "refinement_reason"] = "propagated_new_micro_action"
                    report["propagated_rows"] += 1

    # Save updated action space (overwrite) + refined annotations
    backup_path = micro_actions_json_path + ".bak"
    if not Path(backup_path).exists():
        Path(backup_path).write_text(Path(micro_actions_json_path).read_text(encoding="utf-8"), encoding="utf-8")

    micro_actions_json_path = str(micro_actions_json_path)[:-5] + "_refined.json"
    with open(micro_actions_json_path, "w", encoding="utf-8") as f:
        json.dump(action_space, f, ensure_ascii=False, indent=2)

    refined_csv_path = str(Path(out_dir) / "annotations_micro_actions_refined.csv")
    df.to_csv(refined_csv_path, index=False, encoding="utf-8")

    report_path = str(Path(out_dir) / "micro_action_refinement_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"[TOPA] Micro-action refinement done. Updated rows: {report['updated_rows']}")
    return {"refined_csv": refined_csv_path, "report_json": report_path, "micro_actions_json": micro_actions_json_path}