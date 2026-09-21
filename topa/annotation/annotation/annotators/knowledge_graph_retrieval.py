from copy import deepcopy
from typing import Any, Dict, List, Tuple

import pandas as pd

from .base import BaseAnnotator
from ..prompts import annotate_knowledge_graph_retrieval_prompt
from ...utils import build_logger

logger = build_logger()


class KnowledgeGraphRetrievalAnnotator(BaseAnnotator):
    """Annotate which previous-session KG facts are used by system utterances.

    One LLM call is made per session because all system utterances in the same
    session retrieve from the same previous-session KG snapshot.
    """

    def __init__(self, context_length: int = 0, batch_size: int = 8, dialogue_context_window: int = 40, **kwargs):
        # context_length is accepted only for backward-compatible configs.
        # Retrieval now always uses previous-session KG only.
        super().__init__(**deepcopy(kwargs))
        self.context_length = 0
        self.batch_size = max(1, int(batch_size))
        # Number of current-session utterances to include before the first target in each batch.
        # Set <= 0 to include the full current session, but batching is usually safer for long sessions.
        self.dialogue_context_window = int(dialogue_context_window)

    def build_prompt(self, input_content: Dict[str, Any]) -> str:
        return annotate_knowledge_graph_retrieval_prompt(
            domain=self.domain_params["adj"],
            user=self.domain_params["user"],
            system=self.domain_params["system"],
            interaction_unit=self.domain_params["interaction_unit"],
            session_utterances=input_content["session_utterances"],
            target_utterances=input_content["target_utterances"],
            candidate_nodes=input_content["candidate_nodes"],
            candidate_relations=input_content["candidate_relations"],
            memory_session_idx=input_content["memory_session_idx"],
        )

    @staticmethod
    def _previous_session_kg(required_annotations: pd.DataFrame, user_idx: int, session_idx: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if required_annotations is None or required_annotations.empty or session_idx <= 0:
            return [], []

        memory_session_idx = session_idx - 1
        df = required_annotations[
            (required_annotations["user_idx"] == user_idx)
            & (required_annotations["session_idx"] == memory_session_idx)
        ]
        if df.empty:
            return [], []

        node_rows = df[df["kg_item"] == "node"].copy()
        rel_rows = df[df["kg_item"] == "relation"].copy()

        nodes = []
        for _, row in node_rows.sort_values("node_idx").iterrows():
            nodes.append(
                {
                    "node_idx": int(row["node_idx"]),
                    "type": str(row["node_type"]),
                    "content": str(row["node_content"]),
                    "node_session_idx": int(row.get("node_session_idx", row.get("source_session_idx", memory_session_idx))),
                }
            )

        relations = []
        for _, row in rel_rows.sort_values("relation_idx").iterrows():
            relations.append(
                {
                    "relation_idx": int(row["relation_idx"]),
                    "type": str(row["relation_type"]),
                    "source": int(row["source"]),
                    "target": int(row["target"]),
                    "source_node_type": str(row["source_node_type"]),
                    "source_node_content": str(row["source_node_content"]),
                    "source_node_session_idx": int(row.get("source_node_session_idx", 0)),
                    "target_node_type": str(row["target_node_type"]),
                    "target_node_content": str(row["target_node_content"]),
                    "target_node_session_idx": int(row.get("target_node_session_idx", 0)),
                    "available_session_idx": int(row.get("available_session_idx", memory_session_idx)),
                }
            )
        return nodes, relations

    def _target_batches(self, target_utterances: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        return [target_utterances[i : i + self.batch_size] for i in range(0, len(target_utterances), self.batch_size)]

    def _batch_session_context(self, utterances: List[Dict[str, Any]], target_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.dialogue_context_window <= 0 or not target_batch:
            return utterances

        target_ids = {u["utterance_id"] for u in target_batch}
        target_positions = [i for i, u in enumerate(utterances) if u.get("utterance_id") in target_ids]
        if not target_positions:
            return utterances

        start = max(0, min(target_positions) - self.dialogue_context_window)
        end = max(target_positions) + 1
        return utterances[start:end]

    def postprocess(self, outputs, data):
        if not isinstance(outputs, dict):
            logger.warning("[KG-RET] postprocess: output is not a dict; returning empty batch selection.")
            return {"utterances": []}

        allowed_system_ids = set(data.get("target_system_ids", []))
        allowed_nodes = set(data.get("candidate_node_ids", []))
        allowed_rels = set(data.get("candidate_relation_ids", []))
        rel_endpoints = data.get("relation_endpoints", {})

        cleaned = []
        seen_system_ids = set()
        for item_idx, item in enumerate(outputs.get("utterances", []) or []):
            if not isinstance(item, dict):
                logger.warning(f"[KG-RET] postprocess: dropped utterance item[{item_idx}] because it is not a dict.")
                continue
            system_id = str(item.get("system_id", "")).strip()
            if system_id not in allowed_system_ids:
                logger.warning(f"[KG-RET] postprocess: dropped item[{item_idx}] with invalid system_id={system_id!r}.")
                continue
            if system_id in seen_system_ids:
                logger.warning(f"[KG-RET] postprocess: dropped duplicate item for system_id={system_id!r}.")
                continue
            seen_system_ids.add(system_id)

            selected_nodes = []
            for x in item.get("nodes", []) or []:
                try:
                    node_idx = int(x)
                except Exception:
                    logger.warning(f"[KG-RET] postprocess: dropped non-integer node id {x!r} for {system_id}.")
                    continue
                if node_idx not in allowed_nodes:
                    logger.warning(f"[KG-RET] postprocess: dropped node_idx={node_idx} for {system_id}; not in candidate nodes.")
                    continue
                if node_idx not in selected_nodes:
                    selected_nodes.append(node_idx)

            selected_rels = []
            for x in item.get("relations", []) or []:
                try:
                    rel_idx = int(x)
                except Exception:
                    logger.warning(f"[KG-RET] postprocess: dropped non-integer relation id {x!r} for {system_id}.")
                    continue
                if rel_idx not in allowed_rels:
                    logger.warning(f"[KG-RET] postprocess: dropped relation_idx={rel_idx} for {system_id}; not in candidate relations.")
                    continue
                if rel_idx not in selected_rels:
                    selected_rels.append(rel_idx)
                for node_idx in rel_endpoints.get(rel_idx, []):
                    if node_idx in allowed_nodes and node_idx not in selected_nodes:
                        selected_nodes.append(node_idx)

            cleaned.append({"system_id": system_id, "nodes": selected_nodes, "relations": selected_rels})

        for system_id in sorted(allowed_system_ids):
            if system_id not in seen_system_ids:
                cleaned.append({"system_id": system_id, "nodes": [], "relations": []})

        return {"utterances": sorted(cleaned, key=lambda x: data["system_pos"].get(x["system_id"], 10**9))}

    def sanity_check(self, outputs, data) -> bool:
        if not isinstance(outputs, dict):
            logger.warning(f"[KG-RET] sanity_check failed: output is {type(outputs).__name__}, expected dict.")
            return False
        if set(outputs.keys()) != {"utterances"}:
            logger.warning(f"[KG-RET] sanity_check failed: keys={sorted(outputs.keys())}, expected ['utterances'].")
            return False
        if not isinstance(outputs["utterances"], list):
            logger.warning("[KG-RET] sanity_check failed: 'utterances' must be a list.")
            return False

        allowed_system_ids = set(data.get("target_system_ids", []))
        allowed_nodes = set(data.get("candidate_node_ids", []))
        allowed_rels = set(data.get("candidate_relation_ids", []))
        seen_system_ids = set()

        for item_idx, item in enumerate(outputs["utterances"]):
            if not isinstance(item, dict):
                logger.warning(f"[KG-RET] sanity_check failed: utterance item[{item_idx}] is not a dict: {item}")
                return False
            if set(item.keys()) != {"system_id", "nodes", "relations"}:
                logger.warning(f"[KG-RET] sanity_check failed: item[{item_idx}] has wrong keys {sorted(item.keys())}: {item}")
                return False
            if item["system_id"] not in allowed_system_ids:
                logger.warning(f"[KG-RET] sanity_check failed: invalid system_id={item['system_id']!r}.")
                return False
            if item["system_id"] in seen_system_ids:
                logger.warning(f"[KG-RET] sanity_check failed: duplicate system_id={item['system_id']!r}.")
                return False
            seen_system_ids.add(item["system_id"])
            if not isinstance(item["nodes"], list) or not isinstance(item["relations"], list):
                logger.warning(f"[KG-RET] sanity_check failed: nodes/relations must be lists for {item['system_id']}.")
                return False
            for node_idx in item["nodes"]:
                if not isinstance(node_idx, int):
                    logger.warning(f"[KG-RET] sanity_check failed: node id {node_idx!r} for {item['system_id']} is not int.")
                    return False
                if node_idx not in allowed_nodes:
                    logger.warning(f"[KG-RET] sanity_check failed: node_idx={node_idx} for {item['system_id']} is not in candidate nodes.")
                    return False
            for rel_idx in item["relations"]:
                if not isinstance(rel_idx, int):
                    logger.warning(f"[KG-RET] sanity_check failed: relation id {rel_idx!r} for {item['system_id']} is not int.")
                    return False
                if rel_idx not in allowed_rels:
                    logger.warning(f"[KG-RET] sanity_check failed: relation_idx={rel_idx} for {item['system_id']} is not in candidate relations.")
                    return False

        if seen_system_ids != allowed_system_ids:
            logger.warning(
                f"[KG-RET] sanity_check failed: missing target system ids "
                f"{sorted(allowed_system_ids - seen_system_ids)}."
            )
            return False
        return True

    def fallback(self, last_outputs, data):
        fixed = self.postprocess(last_outputs, data)
        if self.sanity_check(fixed, data):
            return fixed
        logger.warning("[KG-RET] fallback: returning empty KG retrieval selection for all target utterances.")
        return {
            "utterances": [
                {"system_id": sid, "nodes": [], "relations": []}
                for sid in data.get("target_system_ids", [])
            ]
        }

    def annotate_single_session(self, user_idx, session_idx, session, session_metadata, component, required_annotations):
        utterances, system_counter, user_counter = self.process_dialogue(session)
        if system_counter == 0:
            return [], {"inp_t": 0.0, "out_t": 0.0}

        target_utterances = [u for u in utterances if u["speaker"] == self.domain_params["system"]]
        system_pos = {u["utterance_id"]: int(u["utterance_id"].rsplit("_", 1)[1]) for u in target_utterances}
        memory_session_idx = session_idx - 1
        nodes, relations = self._previous_session_kg(required_annotations, user_idx, session_idx)
        node_by_idx = {n["node_idx"]: n for n in nodes}
        rel_by_idx = {r["relation_idx"]: r for r in relations}

        rows = []
        usage_total = {"inp_t": 0.0, "out_t": 0.0}

        def append_empty_rows():
            for utt in target_utterances:
                system_id = utt["utterance_id"]
                rows.append(
                    {
                        "user_idx": user_idx,
                        "session_idx": session_idx,
                        "system_id": system_id,
                        "system_idx": system_pos[system_id],
                        "memory_session_idx": memory_session_idx,
                        "retrieval_scope": "previous_sessions_only",
                        "kg_item": "system",
                        "num_used_nodes": 0,
                        "num_used_relations": 0,
                    }
                )

        if not nodes and not relations:
            append_empty_rows()
            return rows, usage_total

        used_by_system = {}
        target_batches = self._target_batches(target_utterances)
        logger.info(
            f"[KG-RET] user={user_idx}, session={session_idx}: annotating "
            f"{len(target_utterances)} system utterances in {len(target_batches)} batch(es) "
            f"with batch_size={self.batch_size}, dialogue_context_window={self.dialogue_context_window}."
        )

        for batch_idx, target_batch in enumerate(target_batches):
            batch_ids = [u["utterance_id"] for u in target_batch]
            data = {
                "target_system_ids": batch_ids,
                "system_pos": system_pos,
                "candidate_node_ids": [n["node_idx"] for n in nodes],
                "candidate_relation_ids": [r["relation_idx"] for r in relations],
                "relation_endpoints": {r["relation_idx"]: [r["source"], r["target"]] for r in relations},
            }
            input_content = {
                "session_utterances": self._batch_session_context(utterances, target_batch),
                "target_utterances": target_batch,
                "candidate_nodes": nodes,
                "candidate_relations": relations,
                "memory_session_idx": memory_session_idx,
            }
            prompt = self.build_prompt(input_content)
            parsed, usage = self.run_with_retries(prompt, data)
            usage_total["inp_t"] += usage["inp_t"]
            usage_total["out_t"] += usage["out_t"]

            for item in parsed["utterances"]:
                used_by_system[item["system_id"]] = item

            logger.info(
                f"[KG-RET] user={user_idx}, session={session_idx}: finished batch "
                f"{batch_idx + 1}/{len(target_batches)} with {len(batch_ids)} target utterances."
            )
        for utt in target_utterances:
            system_id = utt["utterance_id"]
            item = used_by_system.get(system_id, {"nodes": [], "relations": []})
            used_nodes = item["nodes"]
            used_relations = item["relations"]
            rows.append(
                {
                    "user_idx": user_idx,
                    "session_idx": session_idx,
                    "system_id": system_id,
                    "system_idx": system_pos[system_id],
                    "memory_session_idx": memory_session_idx,
                    "retrieval_scope": "previous_sessions_only",
                    "kg_item": "system",
                    "num_used_nodes": len(used_nodes),
                    "num_used_relations": len(used_relations),
                }
            )

            for node_idx in used_nodes:
                node = node_by_idx[node_idx]
                rows.append(
                    {
                        "user_idx": user_idx,
                        "session_idx": session_idx,
                        "system_id": system_id,
                        "system_idx": system_pos[system_id],
                        "memory_session_idx": memory_session_idx,
                        "retrieval_scope": "previous_sessions_only",
                        "kg_item": "node",
                        "node_idx": node_idx,
                        "node_type": node["type"],
                        "node_content": node["content"],
                        "node_session_idx": node["node_session_idx"],
                    }
                )

            for rel_idx in used_relations:
                rel = rel_by_idx[rel_idx]
                rows.append(
                    {
                        "user_idx": user_idx,
                        "session_idx": session_idx,
                        "system_id": system_id,
                        "system_idx": system_pos[system_id],
                        "memory_session_idx": memory_session_idx,
                        "retrieval_scope": "previous_sessions_only",
                        "kg_item": "relation",
                        "relation_idx": rel_idx,
                        "relation_type": rel["type"],
                        "source": rel["source"],
                        "target": rel["target"],
                        "source_node_type": rel["source_node_type"],
                        "source_node_content": rel["source_node_content"],
                        "source_node_session_idx": rel["source_node_session_idx"],
                        "target_node_type": rel["target_node_type"],
                        "target_node_content": rel["target_node_content"],
                        "target_node_session_idx": rel["target_node_session_idx"],
                        "available_session_idx": rel["available_session_idx"],
                    }
                )

        return rows, usage_total
