import json
import time
from copy import deepcopy
from typing import Any, Dict, List

from .base import BaseAnnotator
from tqdm import tqdm
from ..prompts import annotate_knowledge_graph_prompt, refine_knowledge_graph_prompt

from ...utils import build_logger

logger = build_logger()


class KnowledgeGraphAnnotator(BaseAnnotator):
    """Build one longitudinal KG per user, refining it after each session.

    Output is saved in annotations_knowledge_graph.csv with three row types:
      - kg_item == "session": one KG snapshot summary per session
      - kg_item == "node": nodes in the user KG snapshot after that session
      - kg_item == "relation": relations in the user KG snapshot after that session
    """

    def __init__(self, **kwargs):
        super().__init__(**deepcopy(kwargs))
        if self.num_processes > 1:
            logger.warning("[KG] Longitudinal KG refinement is order-dependent; forcing num_processes=1.")
            self.num_processes = 1
        self.user_kgs: Dict[int, Dict[str, Any]] = {}

    def build_prompt(self, input_content: Dict[str, Any]) -> str:
        common = dict(
            domain=self.domain_params["adj"],
            user=self.domain_params["user"],
            system=self.domain_params["system"],
            interaction_unit=self.domain_params["interaction_unit"],
            kg_components=input_content["kg_components"],
            session_utterances=input_content["session_utterances"],
            current_session_idx=input_content["current_session_idx"],
        )
        previous_kg = input_content.get("previous_kg") or {"nodes": [], "relations": []}
        if previous_kg.get("nodes") or previous_kg.get("relations"):
            return refine_knowledge_graph_prompt(previous_kg=previous_kg, **common)
        return annotate_knowledge_graph_prompt(**common)

    @staticmethod
    def _allowed_sets(component: Dict[str, Any]):
        node_types = {str(x.get("type")) for x in component.get("nodes", []) if x.get("type")}
        rel_types = {str(x.get("relation")) for x in component.get("edges", []) if x.get("relation")}
        return node_types, rel_types

    def postprocess(self, outputs, data):
        if not isinstance(outputs, dict):
            logger.warning(f"[KG] postprocess: expected dict output, got {type(outputs).__name__}.")
            return {"nodes": [], "relations": []}

        current_session_idx = int(data["current_session_idx"])
        cleaned_nodes = []
        seen = {}
        old_to_new = {}
        for old_idx, node in enumerate(outputs.get("nodes", []) or []):
            if not isinstance(node, dict):
                logger.warning(f"[KG] postprocess: dropped node[{old_idx}] because it is not a dict.")
                continue
            ntype = str(node.get("type", "")).strip()
            content = str(node.get("content", "")).strip()
            try:
                node_session_idx = int(node.get("session_idx", current_session_idx))
            except Exception:
                logger.warning(f"[KG] postprocess: dropped node[{old_idx}] because session_idx is invalid: {node}")
                continue
            if not ntype or not content:
                logger.warning(f"[KG] postprocess: dropped node[{old_idx}] because type/content is missing: {node}")
                continue
            if node_session_idx < 0 or node_session_idx > current_session_idx:
                logger.warning(
                    f"[KG] postprocess: dropped node[{old_idx}] because session_idx={node_session_idx} "
                    f"is outside [0, {current_session_idx}]: {node}"
                )
                continue
            key = (ntype.lower(), content.lower())
            if key in seen:
                old_to_new[old_idx] = seen[key]
                continue
            new_idx = len(cleaned_nodes)
            seen[key] = new_idx
            old_to_new[old_idx] = new_idx
            cleaned_nodes.append({"type": ntype, "content": content, "session_idx": node_session_idx})

        cleaned_relations = []
        for rel_idx, rel in enumerate(outputs.get("relations", []) or []):
            if not isinstance(rel, dict):
                logger.warning(f"[KG] postprocess: dropped relation[{rel_idx}] because it is not a dict.")
                continue
            rtype = str(rel.get("type", "")).strip()
            try:
                source = old_to_new[int(rel.get("source"))]
                target = old_to_new[int(rel.get("target"))]
            except Exception:
                logger.warning(
                    f"[KG] postprocess: dropped relation[{rel_idx}] because source/target are invalid "
                    f"or point to a dropped node: {rel}"
                )
                continue
            if not rtype:
                logger.warning(f"[KG] postprocess: dropped relation[{rel_idx}] because type is missing: {rel}")
                continue
            cleaned_relations.append({"type": rtype, "source": source, "target": target})

        return {"nodes": cleaned_nodes, "relations": cleaned_relations}

    def sanity_check(self, outputs, data) -> bool:
        if not isinstance(outputs, dict):
            logger.warning(f"[KG] sanity_check failed: output is {type(outputs).__name__}, expected dict.")
            return False
        if set(outputs.keys()) != {"nodes", "relations"}:
            logger.warning(f"[KG] sanity_check failed: expected keys ['nodes', 'relations'], got {sorted(outputs.keys())}.")
            return False
        if not isinstance(outputs["nodes"], list) or not isinstance(outputs["relations"], list):
            logger.warning("[KG] sanity_check failed: 'nodes' and 'relations' must both be lists.")
            return False

        allowed_nodes = data["allowed_nodes"]
        allowed_rels = data["allowed_rels"]
        current_session_idx = int(data["current_session_idx"])

        for node_idx, node in enumerate(outputs["nodes"]):
            if not isinstance(node, dict):
                logger.warning(f"[KG] sanity_check failed: node[{node_idx}] is not a dict: {node}")
                return False
            if set(node.keys()) != {"type", "content", "session_idx"}:
                logger.warning(f"[KG] sanity_check failed: node[{node_idx}] has wrong keys {sorted(node.keys())}: {node}")
                return False
            if node["type"] not in allowed_nodes:
                logger.warning(f"[KG] sanity_check failed: node[{node_idx}] has invalid type {node['type']!r}.")
                return False
            if not isinstance(node["content"], str) or not node["content"].strip():
                logger.warning(f"[KG] sanity_check failed: node[{node_idx}] has empty/non-string content: {node}")
                return False
            if not isinstance(node["session_idx"], int):
                logger.warning(f"[KG] sanity_check failed: node[{node_idx}] session_idx must be int: {node}")
                return False
            if node["session_idx"] < 0 or node["session_idx"] > current_session_idx:
                logger.warning(
                    f"[KG] sanity_check failed: node[{node_idx}] session_idx={node['session_idx']} "
                    f"outside [0, {current_session_idx}]."
                )
                return False

        n_nodes = len(outputs["nodes"])
        for rel_idx, rel in enumerate(outputs["relations"]):
            if not isinstance(rel, dict):
                logger.warning(f"[KG] sanity_check failed: relation[{rel_idx}] is not a dict: {rel}")
                return False
            if set(rel.keys()) != {"type", "source", "target"}:
                logger.warning(f"[KG] sanity_check failed: relation[{rel_idx}] has wrong keys {sorted(rel.keys())}: {rel}")
                return False
            if rel["type"] not in allowed_rels:
                logger.warning(f"[KG] sanity_check failed: relation[{rel_idx}] has invalid type {rel['type']!r}.")
                return False
            if not isinstance(rel["source"], int) or not isinstance(rel["target"], int):
                logger.warning(f"[KG] sanity_check failed: relation[{rel_idx}] source/target must be ints: {rel}")
                return False
            if rel["source"] < 0 or rel["source"] >= n_nodes:
                logger.warning(f"[KG] sanity_check failed: relation[{rel_idx}].source={rel['source']} out of range for n_nodes={n_nodes}.")
                return False
            if rel["target"] < 0 or rel["target"] >= n_nodes:
                logger.warning(f"[KG] sanity_check failed: relation[{rel_idx}].target={rel['target']} out of range for n_nodes={n_nodes}.")
                return False
        return True

    def fallback(self, last_outputs, data):
        fixed = self.postprocess(last_outputs, data)
        if self.sanity_check(fixed, data):
            return fixed
        logger.warning("[FALLBACK] Returning previous/empty KG for session.")
        return data.get("previous_kg") or {"nodes": [], "relations": []}

    @staticmethod
    def _kg_to_rows(user_idx: int, session_idx: int, kg: Dict[str, Any]) -> List[Dict[str, Any]]:
        nodes = kg["nodes"]
        relations = kg["relations"]
        rows = [
            {
                "user_idx": user_idx,
                "session_idx": session_idx,
                "kg_item": "session",
                "num_nodes": len(nodes),
                "num_relations": len(relations),
            }
        ]

        for node_idx, node in enumerate(nodes):
            rows.append(
                {
                    "user_idx": user_idx,
                    "session_idx": session_idx,
                    "kg_item": "node",
                    "node_idx": node_idx,
                    "node_type": node["type"],
                    "node_content": node["content"],
                    "node_session_idx": int(node["session_idx"]),
                }
            )

        for rel_idx, rel in enumerate(relations):
            src = nodes[rel["source"]]
            tgt = nodes[rel["target"]]
            available_session_idx = max(int(src["session_idx"]), int(tgt["session_idx"]))
            rows.append(
                {
                    "user_idx": user_idx,
                    "session_idx": session_idx,
                    "kg_item": "relation",
                    "relation_idx": rel_idx,
                    "relation_type": rel["type"],
                    "source": rel["source"],
                    "target": rel["target"],
                    "source_node_type": src["type"],
                    "source_node_content": src["content"],
                    "source_node_session_idx": int(src["session_idx"]),
                    "target_node_type": tgt["type"],
                    "target_node_content": tgt["content"],
                    "target_node_session_idx": int(tgt["session_idx"]),
                    "available_session_idx": available_session_idx,
                }
            )
        return rows


    @staticmethod
    def _kg_from_annotation_rows(rows: List[Dict[str, Any]], user_idx: int, session_idx: int):
        """Rebuild the saved KG snapshot for one completed user/session.

        This is only used when resuming annotation, because the longitudinal KG
        state must be restored before moving to later sessions.
        """
        session_rows = [
            r for r in rows
            if int(r.get("user_idx", -1)) == int(user_idx)
            and int(r.get("session_idx", -1)) == int(session_idx)
        ]
        if not session_rows:
            return None

        node_rows = [r for r in session_rows if r.get("kg_item") == "node"]
        rel_rows = [r for r in session_rows if r.get("kg_item") == "relation"]

        nodes = []
        for row in sorted(node_rows, key=lambda r: int(float(r.get("node_idx", 0)))):
            nodes.append(
                {
                    "type": str(row["node_type"]),
                    "content": str(row["node_content"]),
                    "session_idx": int(float(row.get("node_session_idx", session_idx))),
                }
            )

        relations = []
        for row in sorted(rel_rows, key=lambda r: int(float(r.get("relation_idx", 0)))):
            relations.append(
                {
                    "type": str(row["relation_type"]),
                    "source": int(float(row["source"])),
                    "target": int(float(row["target"])),
                }
            )

        return {"nodes": nodes, "relations": relations}

    def annotate(self, data, component, component_name, annotation_path, session_filter: set | None = None):
        """Run longitudinal KG annotation in chronological user/session order.

        We intentionally do not use BaseAnnotator's parallel branch here because
        session s depends on the KG snapshot after session s-1 for the same user.
        """
        start_time = time.time()
        all_annotations, completed_sessions, running_usage, previous_runtime, annotations_path = self.load_existing_output(
            annotation_path, component_name
        )

        if completed_sessions:
            logger.info(
                f"[KG] Resuming {component_name}: found {len(completed_sessions)} completed sessions "
                f"and {len(all_annotations)} existing rows."
            )

        for user_idx, user in enumerate(data):
            self.user_kgs.setdefault(user_idx, {"nodes": [], "relations": []})
            for session_idx, session in tqdm(
                enumerate(user["sessions"]),
                total=len(user["sessions"]),
                desc=f"KG user {user_idx}",
            ):
                if session_filter is not None and (user_idx, session_idx) not in session_filter:
                    continue

                if (user_idx, session_idx) in completed_sessions:
                    saved_kg = self._kg_from_annotation_rows(all_annotations, user_idx, session_idx)
                    if saved_kg is not None:
                        self.user_kgs[user_idx] = saved_kg
                    else:
                        logger.warning(
                            f"[KG] Session user={user_idx}, session={session_idx} is marked completed, "
                            "but no KG snapshot rows were found to restore state."
                        )
                    continue

                if session_idx > 0 and not self.user_kgs.get(user_idx, {}).get("nodes"):
                    logger.warning(
                        f"[KG] Annotating user={user_idx}, session={session_idx} with an empty previous KG. "
                        "Make sure previous sessions were annotated first unless this is intentional."
                    )

                session_annotations, usage = self.annotate_single_session(
                    user_idx,
                    session_idx,
                    session["dialogue"],
                    session["session_metadata"],
                    component,
                    None,
                )
                all_annotations.extend(session_annotations)
                running_usage["inp_t"] += usage["inp_t"]
                running_usage["out_t"] += usage["out_t"]

                cost_data = self.build_cost_data(start_time, previous_runtime, running_usage)
                annotations_path = self.save_output(all_annotations, cost_data, annotation_path, component_name)

        return annotations_path

    def annotate_single_session(self, user_idx, session_idx, session, session_metadata, component, *args):
        utterances, system_counter, user_counter = self.process_dialogue(session)
        previous_kg = self.user_kgs.get(user_idx, {"nodes": [], "relations": []})

        allowed_nodes, allowed_rels = self._allowed_sets(component)
        data = {
            "allowed_nodes": allowed_nodes,
            "allowed_rels": allowed_rels,
            "current_session_idx": session_idx,
            "previous_kg": previous_kg,
        }

        if system_counter == 0 and user_counter == 0:
            self.user_kgs[user_idx] = previous_kg
            return self._kg_to_rows(user_idx, session_idx, previous_kg), {"inp_t": 0.0, "out_t": 0.0}

        input_content = {
            "kg_components": component,
            "previous_kg": previous_kg,
            "session_utterances": utterances,
            "current_session_idx": session_idx,
        }
        prompt = self.build_prompt(input_content)
        parsed, usage = self.run_with_retries(prompt, data)
        if not self.sanity_check(parsed, data):
            logger.warning(f"[KG] No valid KG produced for user={user_idx}, session={session_idx}; keeping previous KG.")
            parsed = previous_kg

        self.user_kgs[user_idx] = parsed
        return self._kg_to_rows(user_idx, session_idx, parsed), usage
