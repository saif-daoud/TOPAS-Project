import json
from copy import deepcopy

from .base import BaseAnnotator
from ..prompts import (
    annotate_micro_actions_direct_prompt,
    annotate_micro_actions_direct_with_rules_prompt,
)

from ...utils import build_logger
logger = build_logger()


class MicroActionsDirectAnnotator(BaseAnnotator):
    def __init__(self, **kwargs):
        _kwargs = deepcopy(kwargs)
        use_annotation_rules = _kwargs.pop("use_annotation_rules")
        batch_size = _kwargs.pop("batch_size")
        context_length = _kwargs.pop("context_length")
        super().__init__(**_kwargs)

        self.context_length = context_length
        self.batch_size = batch_size
        self.use_annotation_rules = use_annotation_rules

    def build_prompt(self, input_content: str) -> str:
        prompt_fn = (
            annotate_micro_actions_direct_with_rules_prompt
            if self.use_annotation_rules
            else annotate_micro_actions_direct_prompt
        )
        return prompt_fn(
            domain=self.domain_params["adj"],
            user=self.domain_params["user"],
            system=self.domain_params["system"],
            interaction_unit=self.domain_params["interaction_unit"],
            input_content=input_content,
        )

    def _flatten_micro_actions(self, component):
        seen = {}
        micro_actions = []
        for macro in component:
            for micro in macro.get("micro_actions", []):
                name = micro.get("name")
                if not name:
                    continue
                if name in seen:
                    prev = seen[name]
                    if (
                        prev.get("description") != micro.get("description")
                        or prev.get("states") != micro.get("states")
                    ):
                        logger.warning(
                            f"[MICRO_DIRECT] Duplicate micro action '{name}' with inconsistent metadata. "
                            "Keeping the first definition."
                        )
                    continue
                item = {
                    "name": name,
                    "description": micro.get("description", ""),
                }
                if self.use_annotation_rules and "states" in micro:
                    item["states"] = micro.get("states")
                micro_actions.append(item)
                seen[name] = item
        return micro_actions

    def sanity_check(self, outputs, data):
        expected_ids = data["expected_ids"]
        allowed_micro = data["allowed_micro"] + ["None"]

        if not isinstance(outputs, list):
            return False

        if not all(isinstance(out, dict) and "utterance_id" in out for out in outputs):
            return False

        annotated_utts = {out["utterance_id"] for out in outputs}
        missing = set(expected_ids) - annotated_utts
        if missing:
            logger.warning(
                f"[SANITY CHECK] Missing {len(missing)}/{len(expected_ids)} annotations for system utterances !"
            )
            return False

        for ann in outputs:
            utterance_id = ann.get("utterance_id")
            if utterance_id not in expected_ids:
                return False
            if ann.get("selected_micro_action") not in allowed_micro:
                logger.warning(
                    f"[SANITY CHECK] Invalid micro action '{ann.get('selected_micro_action')}' !"
                )
                return False
            if self.use_annotation_rules and "annotation_rule_used" not in ann:
                return False
        return True

    def fallback(self, last_outputs, data):
        expected_ids = data["expected_ids"]
        allowed_micro = data["allowed_micro"] + ["None"]
        parsed_dict = {p["utterance_id"]: p for p in last_outputs or []}

        filled_outputs = []
        for eid in expected_ids:
            if eid in parsed_dict:
                micro = parsed_dict[eid].get("selected_micro_action", "None")
                conf = parsed_dict[eid].get("confidence_score", 1.0)
                if micro not in allowed_micro:
                    micro = "None"
                    conf = 1.0
                filled_output = {
                    "utterance_id": eid,
                    "selected_micro_action": micro,
                    "confidence_score": conf,
                }
                if self.use_annotation_rules:
                    filled_output["annotation_rule_used"] = parsed_dict[eid].get(
                        "annotation_rule_used", "None"
                    )
            else:
                filled_output = {
                    "utterance_id": eid,
                    "selected_micro_action": "None",
                    "confidence_score": 1.0,
                }
                if self.use_annotation_rules:
                    filled_output["annotation_rule_used"] = "None"
            filled_outputs.append(filled_output)
        return filled_outputs

    def annotate_single_session(self, user_idx, session_idx, session, session_metadata, component, annotated_data):
        utterances, system_counter, user_counter = self.process_dialogue(session)
        micro_actions = self._flatten_micro_actions(component)
        allowed_micro = [m["name"] for m in micro_actions]
        if not allowed_micro:
            logger.warning(
                f"[MICRO_DIRECT] No micro actions available for user={user_idx} session={session_idx}."
            )
            return [], {"inp_t": 0.0, "out_t": 0.0}

        sys_ids_by_pos = [
            (i, u["utterance_id"])
            for i, u in enumerate(utterances)
            if u["speaker"] == self.domain_params["system"]
        ]
        sys_pos = {uid: idx for idx, uid in sys_ids_by_pos}
        expected_ids = [uid for _, uid in sys_ids_by_pos]

        rows = []
        usage_total = {"inp_t": 0.0, "out_t": 0.0}

        for b_start in range(0, len(expected_ids), self.batch_size):
            b_end = min(b_start + self.batch_size, len(expected_ids))
            batch_ids = expected_ids[b_start:b_end]
            if not batch_ids:
                continue

            start_pos = sys_pos[batch_ids[0]]
            end_pos = sys_pos[batch_ids[-1]]
            last_ctx = utterances[max(0, start_pos - self.context_length):start_pos]
            batch_slice = utterances[start_pos:end_pos + 1]

            input_content = json.dumps(
                {
                    "micro_actions": micro_actions,
                    "last_utterances": last_ctx,
                    "batch_utterances": batch_slice,
                },
                ensure_ascii=False,
                indent=2,
            )

            data = {"expected_ids": batch_ids, "allowed_micro": allowed_micro}
            prompt = self.build_prompt(input_content)
            parsed, usage = self.run_with_retries(prompt, data)

            usage_total["inp_t"] += usage["inp_t"]
            usage_total["out_t"] += usage["out_t"]

            if parsed is None:
                parsed = []

            by_id = {d["utterance_id"]: d for d in parsed}
            for eid in batch_ids:
                d = by_id[eid]
                row = {
                    "user_idx": user_idx,
                    "session_idx": session_idx,
                    "utterance_id": eid,
                    "utterance_idx": sys_pos[eid],
                    "selected_micro_action": d["selected_micro_action"],
                    "confidence_score": d["confidence_score"],
                }
                if self.use_annotation_rules:
                    row["annotation_rule_used"] = d.get("annotation_rule_used", "None")
                rows.append(row)

        return rows, usage_total
