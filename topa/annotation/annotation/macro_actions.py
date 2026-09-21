import json
from copy import deepcopy

from .base import BaseAnnotator
from ..prompts import annotate_macro_actions_prompt, annotate_macro_actions_with_rules_prompt
from ..utils import get_idx

from ...utils import build_logger
logger = build_logger()

class MacroActionsAnnotator(BaseAnnotator):
    def __init__(self, **kwargs):
        _kwargs = deepcopy(kwargs)
        use_annotation_rules = _kwargs.pop("use_annotation_rules")
        super().__init__(**_kwargs)
        self.use_annotation_rules = use_annotation_rules

    def build_prompt(self, input_content):
        if self.use_annotation_rules:
            prompt_fn = annotate_macro_actions_with_rules_prompt
        else:
            prompt_fn = annotate_macro_actions_prompt
        return prompt_fn(
            domain=self.domain_params["adj"],
            user=self.domain_params["user"],
            system=self.domain_params["system"],
            interaction_unit=self.domain_params["interaction_unit"],
            input_content=input_content
        )

    def postprocess(self, outputs, data):
        if not isinstance(outputs, list):
            return []
        spans = [
            (get_idx(o["from"]), get_idx(o["to"]), o)
            for o in outputs
            if (
                isinstance(o, dict)
                and isinstance(o.get("from"), str)
                and isinstance(o.get("to"), str)
                and o["from"].rsplit("_", 1)[-1].isdigit()
                and o["to"].rsplit("_", 1)[-1].isdigit()
            )
        ]
        spans.sort(key=lambda x: x[0])
        cleaned, prev_end = [], -1
        for s, e, obj in spans:
            if s <= prev_end:
                continue
            cleaned.append(obj)
            prev_end = e
        return cleaned

    def sanity_check(self, outputs, data):
        num_utts = data["num_utts"]
        actions = data["allowed_actions"] + ["uncertain"]
        sys_pos = data["sys_pos"]

        if not isinstance(outputs, list) or not outputs:
            return False

        spans = []
        for ann in outputs:
            if not isinstance(ann, dict):
                return False
            action = ann.get("selected_macro_action")
            from_id = ann.get("from")
            to_id = ann.get("to")

            if action not in actions:
                return False
            if not isinstance(from_id, str) or not isinstance(to_id, str):
                return False
            if not self.has_valid_system_ids([from_id, to_id], sys_pos, context="macro action boundaries"):
                return False
            if not isinstance(ann.get("confidence_score"), dict) or not ann["confidence_score"]:
                return False
            if self.use_annotation_rules and "annotation_rule_used" not in ann:
                return False

            start_idx = get_idx(from_id)
            end_idx = get_idx(to_id)
            if start_idx > end_idx:
                return False
            spans.append((start_idx, end_idx))

        # Check coverage
        if spans[0][0] != 0 or spans[-1][1] != num_utts - 1:
            return False

        # Check continuity
        for i in range(len(spans) - 1):
            if spans[i][1] + 1 != spans[i + 1][0]:
                return False

        return True

    def fallback(self, last_outputs, data):
        logger.warning("[FALLBACK] Returning last outputs for macro actions.")
        return last_outputs

    def annotate_single_session(self, user_idx, session_idx, session, session_metadata, component, *args):
        macro_actions = component
        allowed_macro_actions = [a["name"] for a in macro_actions]
        utterances, system_counter, user_counter = self.process_dialogue(session)
        sys_pos = {u["utterance_id"]: i for i, u in enumerate(utterances) 
                   if u["speaker"] == self.domain_params["system"]}

        data = {"num_utts": system_counter, "allowed_actions": allowed_macro_actions, "sys_pos": sys_pos}
        input_content = json.dumps(
            {"macro_actions": macro_actions, "session_utterances": utterances},
            ensure_ascii=False, indent=2
        )
        prompt = self.build_prompt(input_content)
        parsed, usage = self.run_with_retries(prompt, data)
        if not self.sanity_check(parsed, data):
            logger.warning(
                f"[MACRO] No valid macro segmentation produced for user={user_idx}, session={session_idx}."
            )
            return [], usage

        # flatten to rows
        rows = []
        for phase_idx, ann in enumerate(parsed):
            from_id = ann["from"]
            to_id = ann["to"]
            row = {
                "user_idx": user_idx,
                "session_idx": session_idx,
                "phase_idx": phase_idx,
                "from_id": from_id,
                "to_id": to_id,
                "from_idx": sys_pos[from_id],
                "to_idx": sys_pos[to_id],
                "selected_macro_action": ann["selected_macro_action"],
            }

            conf = ann["confidence_score"]
            s = 0.0
            for k, v in conf.items():
                row[f"conf_{k}"] = v
                s += v
            row["confidence_score"] = s / len(conf)
            if self.use_annotation_rules:
                row["annotation_rule_used"] = ann["annotation_rule_used"]
            rows.append(row)

        return rows, usage
