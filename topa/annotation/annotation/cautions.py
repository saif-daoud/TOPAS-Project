import json
from copy import deepcopy

from .base import BaseAnnotator
from ..prompts import annotate_cautions_prompt
from ..utils import format_utterances

from ...utils import build_logger
logger = build_logger()

class CautionsAnnotator(BaseAnnotator):
    def __init__(self, **kwargs):
        _kwargs = deepcopy(kwargs)
        context_length = _kwargs.pop("context_length")
        batch_size = _kwargs.pop("batch_size")
        super().__init__(**_kwargs)

        self.context_length = context_length
        self.batch_size = batch_size

    def build_prompt(self, input_content: str) -> str:
        return annotate_cautions_prompt(
            domain=self.domain_params["adj"],
            user=self.domain_params["user"],
            system=self.domain_params["system"],
            interaction_unit=self.domain_params["interaction_unit"],
            input_content=input_content
        )

    def sanity_check(self, outputs, data):
        expected_ids = data["expected_ids"]
        allowed = data["allowed"]

        if not isinstance(outputs, list):
            return False

        exp = set(expected_ids)
        allowed_set = set(allowed) | {"None", "none"}
        for ann in outputs:
            if not isinstance(ann, dict):
                return False
            if ann.get("utterance_id") not in exp:
                return False
            if ann.get("selected_negative_action") not in allowed_set:
                return False
            if "confidence_score" not in ann:
                return False
        return True

    def postprocess(self, outputs, data):
        return self.fallback(outputs, data)

    def fallback(self, last_outputs, data):
        """
        Fill missing or invalid entries with None:
        - Missing utterances get a placeholder with 'None'.
        - Invalid cautions are replaced with 'None'.
        """
        expected_ids = data["expected_ids"]
        allowed = data["allowed"] + ["None", "none"]
        parsed_dict = {p["utterance_id"]: p for p in last_outputs or []}

        filled_outputs = []
        for eid in expected_ids:
            if eid in parsed_dict:
                na = parsed_dict[eid].get("selected_negative_action", "None")
                conf = parsed_dict[eid].get("confidence_score", 1.0)
                if na not in allowed:
                    na = "None"
                    conf = 1.0
                filled_output = {
                    "utterance_id": eid,
                    "selected_negative_action": na,
                    "confidence_score": conf
                }
            else:
                # missing utterance -> fill with None
                filled_output = {
                    "utterance_id": eid,
                    "selected_negative_action": "None",
                    "confidence_score": 1.0
                }
            filled_outputs.append(filled_output)
        return filled_outputs

    def annotate_single_session(self, user_idx, session_idx, session, session_metadata, component, annotated_data):
        allowed = [a["negative_action"] for a in component]

        utterances, system_counter, user_counter = self.process_dialogue(session)
        sys_pos = {u["utterance_id"]: i for i, u in enumerate(utterances) 
                   if u["speaker"] == self.domain_params["system"]}
        expected_ids = [u["utterance_id"] for u in utterances if u["speaker"] == self.domain_params["system"]]

        rows = []
        usage_total = {"inp_t": 0.0, "out_t": 0.0}

        for b_start in range(0, len(expected_ids), self.batch_size):
            batch_ids = expected_ids[b_start:b_start + self.batch_size]
            start_pos = sys_pos[batch_ids[0]]
            end_pos = sys_pos[batch_ids[-1]]
            last_ctx = utterances[max(0, start_pos - self.context_length):start_pos]
            batch_slice = utterances[start_pos:end_pos + 1]

            input_content = json.dumps(
                {
                    "negative_actions": component, 
                    "last_utterances": format_utterances(last_ctx), 
                    "batch_utterances": format_utterances(batch_slice)
                },
                ensure_ascii=False,
                indent=2,
            )
            prompt = self.build_prompt(input_content)
            data = {"expected_ids": batch_ids, "allowed": allowed}
            parsed, usage = self.run_with_retries(prompt, data)

            usage_total["inp_t"] += usage["inp_t"]
            usage_total["out_t"] += usage["out_t"]

            if parsed is None:
                parsed = []

            by_id = {d["utterance_id"]: d for d in parsed}
            for eid in batch_ids:
                d = by_id[eid]
                rows.append(
                    {
                        "user_idx": user_idx,
                        "session_idx": session_idx,
                        "utterance_id": eid,
                        "utterance_idx": sys_pos[eid],
                        "selected_negative_action": d["selected_negative_action"],
                        "confidence_score": d["confidence_score"]
                    }
                )

        return rows, usage_total
