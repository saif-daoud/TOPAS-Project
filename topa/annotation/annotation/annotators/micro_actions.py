import json
from typing import Optional
from copy import deepcopy
import pandas as pd

from .base import BaseAnnotator
from ..prompts import annotate_micro_actions_prompt, annotate_micro_actions_with_rules_prompt

from ...utils import build_logger
logger = build_logger()

class MicroActionsAnnotator(BaseAnnotator):
    def __init__(self, **kwargs):
        _kwargs = deepcopy(kwargs)
        use_annotation_rules = _kwargs.pop("use_annotation_rules")
        batch_size = _kwargs.pop("batch_size")
        context_length = _kwargs.pop("context_length")
        super().__init__(**_kwargs)

        self.context_length = context_length
        self.batch_size = batch_size
        self.use_annotation_rules = use_annotation_rules

        self._macro_df: Optional[pd.DataFrame] = None

    def build_prompt(self, input_content: str) -> str:
        if self.use_annotation_rules:
            prompt_fn = annotate_micro_actions_with_rules_prompt
        else:
            prompt_fn = annotate_micro_actions_prompt
        return prompt_fn(
            domain=self.domain_params["adj"],
            user=self.domain_params["user"],
            system=self.domain_params["system"],
            interaction_unit=self.domain_params["interaction_unit"],
            input_content=input_content
        )

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
            logger.warning(f"[SANITY CHECK] Missing {len(missing)}/{len(expected_ids)} annotations for system utterances !")
            return False

        for ann in outputs:
            utterance_id = ann.get("utterance_id")
            if utterance_id not in expected_ids:
                return False
            if ann.get("selected_micro_action") not in allowed_micro:
                logger.warning(f"[SANITY CHECK] Invalid micro action '{ann.get('selected_micro_action')}' !")
                return False
            if self.use_annotation_rules and "annotation_rule_used" not in ann:
                return False
        return True

    def fallback(self, last_outputs, data):
        """
        Fill missing or invalid entries with None:
        - Missing utterances get a placeholder with 'None'.
        - Invalid micro actions are replaced with 'None'.
        """
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
                    "confidence_score": conf
                }
                if self.use_annotation_rules:
                    filled_output["annotation_rule_used"] = parsed_dict[eid].get("annotation_rule_used", "None")
            else:
                # missing utterance -> fill with None
                filled_output = {
                    "utterance_id": eid,
                    "selected_micro_action": "None",
                    "confidence_score": 1.0
                }
                if self.use_annotation_rules:
                    filled_output["annotation_rule_used"] = "None"
            filled_outputs.append(filled_output)
        return filled_outputs

    def annotate_single_session(self, user_idx, session_idx, session, session_metadata, component, annotated_data):
        mask = (annotated_data["user_idx"] == user_idx) & (annotated_data["session_idx"] == session_idx)
        phases = annotated_data.loc[mask].copy()
        
        from_col = "from_idx" if "from_idx" in phases.columns else "from_id"
        phases = phases.sort_values(by=[from_col])
        macro_by_name = {ma["name"]: ma for ma in component}

        utterances, system_counter, user_counter = self.process_dialogue(session)
        sys_ids_by_pos = [
            (i, u["utterance_id"])
            for i, u in enumerate(utterances)
            if u["speaker"] == self.domain_params["system"]
        ]
        sys_pos = {uid: idx for idx, uid in sys_ids_by_pos}

        rows = []
        usage_total = {"inp_t": 0.0, "out_t": 0.0}

        for _, ph in phases.iterrows():
            action_name = ph["selected_macro_action"]
            if action_name == "uncertain":
                continue
            macro_def = macro_by_name[action_name]

            micro_actions = macro_def["micro_actions"]
            allowed_micro = [m["name"] for m in micro_actions]

            # Macro annotations store from_idx/to_idx as indices into the full dialogue,
            # so we select the system utterances that fall within that range.
            if "from_idx" in ph.index and "to_idx" in ph.index:
                from_idx = int(ph["from_idx"])
                to_idx = int(ph["to_idx"])

                expected_ids = [
                    uid for idx, uid in sys_ids_by_pos
                    if from_idx <= idx <= to_idx
                ]

                range_msg = f"{from_idx}-{to_idx}"
            else:
                from_id = int(ph["from_id"])
                to_id = int(ph["to_id"])

                expected_ids = [
                    uid for _, uid in sys_ids_by_pos[from_id:to_id + 1]
                ]

                range_msg = f"system turns {from_id}-{to_id}"

            if not expected_ids:
                logger.warning(
                    f"[MICRO] No system utterances in range {range_msg} "
                    f"for user={user_idx} session={session_idx}; skipping."
                )
                continue

            # batch over system utterances
            for b_start in range(0, len(expected_ids), self.batch_size):
                b_end = min(b_start + self.batch_size, len(expected_ids))
                batch_ids = expected_ids[b_start:b_end]
                missing_ids = [eid for eid in batch_ids if eid not in sys_pos]
                if missing_ids:
                    logger.warning(
                        f"[MICRO] Missing system utterance ids in session (user={user_idx}, session={session_idx}): "
                        f"{missing_ids}. Skipping those."
                    )
                    batch_ids = [eid for eid in batch_ids if eid in sys_pos]
                    if not batch_ids:
                        continue
                start_pos = sys_pos[batch_ids[0]]
                end_pos = sys_pos[batch_ids[-1]]
                last_ctx = utterances[max(0, start_pos - self.context_length):start_pos]
                batch_slice = utterances[start_pos:end_pos + 1]

                input_content = json.dumps(
                    {
                        "selected_macro_action": {
                            "name": macro_def["name"],
                            "description": macro_def["description"],
                            "goal": macro_def["goal"],
                        },
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
                    # fill with None
                    parsed = []

                # index by utterance_id
                by_id = {d["utterance_id"]: d for d in parsed}
                for eid in batch_ids:
                    d = by_id[eid]
                    row = {
                        "user_idx": user_idx,
                        "session_idx": session_idx,
                        "utterance_id": eid,
                        "utterance_idx": sys_pos[eid],
                        "selected_macro_action": action_name,
                        "selected_micro_action": d["selected_micro_action"],
                        "confidence_score": d["confidence_score"]
                    }
                    if self.use_annotation_rules:
                        row["annotation_rule_used"] = d.get("annotation_rule_used", 'None')
                    rows.append(row)

        return rows, usage_total