import json
from typing import Optional
from copy import deepcopy
import pandas as pd

from .base import BaseAnnotator
from ..prompts import annotate_conversation_state_prompt
from ..utils import format_utterances

from ...utils import build_logger
logger = build_logger()

class ConversationStateAnnotator(BaseAnnotator):
    def __init__(self, **kwargs):
        _kwargs = deepcopy(kwargs)
        numeric = _kwargs.pop("numeric")
        super().__init__(**_kwargs)

        self.numeric = numeric
        self._macro_df: Optional[pd.DataFrame] = None

    def build_prompt(self, input_content: dict) -> str:
        return annotate_conversation_state_prompt(
            domain=self.domain_params["adj"],
            user=self.domain_params["user"],
            system=self.domain_params["system"],
            interaction_unit=self.domain_params["interaction_unit"],
            macro_st_dim=input_content["macro_st_dim"],
            last_macro_state=input_content["last_macro_state"],
            new_utterances=input_content["new_utterances"],
            numeric=self.numeric
        )

    def sanity_check(self, outputs, data) -> bool:
        dims = data["dims"]
        if not isinstance(outputs, list):
            return False
        if len(outputs) != len(dims):
            logger.info(f"mismatched length {len(outputs)} - {len(dims)} !")
            return False

        if self.numeric:
            for v in outputs:
                try:
                    fv = float(v)
                except Exception:
                    return False
                if fv < 0.0 or fv > 1.0:
                    return False
        else:
            # categorical: each value must be in allowed cat values OR "Unknown"/"none"
            for v, dim in zip(outputs, dims):
                allowed = set(dim["Categorical values"] + ["none", "None"])
                if str(v) not in allowed:
                    # if no allowed list, we accept any string
                    if len(allowed) > 2:
                        logger.info(f"Inconsistent category in dim {dim['Variable Name']}, predicted {v}, allowed {allowed}")
                        return False
        return True

    def postprocess(self, outputs, data):
        dims = data["dims"]
        default_value = "none" if not self.numeric else -1
        return [outputs.get(f"dim_{i}", default_value) for i in range(len(dims))]

    def fallback(self, last_outputs, data):
        logger.warning("[FALLBACK] Returning last outputs for conversation state.")
        outputs = []
        for v, dim in zip(last_outputs, data["dims"]):
            allowed = set(dim["Categorical values"] + ["none", "None"])
            if str(v) not in allowed and len(allowed) > 2:
                outputs.append("none")
            else:
                outputs.append(v)

        return last_outputs

    def annotate_single_session(self, user_idx, session_idx, session, session_metadata, component, annotated_data):
        mask = (annotated_data["user_idx"] == user_idx) & (annotated_data["session_idx"] == session_idx)
        phases = annotated_data.loc[mask].copy()
        phases = phases.sort_values(by=["from_idx"])

        utterances, system_counter, user_counter = self.process_dialogue(session)
        sys_pos = {u["utterance_id"]: i for i, u in enumerate(utterances) 
                   if u["speaker"] == self.domain_params["system"]}

        # initial state
        last_state = [-1 if self.numeric else "none"] * len(component)

        dims_per_iter = 62
        components_idx = [(start, min(start + dims_per_iter, len(component)))
                      for start in range(0, len(component), dims_per_iter)]

        rows = []
        usage_total = {"inp_t": 0.0, "out_t": 0.0}

        for phase_idx, ph in phases.iterrows():
            action_name = ph["selected_macro_action"]
            if action_name == "uncertain":
                continue

            from_id = ph["from_id"]
            to_id = ph["to_id"]
            start_pos = sys_pos[from_id]
            end_pos = sys_pos[to_id]
            new_utts = utterances[start_pos:end_pos + 1]
            all_parsed = []
            for start, end in components_idx:
                _component = deepcopy(component[start: end])
                _last_state = deepcopy(last_state[start: end])
                _component = [{"index": f"dim_{i}", **c} for i, c in enumerate(_component)]
                _last_state = {f"dim_{i}": s for i, s in enumerate(_last_state)}
                input_content = {
                    "macro_st_dim": json.dumps(_component, ensure_ascii=False, indent=2),
                    "last_macro_state": json.dumps(_last_state, ensure_ascii=False),
                    "new_utterances": format_utterances(new_utts)
                }
                prompt = self.build_prompt(input_content)
                data = {"dims": _component}
                parsed, usage = self.run_with_retries(prompt, data)
                
                usage_total["inp_t"] += usage["inp_t"]
                usage_total["out_t"] += usage["out_t"]
                
                if parsed is None:
                    continue

                # ensure types
                if self.numeric:
                    parsed = [float(x) for x in parsed]
                else:
                    parsed = [str(x) for x in parsed]
                all_parsed.extend(parsed)

            if len(all_parsed) < len(component):
                continue

            row = {
                "user_idx": user_idx,
                "session_idx": session_idx,
                "phase_idx": phase_idx,
                "from_id": from_id,
                "to_id": to_id,
                "from_idx": start_pos,
                "to_idx": end_pos,
                "selected_macro_action": action_name
            }
            for dim, val in zip(component, all_parsed):
                name = dim["Variable Name"]
                row[str(name)] = val

            rows.append(row)
            last_state = all_parsed

        return rows, usage_total