from typing import Optional
from copy import deepcopy
import pandas as pd

from .base import BaseAnnotator
from ..prompts import annotate_intrinsic_reward_prompt
from ..utils import format_utterances

from ...utils import build_logger
logger = build_logger()

class IntrinsicRewardAnnotator(BaseAnnotator):
    def __init__(self, **kwargs):
        _kwargs = deepcopy(kwargs)
        context_length = _kwargs.pop("context_length")
        super().__init__(**_kwargs)

        self.context_length = int(context_length)
        self._macro_df: Optional[pd.DataFrame] = None

    def build_prompt(self, input_content: dict) -> str:
        return annotate_intrinsic_reward_prompt(
            domain=self.domain_params["adj"],
            user=self.domain_params["user"],
            system=self.domain_params["system"],
            interaction_unit=self.domain_params["interaction_unit"],
            goal=input_content["goal"],
            last_utterances=input_content["last_utterances"],
            new_utterances=input_content["new_utterances"]
        )
    
    def sanity_check(self, outputs, *args):
        if not isinstance(outputs, dict):
            return False
        lab = outputs.get("label")
        if not isinstance(lab, int) or lab < 0 or lab > 2:
            return False
        if "confidence_score" not in outputs:
            return False
        return True

    def postprocess(self, outputs, data):
        if isinstance(outputs, list):
            if len(outputs) > 1:
                logger.info(f"[Warning] output length is large than 1 {outputs} !")
            return outputs[0]
        return outputs

    def fallback(self, last_outputs, data):
        logger.warning("[FALLBACK] Returning unsuccessful.")
        return {"label": 0, "confidence_score": 1.0}

    def annotate_single_session(self, user_idx, session_idx, session, session_metadata, component, annotated_data):
        mask = (annotated_data["user_idx"] == user_idx) & (annotated_data["session_idx"] == session_idx)
        phases = annotated_data.loc[mask].copy()
        from_col = "from_idx" if "from_idx" in phases.columns else "from_id"
        phases = phases.sort_values(by=[from_col])

        macro_by_name = {ma["name"]: ma for ma in component}

        utterances, system_counter, user_counter = self.process_dialogue(session)
        sys_pos = {
            u["utterance_id"]: i for i, u in enumerate(utterances) 
            if u["speaker"] == self.domain_params["system"]
        }
        
        sys_ids_by_pos = [
            (i, u["utterance_id"]) for i, u in enumerate(utterances)
            if u["speaker"] == self.domain_params["system"]
        ]

        rows = []
        usage_total = {"inp_t": 0.0, "out_t": 0.0}

        for phase_idx, ph in phases.iterrows():
            action_name = ph["selected_macro_action"]
            if action_name == "uncertain":
                continue
            macro_def = macro_by_name[action_name]
            
            if isinstance(ph["from_id"], str) and isinstance(ph["to_id"], str):
                from_id = ph["from_id"]
                to_id = ph["to_id"]
            else:
                from_id = f'{self.domain_params["system"]}_{int(ph["from_id"])}'
                to_id = f'{self.domain_params["system"]}_{int(ph["to_id"])}'

            start_pos = sys_pos[from_id]
            end_pos = sys_pos[to_id]
            last_ctx = utterances[max(0, start_pos - self.context_length):start_pos]
            new_utterances = utterances[start_pos:end_pos + 1]

            input_content = {
                "goal": macro_def["goal"],
                "last_utterances": format_utterances(last_ctx),
                "new_utterances": format_utterances(new_utterances)
            }
            prompt = self.build_prompt(input_content)
            parsed, usage = self.run_with_retries(prompt, {})
            
            usage_total["inp_t"] += usage["inp_t"]
            usage_total["out_t"] += usage["out_t"]
            
            if parsed is None:
                continue

            rows.append(
                {
                    "user_idx": user_idx,
                    "session_idx": session_idx,
                    "phase_idx": phase_idx,
                    "from_id": from_id,
                    "to_id": to_id,
                    "from_idx": start_pos,
                    "to_idx": end_pos,
                    "selected_macro_action": action_name,
                    "reward": parsed["label"],
                    "confidence_score": parsed["confidence_score"]
                }
            )
        return rows, usage_total