from typing import Dict, List
import json

def build_refine_conversation_state_prompt(domain: str, macro_action: Dict, existing_dimensions: List[Dict]) -> str:
    """
    Prompt: given ONE macro action (+ its micro actions) and current conversation-state dimensions,
    return ONLY *new* dimensions needed to cover missing state concepts.
    """
    return f"""
You are helping design a proactive agent in the domain: {domain}.

Your task is to refine the agent's **conversation_states** (dimensions the agent should track).
You will be given:
1) EXISTING_DIMENSIONS_JSON: the current conversation_states list.
2) MACRO_ACTION_JSON: ONE macro action, including its micro_actions.

What to do:
- Look at MACRO_ACTION_JSON.states (macro states) AND each micro_actions[].states (micro states).
- Check whether these state concepts are already covered by EXISTING_DIMENSIONS_JSON.
- If anything is missing, propose NEW conversation-state dimensions (variables) that would allow tracking them.

CRITICAL constraints (avoid duplicates):
- DO NOT output any dimension that already exists.
- DO NOT output any new dimension that is semantically similar to an existing one.
- If unsure, prefer NOT adding a dimension.

Coverage guidance:
- A state is "covered" ONLY if EXISTING_DIMENSIONS_JSON contains ALL the necessary dimensions that would let the agent reliably detect/monitor a policy for that state.
- If some needed dimensions exist but others are missing, include existing ones in by_dimensions, and explain the missing part in gap_reason.
- If none exist, by_dimensions should be [], and gap_reason should explain what information is missing.
- If fully covered: gap_reason should be "".

State representation:
- Treat MACRO_ACTION_JSON.states as a LIST where each item is already ONE atomic state.
- Treat each micro_actions[].states as a LIST where each item is already ONE atomic state.

Output format:
Return ONLY a valid JSON object matching EXACTLY the schema below.

OUTPUT JSON SCHEMA
{{
  "per_action": [
    {{
      "action": "string exact action name, whether it is macro or micro action",
      "states_mappings": [
        {{
          "state_phrase": "string exact state description",
          "by_dimensions": ["Variable Name", "..."],
          "gap_reason": "string (what's missing to be sufficient?)"
        }}
      ],
    }}
  ],
  "new_dimensions": [
    {{
      "Variable Name": "string",
      "Description": "string",
      "Categorical values": [list of options],
      "Numerical values": " regression from 0 to 1 scale "
      "Rationale": "string"
    }},
  ]
}}

Steps:
1) Create per_action entries for:
   - the macro action itself: action = MACRO_ACTION_JSON.name
   - each micro action: action = micro_action.name (exact)
2) For each state item in the macro states list and micro states lists:
   - Identify the COMPLETE minimal set of dimensions needed to cover it.
   - Populate by_dimensions with the Variable Name(s) that already exist and are required.
   - If missing dimensions are required, fill gap_reason with what is missing.
   - If fully covered:  gap_reason="".
3) In new_dimensions:
   - Define ONLY the new dimensions that are truly needed (union across all states_mappings).
   - Each must include Variable Name, Description, Values, and a short Rationale.

EXISTING_DIMENSIONS_JSON:
```json
{json.dumps(existing_dimensions, ensure_ascii=False)}
```

MACRO_ACTION_JSON:
```json
{json.dumps(macro_action, ensure_ascii=False)}
```
"""

def build_refine_action_space_prompt(domain_params: dict, macro_action: Dict, examples: List[Dict], max_new_actions: int) -> str:
    """Ask the LLM to propose missing micro actions and relabel candidates."""
    system = domain_params["system"]
    user = domain_params["user"]
    domain = domain_params["adj"]

    micro_actions = macro_action["micro_actions"]
    micro_actions_min = [
        {
            "name": m["name"],
            "description": m["description"],
            "states": m["states"],
        }
        for m in micro_actions
    ]

    payload = {
        "macro_action": {
            "name": macro_action["name"],
            "description": macro_action["description"],
            "goal": macro_action["goal"]["objective"],
            "states": macro_action["states"]
        },
        "existing_micro_actions": micro_actions_min,
        "examples": examples,
        "max_new_actions": max_new_actions
    }

    return f"""You are refining the micro action space for a {domain} dialogue system.

You will be given:
1) One macro action (high-level strategy).
2) The current list of existing micro actions for that macro action.
3) A list of {system} utterances that were annotated with low confidence or 'None'. Each example includes a short preceding context.

Your tasks:
A) Propose up to {max_new_actions} NEW micro actions that are missing, if needed.
   - Only propose a new micro action if several examples truly don't fit existing micro actions.
   - Each new micro action must have: name, description, a short reason, a list of states, and a confidence score.
   - "states" is a list of descriptions of all conversation states or {user} states where this micro action should be taken by the {system}.
   - "confidence_score" <float, range 0.0–1.0> which indicates the confidence in both the accuracy of the new micro action and its appropriateness or relevance within the given context.
B) For each example, assign the best micro action name among (existing + new) OR keep "None".
   - Provide a confidence_score between 0.0 and 1.0 based on how clearly the {system}'s intent matches the micro action, and whether alternative micro actions were plausible.

Rules:
- Use only JSON output.
- Do not rename existing micro actions.
- Prefer mapping to an existing micro action when reasonable.
- The value of "selected_micro_action" MUST be EXACTLY ONE of:
  1) one of the strings in existing_micro_actions[].name, OR
  2) one of the strings in new_micro_actions[].name (only if you proposed it), OR
  3) the literal string "None".

Input JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Output JSON format:
{{
  "new_micro_actions": [
    {{"name": "...", "description": "...", "states": ["..."], "reason": "...", "confidence_score": <float between 0.0 and 1.0>}},
    ...
  ],
  "utterance_to_action": [
    {{
      "row_key": "<copy from input examples>",
      "selected_micro_action": "<existing name or new name or 'None'>",
      "confidence_score": <float between 0.0 and 1.0>,
      "reason": "..."
    }},
    ...
  ]
}}
"""