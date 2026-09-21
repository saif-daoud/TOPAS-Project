def annotate_macro_actions_prompt(domain: str, system: str, user: str, interaction_unit: str, input_content):
    prompt= f"""You are an expert assistant helping annotate {domain} dialogue transcripts.
You will be **looping through the utterances of a {domain} {interaction_unit}**, annotating {system} utterances.

You will be given:
1. **List of macro actions** — each with:
- name: Concise, descriptive name.
- description: A short explanation of the macro action.
- goal: the goal for the macro action.
2. **A full {interaction_unit} transcript**, in chronological order, labeled with speaker and unique IDs.
- {user.capitalize()} utterances are for context only and must not be annotated.
- {system.capitalize()} utterances are the ones you must annotate.

Your task:
1. Organize {system} utterances into **phases**. A phase is a continuous sequence of utterances assigned to the same macro action. 
2. Each phase starts with the first {system} utterance where that macro action begins and ends with the last utterance before its goal has been achieved.
3. For each phase, output:
- `"from"`: the utterance_id of the first {system} utterance in the phase.
- `"to"`: the utterance_id of the last {system} utterance in the phase (must satisfy `"to" >= "from"`).
- `"selected_macro_action"`: the macro action name.
- `"confidence_score"`: an object containing **four sub-scores**.
     You must output the following sub-scores (each between 0.0 and 1.0):
       - `"intent_alignment"`: how clearly the {system}'s intent matches the macro action.
       - `"goal_fit"`: how well the {user}’s utterances move toward achieving the macro action’s stated goal. High = the utterances meaningfully contribute to completing the goal.
       - `"alternative_actions_possible"`: whether other macro actions would have been equally or more appropriate for the same utterances. High = low ambiguity. Low = the utterances could reasonably belong to multiple macro actions.
       - `"utterance_coherence"`: how coherent, consistent, and logically connected the {user}’s utterances are within the phase. High = the utterances form a clear, internally consistent sequence.

Guidelines:
- Focus on the {system}’s **intent and purpose**.
- Do not invent macro actions not provided in the list.
- You must output **a continuous sequence of phases that fully covers all {system} utterances in the {interaction_unit}** (no skipping).
- Ensure that in every phase, `"to"` is greater than or equal to `"from"`.
- If no macro action fits with ≥ 0.6 average confidence score, label the phase as "uncertain".

Input format:
{{
"macro_actions": [
    {{
    "name": "<string>",
    "description": "<string>",
    "goal": "<string>"
    }},
    ...
],
"session_utterances": [
    {{"utterance_id": "{user}_X", "speaker": "{user}", "utterance": "<string>"}},
    {{"utterance_id": "{system}_X", "speaker": "{system}", "utterance": "<string>"}},
    ...
]
}}

Output format:
[
{{
    "from": "<string: first {system}_X in phase>",
    "to": "<string: last {system}_X in phase>",
    "selected_macro_action": "<string: name of macro action>",
    "confidence_score": {{
        "intent_alignment": <float between 0.0 and 1.0>,
        "goal_fit": <float between 0.0 and 1.0>,
        "alternative_actions_possible": <float between 0.0 and 1.0>,
        "utterance_coherence": <float between 0.0 and 1.0>
    }}
}},
...
]

Input:
{input_content}

Now return ONLY the JSON output, no explanations.
"""
    return prompt

def annotate_macro_actions_with_rules_prompt(domain: str, system: str, user: str, interaction_unit: str, input_content):
    prompt = f"""You are an expert assistant helping annotate {domain} dialogue transcripts.
You will be **looping through the utterances of a {domain} {interaction_unit}**, annotating {system} utterances.

You will be given:
1. **List of macro actions** — each with:
- name: Concise, descriptive name.
- description: A short explanation of the macro action.
- goal: The goal for the macro action.
- states: The **annotation rules** that specify **when** this macro action should apply.
2. **A full {interaction_unit} transcript**, in chronological order, labeled with speaker and unique IDs.
- {user.capitalize()} utterances are for context only and must not be annotated.
- {system.capitalize()} utterances are the ones you must annotate.

Your task:
1. Organize {system} utterances into **phases**. A phase is a continuous sequence of utterances assigned to the same macro action.
2. Each phase starts with the first {system} utterance where the chosen macro action begins working toward its goal, and ends with the last utterance before its goal has been achieved.
3. For each phase, output:
- `"from"`: the utterance_id of the first {system} utterance in the phase.
- `"to"`: the utterance_id of the last {system} utterance in the phase (must satisfy `"to" >= "from"`).
- `"selected_macro_action"`: the macro action name.
- `"annotation_rule_used"`: the exact annotation rule (from "states") that you used to **resolve ambiguity** or determine a **boundary** when ambiguous; if no rule was used, set this to null.
- `"confidence_score"`: an object containing **four sub-scores**.  
     You must output the following sub-scores (each between 0.0 and 1.0):
       - `"intent_alignment"`: how clearly the {system}'s intent matches the macro action.
       - `"goal_fit"`: how well the {user}’s utterances move toward achieving the macro action’s stated goal. High = the utterances meaningfully contribute to completing the goal.
       - `"alternative_actions_possible"`: whether other macro actions would have been equally or more appropriate for the same utterances. High = low ambiguity. Low = the utterances could reasonably belong to multiple macro actions.
       - `"utterance_coherence"`: how coherent, consistent, and logically connected the {user}’s utterances are within the phase. High = the utterances form a clear, internally consistent sequence.

When to use annotation rules:
- Choose the macro action by matching the {system}’s **intent/purpose** to the macro action’s **name / description / goal**.
- **Only if ambiguous:** If two or more macro actions remain plausible, then consult their **states** (annotation rules) as **tie-breakers**. If you used a specific rule to resolve the ambiguity, set "annotation_rule_used" to that rule’s exact text.
- **Do not use** annotation rules for selection when there is no ambiguity.

Guidelines:
- Do not invent macro actions not provided in the list.
- Focus on the {system}’s **intent and purpose**.
- You must output **a continuous sequence of phases that fully covers all {system} utterances** in the {interaction_unit} (no skipping).
- Ensure that in every phase, "to" is greater than or equal to "from".
- If no macro action fits with ≥ 0.6 average confidence score, label the phase as "uncertain".

Input format:
{{
"macro_actions": [
    {{
    "name": "<string>",
    "description": "<string>",
    "goal": "<string>",
    "states": "<list>"
    }},
    ...
],
"session_utterances": [
    {{"utterance_id": "{user}_X", "speaker": "{user}", "utterance": "<string>"}},
    {{"utterance_id": "{system}_X", "speaker": "{system}", "utterance": "<string>"}},
    ...
]
}}

Output format:
[
{{
    "from": "<string: first {system}_X in phase>",
    "to": "<string: last {system}_X in phase>",
    "selected_macro_action": "<string: name of macro action>",
    "annotation_rule_used": None or <string>,
    "confidence_score": {{
        "intent_alignment": <float between 0.0 and 1.0>,
        "goal_fit": <float between 0.0 and 1.0>,
        "alternative_actions_possible": <float between 0.0 and 1.0>,
        "utterance_coherence": <float between 0.0 and 1.0>
    }}
}},
...
]

Input:
{input_content}

Now return ONLY the JSON output, no explanations.
"""
    return prompt