def annotate_micro_actions_prompt(domain: str, system: str, user: str, interaction_unit: str, input_content):
    prompt=f"""You are an expert assistant helping annotate {domain} dialogue transcripts.

You will be **looping through the utterances of a {domain} {interaction_unit}**, annotating {system} utterances in batches.

You will be given:
1. **The selected macro action for this batch**:
- name: Concise, descriptive name.
- description: A short explanation of the macro action.
- goal: the goal for the macro action.
- All utterances in this batch are labeled to this macro action (a high-level strategy).
2. **List of micro actions** — each with:
- name: Concise, descriptive name.
- description: A short explanation of the micro action.
3. **The last few utterances** from a {domain} {interaction_unit} (context), in chronological order, labeled with speaker and unique IDs.
4. **A batch of new utterances** (also chronological, labeled with speaker and unique IDs).
- {user.capitalize()} utterances are for context only and must not be annotated.
- {system.capitalize()} utterances are the ones you must annotate.

Your task:
For each {system} utterance in the batch:
1. Select **exactly ONE** micro action from the provided list that best matches the {system}’s utterance.
- If none of the micro actions are appropriate, label it with `"None"`.
2. Provide a **confidence score** between 0.0 and 1.0 based on how clearly the {system}'s intent matches the micro action, and whether alternative micro actions were plausible.

Guidelines:
- Focus on the {system}’s **intent and purpose**, considering the prior utterances for context.
- If no suitable micro action exists, use `"None"` instead of forcing a match.
- Do not invent micro actions that are not provided in the list.
- You must output one result for each {system} utterance in the batch.

Input format:
{{
"macro_action": {{
    "name": "<string>",
    "description": "<string>",
    "goal": "<string>"
}},
"micro_actions": [
    {{
    "name": "<string>",
    "description": "<string>"
    }},
    ...
],
"last_utterances": [
    {{"utterance_id": "{system}_X", "speaker": "{system}", "utterance": "<string>"}},
    {{"utterance_id": "{user}_X", "speaker": "{user}", "utterance": "<string>"}},
    ...
],
"batch_utterances": [
    {{"utterance_id": "{system}_X", "speaker": "{system}", "utterance": "<string>"}},
    {{"utterance_id": "{user}_X", "speaker": "{user}", "utterance": "<string>"}},
    ...
]
}}

Output format:
[
{{
    "utterance_id": "<string: {system}_X>",
    "selected_micro_action": "<string: name of micro action>",
    "confidence_score": <float between 0.0 and 1.0>
}},
...
]

Input:
{input_content}

Now return ONLY the JSON output, no explanations.
"""
    return prompt

def annotate_micro_actions_with_rules_prompt(domain: str, system: str, user: str, interaction_unit: str, input_content):
    prompt=f"""You are an expert assistant helping annotate {domain} dialogue transcripts.

You will be **looping through the utterances of a {domain} {interaction_unit}**, annotating {system} utterances in batches.

You will be given:
1. **The selected macro action for this batch**:
- name: Concise, descriptive name.
- description: A short explanation of the macro action.
- goal: the goal for the macro action.
- All utterances in this batch are labeled to this macro action (a high-level strategy).
2. **List of micro actions** — each with:
- name: Concise, descriptive name.
- description: A short explanation of the micro action.
- states: The **annotation rules** that indicate **when** this micro action should apply.
3. **The last few utterances** from a {domain} {interaction_unit} (context), in chronological order, labeled with speaker and unique IDs.
4. **A batch of new utterances** (also chronological, labeled with speaker and unique IDs).
- {user.capitalize()} utterances are for context only and must not be annotated.
- {system.capitalize()} utterances are the ones you must annotate.

Your task:
For each {system} utterance in the batch:
1. Select **exactly ONE** micro action from the provided list that best matches the {system}’s utterance.
- If none of the micro actions are appropriate, label it with `"None"`.
2. Provide a **confidence score** between 0.0 and 1.0 based on how clearly the {system}'s intent matches the micro action, and whether alternative micro actions were plausible.

When to use annotation rules:
- Choose the micro action by matching the {system}’s **intent/purpose** to the micro action’s **name / description**.
- **Only if ambiguous:** If two or more micro actions remain plausible, then consult their **states** (annotation rules) as **tie-breakers**. If you used a specific rule to resolve the ambiguity, set "annotation_rule_used" to that rule’s exact text.
- **Do not use** annotation rules for selection when there is no ambiguity, and set annotation_rule_used this to null.

Guidelines:
- Focus on the {system}’s **intent and purpose**, considering the prior utterances for context.
- If no suitable micro action exists, use `"None"` instead of forcing a match.
- Do not invent micro actions that are not provided in the list.
- You must output one result for each {system} utterance in the batch.

Input format:
{{
"macro_action": {{
    "name": "<string>",
    "description": "<string>",
    "goal": "<string>"
}},
"micro_actions": [
    {{
    "name": "<string>",
    "description": "<string>",
    "states": "<string>"
    }},
    ...
],
"last_utterances": [
    {{"utterance_id": "{system}_X", "speaker": "{system}", "utterance": "<string>"}},
    {{"utterance_id": "{user}_X", "speaker": "{user}", "utterance": "<string>"}},
    ...
],
"batch_utterances": [
    {{"utterance_id": "{system}_X", "speaker": "{system}", "utterance": "<string>"}},
    {{"utterance_id": "{user}_X", "speaker": "{user}", "utterance": "<string>"}},
    ...
]
}}

Output format:
[
{{
    "utterance_id": "<string: {system}_X>",
    "selected_micro_action": "<string: name of micro action>",
    "annotation_rule_used": None or <string>,
    "confidence_score": <float between 0.0 and 1.0>
}},
...
]

Input:
{input_content}

Now return ONLY the JSON output, no explanations.
"""
    return prompt