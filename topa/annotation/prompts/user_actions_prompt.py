def annotate_user_actions_prompt(domain: str, system: str, user: str, type: str, input_content):
    prompt = f"""You are an expert assistant helping annotate {domain} dialogue transcripts.

You will be **looping through the utterances of a {domain} {type}**, annotating {user} utterances in batches.

You will be given:
1. **List of actions** — each with:
- name: Concise, descriptive name.
- description: A short description of the action.
2. **The last few utterances** from a {domain} {type} (context), in chronological order, labeled with speaker and unique IDs.
3. **A batch of new utterances** (also chronological, labeled with speaker and unique IDs).
- {system.capitalize()} utterances are for context only and must not be annotated.
- {user.capitalize()} utterances are the ones you must annotate.

Your task:
For each {user} utterance in the batch:
1. Select **exactly ONE** action from the provided list that best matches the {user}’s utterance.
- If none of the actions are appropriate, label it with `"None"`.
2. Provide a **confidence score** between 0.0 and 1.0 based on how clearly the {user}'s intent matches the action, and whether alternative actions were plausible.

Guidelines:
- If no suitable action exists, use `"None"` instead of forcing a match.
- Do not invent actions that are not provided in the list.
- You must output one result for each {user} utterance in the batch.

Input format:
{{
"actions": [
    {{
    "name": "<string>",
    "description": "<string>"
    }},
    ...
],
"last_utterances": [
    {{"utterance_id": "{user}_X", "speaker": "{user}", "utterance": "<string>", "selected_action": "<string>"}},
    {{"utterance_id": "{system}_X", "speaker": "{system}", "utterance": "<string>"}},
    ...
],
"batch_utterances": [
    {{"utterance_id": "{user}_X", "speaker": "{user}", "utterance": "<string>"}},
    {{"utterance_id": "{system}_X", "speaker": "{system}", "utterance": "<string>"}},
    ...
]
}}

Output format:
[
{{
    "utterance_id": "<string: {user}_X>",
    "selected_action": "<string: name of action>",
    "confidence_score": <float between 0.0 and 1.0>
}},
...
]

Input:
{input_content}

Now return ONLY the JSON output, no explanations.
"""
    return prompt