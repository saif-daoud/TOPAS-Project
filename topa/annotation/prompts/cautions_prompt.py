def annotate_cautions_prompt(domain: str, system: str, user: str, interaction_unit: str, input_content: str):
    prompt = f"""You are an expert assistant annotating {domain} {interaction_unit} transcripts.

You will be given:
1) A list of **cautions** (things the {system} should avoid). Each item has:
   - negative_action: concise name of the caution.
   - description: a short explanation of the caution.
   - risk: why it's harmful.
2) **The last few utterances** from a {domain} {interaction_unit} (context), in chronological order, labeled with speaker and unique IDs.
3) **A batch of new utterances** (also chronological, labeled with speaker and unique IDs).
   - {user.capitalize()} utterances are for context only and must not be annotated.
   - {system.capitalize()} utterances are the ones you must annotate.

Your task (for each {system} utterance in the batch):
- Select **exactly ONE** `negative_action` from the provided cautions **or** `"None"` if the {system}’s utterance does not exhibit any caution.
- Provide a **confidence score** between 0.0 and 1.0 based on how clearly the {system}'s intent matches the caution.

Rules:
- Focus on the {system}’s **intent and purpose**, considering the prior utterances for context.
- Use **only** the `negative_action` values provided in the `cautions` list, or `"None"`.
- Do not invent cautions that are not provided in the list.
- Use the **preceding context** and the **current batch** to infer the {system}’s **intent and purpose**; some cautions require context to identify.
- **Be precise**: assign a caution **only** when the utterance itself (considering context) **explicitly or strongly** exhibits the behavior described.
- If evidence is **ambiguous, borderline, or weak**, choose **"None"** rather than over-labeling.
- Output **only JSON**, no explanations.

Input:
{input_content}

Output format:
[
  {{
    "utterance_id": "{system}_X",
    "selected_caution": "<exact negative_action from cautions or 'None'>",
    "confidence_score": 0.0-1.0
  }},
  ...
]
"""
    return prompt
