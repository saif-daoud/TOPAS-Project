def annotate_intrinsic_reward_prompt(domain, system, user, interaction_unit, goal, last_utterances, new_utterances):
    output_block = f"""
{{
    "label": <integer 0-2>,
    "confidence_score": <float between 0.0 and 1.0>
}},
"""

    prompt = f"""You are an expert {domain} assistant evaluating {system} utterances in {domain} {interaction_unit}s.

Your task: Assign ONE integer **progress score** from 0 to 2 representing how far we have progressed toward the goal, using all available context.
You will be given the goal, the success levels ("ideal": description of full success — the optimal desired outcome, "acceptable": description of partial or satisfactory success, "unsuccessful": description of failure or undesired outcome), 
and reward_hint which is a qualitative indicator or signal that can be used to assign a reward.

### Goal
{goal["objective"]}

### Scoring Rubric (0–2)
- **0 — unsuccessful**: {goal["success_levels"]["unsuccessful"]}.
- **1 — acceptable**: {goal["success_levels"]["acceptable"]}.
- **2 — ideal**: {goal["success_levels"]["ideal"]}.

### Reward Hint
{goal["reward_hint"]}

### Instructions
- Use **all prior utterances** (`last_utterances`) as context.
- Always return a **JSON array**, in this format:
{output_block}

### Dialogue Context (previous utterances)
{last_utterances}

### New Utterances (to evaluate as a whole)
{new_utterances}

### Answer:
"""
    return prompt