def annotate_conversation_state_prompt(domain, system, user, interaction_unit, macro_st_dim, last_macro_state, new_utterances, numeric):
    if numeric:
        prompt = f"""You are an expert {domain} assistant that tracks changes in {user}'s state during a {domain} {interaction_unit} using a set of structured.

### What you are given:
1) **Dimensions metadata** (ordered list):
Each dimension has:
- a name
- a definition
- a short definition of what increasing vs. decreasing the value means
Dimensions:
{macro_st_dim}
2) **Previous state vector**:
This vector contains one value per dimension, in the same order:
{last_macro_state}
3) **New dialogue utterances** ({system} + {user} turns):
{new_utterances}
### Your task:
Update the state vector by evaluating how the new utterances affect each dimension.
For each dimension:
- Increase its value if the new utterances strongly indicate an increase.
- Decrease its value if the new utterances strongly indicate a decrease.
- Leave the value unchanged if the utterances do not provide enough evidence.

### Important rules:
- Keep the SAME vector length.
- Each element must be a float between **0.0 and 1.0**.
- Changes should be **small and realistic** (e.g., 0.05–0.20) unless the dialogue contains very strong evidence.
- If evidence is ambiguous or weak → keep the previous value.
- If the dimension is not mentioned → keep the previous value.
- Base updates ONLY on the meanings of the dimensions.

### Think step-by-step (hidden from final output):
For each dimension:
1. Identify any utterance that is relevant to the dimension.
2. Decide whether this evidence suggests an increase, decrease, or no change.
3. Adjust the previous value accordingly.
DO NOT include this reasoning in the final answer.

### Output format (STRICT):
Output ONLY the final updated vector in this exact format:
[ float, float, float, ..., float ]
No text before or after.
"""
    else:
        prompt = f"""You are an expert {domain} assistant that tracks changes in {user}'s state during a {domain} {interaction_unit} using a set of structured categorical dimensions.

### What you are given:
1) **Dimensions metadata** (ordered list):
Each dimension has:
- a name
- a definition
- a list of categories
Dimensions:
{macro_st_dim}
2) **Previous state vector (categorical)**:
This vector contains one categorical value per dimension, in the same order:
{last_macro_state}
3) **New dialogue utterances** ({system} + {user} turns):
{new_utterances}

### Your task:
Update the categorical state vector by evaluating how the new utterances affect each dimension.
For each dimension:
- If the new utterances clearly indicate a shift toward a different category → update to that category.
- If multiple categories are possible, select the one most strongly supported by the utterances.
- If evidence is weak, ambiguous, or irrelevant → keep the previous category.
- Do NOT invent categories not listed in the metadata.
- Keep the same vector length.

### Think step-by-step (hidden from final output):
For each dimension:
1. Identify utterances relevant to the dimension.
2. Determine which (if any) of the allowed categories best matches the new evidence.
3. Compare with the previous category.
4. Update only if the evidence is clearly stronger for another category.
DO NOT include this reasoning in the final answer.

### Guideline (STRICT):
You must output in each dimension one category from the list in "Categorical values", do not invent new categories.
If there is no appropriate category, put none.

### Output format (STRICT):
Output ONLY the final updated vector in this exact format and length (JSON-style):
{{
    "dim_0": <string: exact value from categories list>,
    ...
    "dim_N": <string: exact value from categories list>
}}
"""
    return prompt