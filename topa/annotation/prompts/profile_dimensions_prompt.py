def annotate_user_profile_dim_prompt(domain: str, system: str, user: str, interaction_unit: str, input_content):
    prompt = f"""You are an expert assistant helping annotate {domain} dialogue transcripts.

Goal:
Infer {user} **profile dimension values** from the provided evidence and return a **single JSON object**
that matches the **same schema as the extracted user_profile component**.

You will be given:
1) "categorical_dimensions": a list of dimension objects, each with:
   - "dimension_name": concise name of the profile component
   - "description": a short description of the profile dimension
   - "options": list of allowed categorical values
2) "free_form_dimensions": a list of dimension objects, each with:
   - "dimension_name": concise name of the profile component
   - "description": a short description of the profile dimension
3) "session_script": the full transcript (with speaker labels)

Task:
For **every** dimension:
- If it is categorical, fill its "value" with **exactly one** option from "options".
  If there is **insufficient evidence**, set "value" to "Unknown".
- If it is free-form, fill its "value" with a short natural language summary grounded in the transcript.
  If there is **insufficient evidence**, set "value" to "Unknown".

Guidelines:
- Include **every** input dimension exactly once.
- Do **not** add extra keys or fields (no "notes", "confidence", etc.).
- Do **not** invent new categorical options.
- Output **only** the JSON object, no prose.

Input format:
{{
  "categorical_dimensions": [
    {{
      "dimension_name": "<string>",
      "description": "<string>",
      "options": ["<string>", "..."]
    }},
    ...
  ],
  "free_form_dimensions": [
    {{
      "dimension_name": "<string>",
      "description": "<string>"
    }},
    ...
  ],
  "session_script": "<string>"
}}

Output format:
{{
  "categorical_dimensions": [
    {{
      "dimension_name": "<string>",
      "value": "<one of options | Unknown>"
    }},
    ...
  ],
  "free_form_dimensions": [
    {{
      "dimension_name": "<string>",
      "value": "<free-form text | Unknown>"
    }},
    ...
  ]
}}

Input:
{input_content}

Now return ONLY the JSON object, no explanations."""
    return prompt