
macro_action_definition="""
high-level dialogue phases, strategies or interventions performed by the {system} toward the {user}.
"""

macro_action_description = """ 
    Each macro action must:
        - Represent a **phase of dialogue interaction**.  
        - Have a **clearly defined communicative goal** and play a distinct role in the dialogue process.  
        - Be one of the **most important and central strategies** for structuring the dialogue.
    The phases should be distinct and do not overlap in description nor goals. 

    Each macro action includes:
    - `name`: concise, descriptive title of the dialogue phase.  
    - `description`: short explanation of what the dialogue strategy is. 
    - `states` : a list of descriptions of all conversation states or {user} states where this macro action should be taken by the {system}.
    - `goal`: the intended dialogue outcome of this phase (will be used for reward).  
    - `confidence_score`: confidence in this extraction (float between 0.0 and 1.0). """ 

macro_action_format=""" 
    {
        "macro_actions": [
            {
                "name": "<string: macro action name>",
                "description": "<string: short description of the macro action>",
                "goal": {
                    "objective": "<string: primary intended outcome of this macro action>",
                    "success_levels": {
                        "ideal": "<string: description of full success — the optimal desired outcome>",
                        "acceptable": "<string: description of partial or satisfactory success>",
                        "unsuccessful": "<string: description of failure or undesired outcome>"
                    },
                    "reward_hint": "<string: qualitative indicator or signal that can be used to assign a reward>"
                },
                "states":"<list: a list of descriptions of states where this macro action should be taken>",
                "confidence_score": <float: between 0.0 and 1.0, confidence in this macro action>
            },
            ...
        ]
    }

"""

macro_action_format_str="""
    macro_actions:
    - name: "..."
      goal: 
        - objective: "..."
          success_levels: 
            - ideal: "..."
              acceptable: "..."
              unsuccessful: "..."
          reward_hint: "..."
      description: "..."
      states: 
        - "..."
      confidence_score:"..."
"""

micro_action_definition="""
micro-intervention actions that operationalize the high-level strategies and that must be executed by the {system} at the level of each utterance (interaction step) during the conversation with the {user}.
"""

micro_action_description=""" 
    - Directly actionable at the utterance level
    - Can guide a dialogue model to generate appropriate {system} utterances
    - Each micro-action must include:
        - "name"
        - "description" (how the {system} performs it)
        - "states" : a list of descriptions of all conversation states or {user} states where this micro action should be taken by the {system}.
        - "confidence_score" <float, range 0.0–1.0> — Indicates the confidence in both the accuracy of the extracted element and its appropriateness or relevance within the given context.

"""

micro_action_format="""
    [
    {{
        "name": "<string: macro action name>",
        "description": "<string: short description of the macro action>",
        "goal": {
            "objective": "<string: primary intended outcome of this macro action>",
            "success_levels": {
                "ideal": "<string: description of full success — the optimal desired outcome>",
                "acceptable": "<string: description of partial or satisfactory success>",
                "unsuccessful": "<string: description of failure or undesired outcome>"
            },
            "reward_hint": "<string: qualitative indicator or signal that can be used to assign a reward>"
        },
        "states":"<list: a list of descriptions of states where this macro action should be taken (macro state)>",
        "confidence_score": <float between 0.0 and 1.0>,
        "micro_actions": [
        {{
            "name": "<micro action name>",
            "description": "<description of {system} utterance>",
            "states":"<list: a list of descriptions of states where this micro action should be taken (micro state)>",
            "confidence_score": <float between 0.0 and 1.0>
        }},
        ...
        ]
    }},
    ...
    ]
"""

micro_action_format_str="""
    macro_actions:
    - name: "..."
      goal: 
        - objective: "..."
          success_levels: 
            - ideal: "..."
              acceptable: "..."
              unsuccessful: "..."
          reward_hint: "..."
      description: "..."
      states: 
        - "..."
      confidence_score:"..."
      micro_actions:
        - name: "..."
          description: "..."
          states:"..."
          confidence_score:"..."
"""

conversation_states_definition="""
variables, dimensions (features) that a {system} should continuously track during the {interaction_unit} that are explicitly or implicitly relevant to {domain} dialogue management and decision-making.
"""

conversation_states_description=""" 

    Each variable must have:
        - Variable Name: A short descriptive label  
        - Description: What it means and why it matters for guiding the {interaction_unit}  
        - Categorical values: List of possible values.
        - Numerical values: regression from 0 to 1 scale"
"""

conversation_states_format=""" 
        [
        {{
            "Variable Name": "...",
            "Description": "...",
            "Categorical values": [list of options],
            "Numerical values":" regression from 0 to 1 scale "
        }},
        ...
        ]
"""


conversation_states_format_str="""
    - Variable Name: "..."
      Description: "..."
      Categorical values: [list of options]
      Numerical values: "..."
"""

knowledge_graph_definition="""
    long-term memory dimensions that should be tracked across multiple {interaction_unit}s about the {user}
""" 

knowledge_graph_description="""
   - Each concept is a **node Type**.  
   - Connect related nodes with meaningful edges.  
   - Group all nodes under a root node "Patient".  
"""

knowledge_graph_format="""
    {{
    "nodes": [
        {{"type": "...", "description": ...}},
    ],
    "edges": [
        {{"source": "...", "target": "...", "relation": "..."}},
    ]
    }}

"""

knowledge_graph_format_str="""
    nodes: [["type": "...", "description": ...],...]
    edges: [["source": "...", "target": "...", "relation": "..."],...]
"""

cautions_definition="""
    Warnings or risks that describe what the {system} should *not* do during a {domain} {interaction_unit} with the {user}.

"""

cautions_description="""
        Each caution includes:
        - `name`: concise negative action name.
        - `description`: short description of what the caution is.
        - `risk`: the negative outcome why this behavior is discouraged.
        - `confidence_score`: float between 0.0 and 1.0.

"""

cautions_format= """
        [
            {
                "negative_action": concise negative action name,
                "description": "<string: short description of the caution>",
                "risk": "<string: potential negative outcome if this action is taken>",
                "confidence_score": <float: between 0.0 and 1.0>
            },
            ...
        ]
"""

cautions_format_str="""
    - negative_action: concise negative action name,
      description: short description of the caution,
      risk: potential negative outcome if this action is taken,
      confidence_score: float: between 0.0 and 1.0

"""


rules_definition="""
    Atomic, implementation-ready rules. Each rule MUST be an object with EXACTLY these two keys:
    - "if": a concise description of the situation, conversation state, or {user} state.
    - "then": a concise description of what the {system} should do next (action, constraint, or response strategy)
"""

rules_format="""
    [
    {{"if": "xxx", "then": "xxx"}},
    {{"if": "xxx", "then": "xxx"}}
    ]
"""
rules_format_str="""
    ["if": "xxx", "then": "xxx",...]

"""

user_action_definition= """
    distinct types of **{user} verbal actions** that occur in response to {system} inputs covering every possible utterance level action.
"""

user_action_format="""

        [ {{"Action Name": "short label", "Description": "what the action means" }},
        ...
        ]

"""


user_profile_definition="""
    stable {user} attributes that shape how the {user} typically thinks, feels, behaves, and engages in {domain} {interaction_unit} with the {system} that help simulate realistic responses.

"""

user_profile_description = """
    The user profile is composed of TWO DISTINCT TYPES of dimensions:

    1. Categorical Dimensions:
    - Dimensions whose values must be selected from a predefined set of categories.
    - Used for structured reasoning, routing, statistics, and policy decisions.

    2. Free-form Dimensions:
    - Dimensions whose values are expressed as open-ended natural language text.
    - Used for contextual understanding, explanation, personalization, and memory.

    Each dimension belongs to EXACTLY ONE type: categorical OR free-form.
"""

user_profile_format = """
    {
    "categorical_dimensions": [
        {
        "dimension_name": "...",
        "description": "...",
        "options": ["option_1", "option_2", "option_3"],
        }
    ],
    "free_form_dimensions": [
        {
        "dimension_name": "...",
        "description": "...",
        "value": "Open-ended natural language text"
        }
    ]
    }
"""

user_profile_format_str = """
    Categorical Dimensions:
    - Dimension Name: "..."
    Description: "..."
    Options: [option_1, option_2, option_3]

    Free-form Dimensions:
    - Dimension Name: "..."
    Description: "..."
    Value: "Open-ended text"
"""