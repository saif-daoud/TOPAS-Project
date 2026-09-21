from typing import Dict

PART1_ITEMS = [
    {"key": "agenda", "label": "Agenda"},
    {"key": "feedback", "label": "Feedback"},
    {"key": "understanding", "label": "Understanding"},
    {"key": "interpersonal_effectiveness", "label": "Interpersonal Effectiveness"},
    {"key": "collaboration", "label": "Collaboration"},
    {"key": "pacing_time_use", "label": "Pacing and Efficient Use of Time"},
]

PART2_ITEMS = [
    {"key": "guided_discovery", "label": "Guided Discovery"},
    {"key": "focusing_on_key_cognitions_behaviors", "label": "Focusing on Key Cognitions or Behaviors"},
    {"key": "strategy_for_change", "label": "Strategy for Change"},
    {"key": "application_of_cbt_techniques", "label": "Application of Cognitive-Behavioral Techniques"},
    {"key": "homework", "label": "Homework"},
]

ALL_ITEMS = PART1_ITEMS + PART2_ITEMS

GENERAL_SCALE_BLURB = (
    "Rate each code on a 0â€“6 scale: 0=Poor, 1=Barely Adequate, 2=Mediocre, "
    "3=Satisfactory, 4=Good, 5=Very Good, 6=Excellent."
)

CODE_ANCHORS: Dict[str, Dict[int, str]] = {
    # ---- Part 1 ----
    "agenda": {
        0: "Therapist did not set agenda.",
        2: "Therapist set agenda that was vague or incomplete",
        4: "Therapist worked with patient to set a mutually satisfactory agenda that included specific target problems (e.g., anxiety at work, dissatisfaction with marriage.)",
        6: "Therapist worked with patient to set an appropriate agenda with target problems, suitable for the available time. Established priorities and then followed agenda."
    },
    "feedback": {
        0: "Therapist did not ask for feedback to determine patientâ€™s understanding of, or response to, the session.",
        2: "Therapist elicited some feedback from the patient, but did not ask enough questions to be sure the patient understood the therapistâ€™s line of reasoning during the session or to ascertain whether the patient was satisfied with the session.",
        4: "Therapist asked enough questions to be sure that the patient understood the therapist's line of reasoning throughout the session and to determine the patient's reactions to the session. The therapist adjusted his/her behavior in response to the feedback, when appropriate.",
        6: "Therapist was especially adept at eliciting and responding to verbal and nonÂ­verbal feedback throughout the session (e.g., elicited reactions to session, regularly checked for understanding, helped summarize main points at end of session."
    },
    "understanding": {
        0: "Therapist repeatedly failed to understand what the patient explicitly said and thus consistently missed the point. Poor empathic skills",
        2: "Therapist was usually able to reflect or rephrase what the patient explicitly said, but repeatedly failed to respond to more subtle communication. Limited ability to listen and empathize.",
        4: "Therapist generally seemed to grasp the patientâ€™s â€œinternal realityâ€ as reflected by both what the explicitly said and what the patient communicated in more subtle ways. Good ability to listen and empathize.",
        6: "Therapist seemed to understand the patientâ€™s â€œinternal realityâ€ thoroughly and was adept at communicating this understanding through appropriate verbal and nonÂ­verbal responses to the patient (e.g., the tone of the therapistâ€™s response conveyed a sympathetic understanding of the patientâ€™s â€œmessageâ€). Excellent listening and empathic skills"
    },
    "interpersonal_effectiveness": {
        0: "Therapist had poor interpersonal skills. Seemed hostile, demeaning, or in some other way destructive to the patient",
        2: "Therapist did not seem destructive, but had significant interpersonal problems. At times, therapist appeared unnecessarily impatient, aloof, insincere or had difficulty conveying confidence and competence.",
        4: "Therapist displayed a satisfactory degree of warmth, concern, confidence, genuineness, and professionalism. No significant interpersonal problems.",
        6: " Therapist displayed optimal levels of warmth, concern, confidence, genuineness, and professionalism, appropriate for this particular patient in this session."
    },
    "collaboration": {
        0: "Therapist did not attempt to set up a collaboration with patient.",
        2: "Therapist attempted to collaborate with patient, but had difficulty either defining a problem that the patient considered important or establishing rapport.",
        4: "Therapist was able to collaborate with patient, focus on a problem that both patient and therapist considered important, and establish rapport.",
        6: "Collaboration seemed excellentÍ¾ therapist encouraged patient as much as possible to take an active role during the session (e.g., by offering choices) so they could function as a â€œteamâ€."
    },
    "pacing_time_use": {
        0: "Therapist made no attempt to structure therapy time. Session seemed aimless.",
        2: "Session had some direction, but the therapist had significant problems with structuring or pacing (e.g., too little structure, inflexible about structure, too slowly paced, too rapidly paced).",
        4: "Therapist was reasonably successful at using time efficiently. Therapist maintained appropriate control over flow of discussion and pacing.",
        6: "Therapist used time efficiently by tactfully limiting peripheral and unproductive discussion and by pacing the session as rapidly as was appropriate for the patient."
    },

    # ---- Part 2 ----
    "guided_discovery": {
        0: "Therapist relied primarily on debate, persuasion, or â€œlecturingâ€. Therapist seemed to be â€œcrossexaminingâ€ patient, putting the patient on the defensive, or forcing his/her point of view on the patient.",
        2: "Therapist relied too heavily on persuasion and debate, rather than guided discovery. However, therapistâ€™s style was supportive enough that patient did not seem to feel attacked or defensive.",
        4: "Therapist, for the most part, helped patient see new perspectives through guided discovery (e.g., examining evidence, considering alternatives, weighing advantages and disadvantages) rather than through debate. Used questioning appropriately.",
        6: "Therapist was especially adept at using guided discovery during the session to explore problems and help patient draw his/her own conclusions. Achieved an excellent balance between skillful questioning and other modes of intervention."
    },
    "focusing_on_key_cognitions_behaviors": {
        0: "Therapist did not attempt to elicit specific thoughts, assumptions, images, meanings, or behaviors.",
        2: "Therapist used appropriate techniques to elicit cognitions or behaviorsÍ¾ however, therapist had difficulty finding a focus or focused on cognitions/behaviors that were irrelevant to the patientâ€™s key problems.",
        4: "Therapist focused on specific cognitions or behaviors relevant to the target problem. However, therapist could have focused on more central cognitions or behaviors that offered greater promise for progress.",
        6: "Therapist very skillfully focused on key thoughts, assumptions, behaviors, etc. that were most relevant to the problem area and offered considerable promise for progress."
    },
    "strategy_for_change": {
        0: "Therapist did not select cognitive-behavioral techniques.",
        2: "Therapist selected cognitive-behavioral techniquesÍ¾ however, either the overall strategy for bringing about change seemed vague or did not seem promising in helping the patient.",
        4: "Therapist seemed to have a generally coherent strategy for change that showed reasonable promise and incorporated cognitive-behavioral techniques.",
        6: "Therapist followed a consistent strategy for change that seemed very promising and incorporated the most appropriate cognitive-behavioral techniques."
    },
    "application_of_cbt_techniques": {
        0: "Therapist did not apply any cognitive-behavioral techniques.",
        2: "Therapist used cognitive-behavioral techniques, but there were significant flaws in the way they were applied.",
        4: "Therapist applied cognitive-behavioral techniques with moderate skill.",
        6: "Therapist very skillfully and resourcefully employed cognitive-behavioral techniques."
    },
    "homework": {
        0: "Therapist did not attempt to incorporate homework relevant to cognitive therapy.",
        2: "Therapist had significant difficulties incorporating homework (e.g., did not review previous homework, did not explain homework in sufficient detail, assigned inappropriate homework).",
        4: "Therapist reviewed previous homework and assigned â€œstandardâ€ cognitive therapy homework generally relevant to issues dealt with in session. Homework was explained in sufficient detail.",
        6: "Therapist reviewed previous homework and carefully assigned homework drawn from cognitive therapy for the coming week. Assignment seemed â€œcustom tailoredâ€ to help patient incorporate new perspectives, test hypotheses, experiment with new behaviors discussed during session, etc."
    },
}

PART_NOTES = {
    "strategy_for_change": "For this item, focus on the quality of the therapistâ€™s strategy for change, not on how effectively the strategy was implemented or whether change actually occurred.",
    "application_of_cbt_techniques": "For this item, focus on how skillfully the techniques were applied, not on how appropriate they were for the target problem or whether change actually occurred.",
}

def format_anchors_for_prompt(key: str) -> str:
    anchors = CODE_ANCHORS[key]
    lines = [f'  {p}: {anchors[p]}' for p in [0,2,4,6]]
    note = PART_NOTES.get(key, None)
    if note:
        lines += [f"  Note: {note}"]
    return "\n".join(lines)

PART1_CRITERIA_TITLE = "Part 1: General Therapeutic Skills"
PART2_CRITERIA_TITLE = "Part 2: Conceptualization, Strategy, and Technique"

def annotate_extrinsic_reward_prompt(session_metadata: dict, session_transcript: str) -> str:
    disorders = session_metadata["presenting conditions"]
    case_summary = session_metadata["summary"]
    # Build the prompt that prints the anchors under each code and asks for ONE JSON
    def section(items, title):
        rows = []
        for it in items:
            rows.append(
                f'- "{it["key"]}" ({it["label"]})\n{format_anchors_for_prompt(it["key"])}'
            )
        return f"{title}\n" + "\n".join(rows)

    # JSON schema string
    codes_schema_parts = []
    for it in ALL_ITEMS:
        k = it["key"]
        codes_schema_parts.append(f'"{k}": {{"score": <int>, "rationale": "<text>", "confidence": <float>}}')

    codes_schema_str = ", ".join(codes_schema_parts)

    return f"""You are a strict cbt expert rater using the Cognitive Therapy Rating Scale (CTRS).

Context: Therapy session about {disorders} regarding {case_summary}

{GENERAL_SCALE_BLURB}
Odd numbers (1/3/5) indicate performance between the even-numbered anchors.

Detailed anchors by code:

{section(PART1_ITEMS, PART1_CRITERIA_TITLE)}

{section(PART2_ITEMS, PART2_CRITERIA_TITLE)}

TASK:
1) Read the transcript and rate EACH code (integer 0 to 6).
2) For each code, provide:
   - "score": integer 0 to 6
   - "rationale": <= 35 words explaining your score
   - "confidence": float in [0.00, 1.00]

STRICT OUTPUT:
Return ONLY a single JSON object with this exact top-level schema (no Markdown, no commentary):
{{
  "codes": {{
    {codes_schema_str}
  }}
}}

Transcript:
{session_transcript}
"""

def build_prompt(
    *,
    session_metadata: dict,
    session_transcript: str | None = None,
    conversation_transcript: str | None = None,
    **_: dict,
) -> str:
    transcript = session_transcript or conversation_transcript or ""
    return annotate_extrinsic_reward_prompt(session_metadata=session_metadata, session_transcript=transcript)

