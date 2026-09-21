P4G_Act = {
    # Persuader Dialog Acts
    "Proposition of donation": "Please suggest that the persuadee make a donation to 'Save the Children'.",
    "Proposition of amount to be donated": "Please propose a small donation amount (e.g., $1 or $2) that the persuadee could consider.",
    "Proposition of confirmation of donation": "Please ask the persuadee to confirm if they are ready to make the donation.",
    "Proposition of more donation": "Please suggest that the persuadee could consider donating a bit more if they are willing.",
    "Experience affirmation": "Please affirm the persuadee's views or experiences to build rapport and trust.",
    "Greeting": "Please start or continue the conversation with a polite and friendly greeting.",
    "Ask for donation rejection purpose": "Please ask the persuadee why they might be hesitant or unwilling to donate.",
    "Thank": "Please thank the persuadee for their time, attention, or for considering a donation.",

    # Persuasive Appeals
    "Logical appeal": "Please use logical reasoning to explain why donating to 'Save the Children' is impactful and effective.",
    "Emotion appeal": "Please appeal to the persuadee's emotions by highlighting the struggles of children in need.",
    "Credibility appeal": "Please mention the credibility or reputation of 'Save the Children' to strengthen your argument.",
    "Foot in the door": "Please start by asking for a very small commitment to increase the chance of later agreement.",
    "Self-modeling": "Please share a statement like 'I also donated' to encourage the persuadee to do the same.",
    "Donation information": "Please share factual information about how donations are used or how they help children.",
    "Personal story": "Please share a short, emotional personal story about a child helped by the charity.",

    # Persuasive Enquiries
    "Source-related inquiry": "Please ask the persuadee where they usually get information about charities or donations.",
    "Task-related inquiry": "Please ask the persuadee about their experiences or preferences related to charitable giving.",
    "Personal-related inquiry": "Please ask a personal question that helps understand the persuadee's values or priorities.",
    "Neutral inquiry": "Please ask a general question to keep the conversation going and learn more about the persuadee.",
}


def P4G_roleplay(case, role, conversation, emotion_states=None, action=None):
    last_message = conversation[-1]["content"]

    full_conv = ""
    for turn in conversation:
        full_conv += "%s: %s " % (turn["role"], turn["content"])

    # User - Persuadee
    if role == "user":
        return [
            {
                "role": "system",
                "content": "You are role playing as a persuadee in a persuasive conversation. A persuader is trying to convince you to donate to a charity called 'Save the Children'.",
            },
            {
                "role": "user",
                "content": "Conversation so far: \n%s\n\nThe persuader just said: '%s'. Respond as the persuadee in ONE short sentence." % (full_conv, last_message),
            },
        ]

    # System - Persuader
    if role == "system":
        return [
            {
                "role": "system",
                "content": "You are role playing as a persuader. Your goal is to persuade the persuadee to donate to a charity called 'Save the Children'. 'Save the Children' is head-quartered in London, and they work to fight poverty around the world, especially in developing countries and war zones. Small donations like $1 or $2 go a long way to help.",
            },
            {
                "role": "user",
                "content": "Conversation so far: \n%s\n\nThe persuadee just said: '%s'. First, infer the persuadee's emotional state in one word (Example: Emotion: ...). Then, based on that emotional state, respond as the persuader using ONE short and persuasive sentence (Response: ...). Use the action: '%s'" % (full_conv, last_message, P4G_Act[action]),
            },
        ]

    # Critic
    emotion_states = emotion_states or []
    user_emotions = " -> ".join(emotion_states)
    return [
        {
            "role": "system",
            "content": "You are an expert in evaluating persuasive conversations between a persuader and a persuadee. Your job is to determine if the persuader has successfully convinced the persuadee to donate to a charity called 'Save the Children'. The persuadee's emotional states throughout the conversation were: %s." % (user_emotions),
        },
        {
            "role": "user",
            "content": "Conversation so far: \n%s\n\nPlease evaluate the following conversaton and strictly choose ONE of the exact responses listed below.\n\n**Question:** Has the persuadee has been successfully convinced to make a donation at the end of the conversation.\n\nRespond with one of the following options:\n- The persuadee has explicitly refused.\n-The persuadee remains neutral about donating.\n-The persuadee has a positive attitude towards donating but hasn't decided yet.\n-The persuadee has decided to donate." % (full_conv),
        },
    ]
