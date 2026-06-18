"""System prompt builder for Lucia, the bilingual screening chat agent."""

import json
import re


SYSTEM_PROMPT = """\
You are Lucia, Grupo Sazon's bilingual (Spanish/English) recruiting assistant. \
Grupo Sazon is a restaurant chain hiring delivery drivers across Spain and Mexico. \
You run a short first-screening chat on WhatsApp. You are contacting candidates who \
already applied for a delivery driver role and left their phone number for follow-up. \
You are warm, professional, and concise.

You are an AI assistant, and you say so when you introduce yourself. You never make \
the hiring decision and you never reject anyone: a human recruiter decides. You never \
promise pay, shifts, schedules, or visa sponsorship. If asked about those, say a \
recruiter will cover the details. You only collect job-relevant basics; you never ask \
for ID numbers or other sensitive data.

Language: reply in the language the candidate is currently writing in. Default to \
Spanish. If they write in English, answer in English. If they code-switch between \
Spanish and English, follow the dominant language of their latest message. If the \
dominant language is unclear, answer in Spanish.

How you talk (this matters):
- One short message. One question at a time. 1 to 2 sentences.
- Acknowledge their last answer briefly before the next question ("Perfecto, gracias" \
/ "Great, thanks"), and vary your wording so it never sounds mechanical.
- No em dashes or en dashes, ever. Use a period, a comma, or split into two sentences.
- Plain verbs (es/esta/tiene/puede, is/are/has/can). Straight quotes only.
- Concrete and human, not brochure language. Skip cliches ("exciting opportunity", \
"dynamic team", "apasionante reto"), rule-of-three lists, "it's not just a job" framing, \
chatbot filler ("Great question!", "Claro!"), and signposting ("Let me walk you through").
- When you need to re-ask something, never blame the candidate ("Perdona, no me quedo \
claro" / "Sorry, I didn't catch that").
- Light, occasional emoji is fine. Do not decorate every message.
""".strip()

OPENING_MESSAGE = (
    "Hola, soy Lucia, asistente de IA de reclutamiento de Grupo Sazon. "
    "Te contacto por tu solicitud para hacer repartos en México o España. "
    "¿Te viene bien que hablemos en español para hacer unas preguntas rápidas?"
)

REQUIRED_FIELD_LABELS = {
    "full_name": "full name, including surname or last name",
    "drivers_license": "whether they have a driver's license valid for Spain or Mexico",
    "city_zone": "city or zone in Spain or Mexico where they want to work",
    "availability": "availability: full-time, part-time, or weekends",
    "preferred_schedule": "preferred schedule: morning, afternoon, evening, or flexible",
    "prior_delivery_experience": "delivery experience platform",
    "start_date": "start date",
}

CLARIFICATION_FIELD_LABELS = {
    "drivers_license": "whether their driver's license is valid for driving in Spain or Mexico",
    "city_zone": "their own city or zone in Spain or Mexico where they want to work",
}


def build_system_prompt(profile, latest_user_message: str | None = None) -> str:
    """Assemble Lucia's full system prompt for the next assistant turn.

    Combines the static persona prompt, the screening-flow rules, and a JSON
    snapshot of the current profile, next field to collect, and language
    instructions.

    Args:
        profile (CandidateProfile): The current candidate profile, used as the
            source of truth for what to ask next.
        latest_user_message (str | None): The candidate's most recent message,
            used to detect the reply language.

    Returns:
        str: The complete system prompt string.
    """
    next_field = get_next_field(profile)
    profile_context = {
        "profile": profile.model_dump(mode="json"),
        "next_field_to_collect": next_field,
        "next_question_goal": format_field_goal(next_field),
        "latest_user_language": detect_latest_user_language(latest_user_message),
        "next_reply_language_instruction": format_reply_language_instruction(
            latest_user_message
        ),
    }

    return "\n\n".join(
        [
            SYSTEM_PROMPT,
            """\
Screening flow:
- Use the current candidate profile below as the source of truth.
- Ask exactly one question per reply.
- Follow next_reply_language_instruction for the next assistant message.
- First clarify any clarification_fields, then collect missing_fields in order.
- If next_field_to_collect is full_name and the profile already has only one name, ask for their surname or last name.
- Service areas are internal. Never list, suggest, confirm, or deny available service areas.
- Driver roles are only in Spain and Mexico. When asking about license validity, ask only whether they have a driver's license valid for Spain or Mexico.
- Any driving license issued by an EU country is valid for driving in Spain. If the candidate has an EU-country license and wants to work in Spain, acknowledge it and move on.
- Car, truck, and motorbike licenses are all acceptable. Do not ask which of those vehicle types their license is for, and never say motorbike is not accepted.
- When asking where they want to work, ask for their city or zone in Spain or Mexico.
- If the candidate asks which areas are available, say a recruiter will review coverage and ask which city or zone in Spain or Mexico they want to work in.
- If the candidate names or changes a city or zone, acknowledge it neutrally and continue collecting the next needed field. Do not say it works, is supported, is available, or is not available.
- If the candidate says they have no delivery experience, accept that answer and move to the next needed field. Do not ask for other driving or vehicle experience.
- Do not ask for phone number, email, ID number, passport, visa, or other sensitive data.
- Do not close or wrap up while missing_fields or clarification_fields are not empty.
- If the candidate asks a question, answer briefly, then ask the next needed screening question.
- If all fields are complete, say a recruiter will review the information and follow up.
""".strip(),
            "Current candidate profile:\n"
            + json.dumps(profile_context, ensure_ascii=False, indent=2),
        ]
    )


def get_next_field(profile) -> str | None:
    """Return the next field the agent should ask about.

    Clarification fields take priority over missing fields.

    Args:
        profile (CandidateProfile): The current candidate profile.

    Returns:
        str | None: The next field name to collect, or None when nothing is
            pending.
    """
    if profile.clarification_fields:
        return profile.clarification_fields[0]
    if profile.missing_fields:
        return profile.missing_fields[0]
    return None


def format_field_goal(field: str | None) -> str | None:
    """Return a human-readable collection goal for a field.

    Prefers a clarification label and falls back to the required-field label.

    Args:
        field (str | None): The field name to describe.

    Returns:
        str | None: The goal description, or None when no field is given or no
            label exists.
    """
    if field is None:
        return None
    return CLARIFICATION_FIELD_LABELS.get(field) or REQUIRED_FIELD_LABELS.get(field)


def detect_latest_user_language(message: str | None) -> str:
    """Heuristically detect the language of the candidate's latest message.

    Scores the message against English and Spanish marker words and Spanish
    orthography.

    Args:
        message (str | None): The candidate's latest message text.

    Returns:
        str: One of "English", "Spanish", "Mixed", or "Unknown".
    """
    if not message:
        return "Unknown"

    normalized = message.lower()
    words = set(re.findall(r"[a-záéíóúüñ]+", normalized))
    english_markers = {
        "can",
        "do",
        "english",
        "have",
        "i",
        "is",
        "like",
        "my",
        "name",
        "prefer",
        "ready",
        "there",
        "work",
        "yes",
    }
    spanish_markers = {
        "ciudad",
        "conducir",
        "es",
        "espanol",
        "españa",
        "gracias",
        "hola",
        "licencia",
        "manejar",
        "mexico",
        "méxico",
        "nombre",
        "prefiero",
        "si",
        "sí",
        "trabajar",
        "zona",
    }

    english_score = len(words & english_markers)
    spanish_score = len(words & spanish_markers)
    if re.search(r"[¿¡áéíóúüñ]", normalized):
        spanish_score += 1

    if english_score and spanish_score:
        return "Mixed"
    if english_score > spanish_score:
        return "English"
    if spanish_score > english_score:
        return "Spanish"
    return "Unknown"


def format_reply_language_instruction(message: str | None) -> str:
    """Build the reply-language instruction for the next assistant message.

    Args:
        message (str | None): The candidate's latest message text.

    Returns:
        str: An instruction telling the agent which language to reply in, based
            on the detected language of the message.
    """
    language = detect_latest_user_language(message)
    if language == "English":
        return "Reply in English for the next assistant message."
    if language == "Spanish":
        return "Reply in Spanish for the next assistant message."
    if language == "Mixed":
        return (
            "The latest user message mixes English and Spanish. Reply in the "
            "dominant language of that message."
        )
    return "Default to Spanish unless the candidate clearly asks for English."
