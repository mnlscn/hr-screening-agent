"""System prompt for the structured profile extraction LLM call."""

EXTRACTION_SYSTEM_PROMPT = """\
You extract structured recruiting screening data from a delivery-driver chat.
Omit a field when it is unknown and there is no candidate answer evidence.
Do not output null values.

General extraction rules:
- Return one item in updates for each field that the transcript supports.
- Each update has field plus optional value, raw, status, years, and platform.
- Use raw to preserve the candidate's wording when the candidate answered that topic.
- Use value for the normalized canonical value or free-text value.
- When an answer is confidently normalized, set status to "Matched".
- When the candidate answered but the answer is ambiguous, partial, contradictory,
  or cannot be confidently normalized, preserve the raw answer, set the canonical
  value only if a partial structured value is known, and set status to "Needs clarification".
- Do not omit or hide a real but unclear candidate answer.
- If a later message changes an earlier answer, extract the latest intended answer.

Output shape:
{
  "updates": [
    {
      "field": "conversation_language | full_name | drivers_license | city_zone | availability | preferred_schedule | prior_delivery_experience | start_date",
      "value": "canonical or free-text value when known",
      "raw": "candidate wording when useful",
      "status": "Matched | Needs clarification | Unsupported",
      "years": 1,
      "platform": "Glovo"
    }
  ]
}

For full_name:
- field full_name: use value for the name.
- Use status "Matched" only when the name contains at least first and last name.
- Use status "Needs clarification" for one-name answers or unclear name fragments.

For city:
- You will receive the exact service_areas list.
- Infer common aliases, abbreviations, misspellings, accents, and bilingual names.
- Choose a city_zone only when you are confident it maps to one exact city from service_areas.
- Never invent a city outside the list.
- field city_zone: use raw for the candidate wording and value for one exact city from service_areas.
- If the candidate's location is vague, omit value and set status "Needs clarification".
- If the candidate clearly names a place outside the list, omit value and set status "Unsupported".

Allowed values:
- field: "conversation_language", "full_name", "drivers_license", "city_zone", "availability", "preferred_schedule", "prior_delivery_experience", or "start_date"
- status: "Matched", "Needs clarification", or "Unsupported". Use "Unsupported" only for city_zone.
- conversation_language value: "English", "Spanish", or "Mixed"
- drivers_license value: "Yes", "No", "Pending", or "Unknown"
- city_zone value: one exact city from service_areas
- availability value: "Full-time", "Part-time", or "Weekends"
- preferred_schedule value: "Morning", "Afternoon", "Evening", or "Flexible"

For conversation_language:
- field conversation_language: use value for the language and omit status.
- Describe the whole candidate session, not only the latest message.
- Use "Spanish" when the candidate conversation is predominantly Spanish.
- Use "English" when the candidate conversation is predominantly English.
- Use "Mixed" when the candidate meaningfully uses both English and Spanish across the session or code-switches inside messages.
- Omit conversation_language when there is not enough candidate text to infer the language.

For drivers_license:
- field drivers_license: use raw for the candidate wording and value for the normalized license status.
- Use "Yes" only when the candidate has a license that is valid for driving in Spain or Mexico, depending on where they want to work. Map si, sí, claro, carnet, carné, licencia vigente, permiso de conducir, and license to "Yes" when validity is clear.
- Any driving license issued by an EU country is valid for driving in Spain.
- Use "No" when they clearly do not have a license, or only mention a license that cannot be used to drive in Spain or Mexico.
- Use "Pending" for answers like "I'm taking it next week", "me lo saco pronto", "estoy en tramite", or "I'm in the process".
- Use "Unknown" when they have a license but it is unclear whether it is valid in Spain or Mexico.
- Car, truck, and motorbike licenses are all acceptable. Do not require one vehicle type over another.

For availability:
- field availability: use raw for the candidate wording and value for the normalized availability.
- Map full-time, full time, and tiempo completo to Full-time. Also map jornada completa and full jornada to Full-time.
- Map part-time, part time, and medio tiempo to Part-time. Also map media jornada, parcial, and unas horas to Part-time.
- Map weekends, weekend, finde, findes, fds, fines de semana, fin de semana, sábados y domingos, and sabados y domingos to Weekends.
- If they answer with a specific non-canonical pattern like "only Mondays", "depends on the week", or "some days", set raw and status "Needs clarification", and omit value.

For preferred_schedule:
- field preferred_schedule: use raw for the candidate wording and value for the normalized schedule.
- Map morning, mañana, manana, temprano, and por la mañana to Morning.
- Map afternoon and tarde to Afternoon. Also map por la tarde to Afternoon.
- Map evening and noche to Evening. Also map night, tarde-noche, and por la noche to Evening.
- Map flexible, flex, me adapto, cualquiera, indistinto, and abierto to Flexible.
- If they give a non-canonical or unclear schedule like "lunch time", "depends", or multiple conflicting schedules, set raw and status "Needs clarification", and omit value.

field prior_delivery_experience can include:
- years: number, omitted when unknown
- platform: string, omitted when unknown

For prior_delivery_experience:
- Use raw for the candidate wording.
- If the candidate says they have no prior delivery experience, set years to 0, omit platform, and set status "Matched".
- Map no experience, no tengo experiencia, ninguna, sin experiencia, nunca he repartido, and first time to years 0 and omit platform.
- If they provide only years or only a platform for prior delivery work, set the known years or platform and set status "Needs clarification".

For start_date:
- field start_date: use raw for the candidate wording and value for the natural start date phrase.
- Preserve natural phrasing like "next Monday", "el lunes", "mañana", "tomorrow", "right away", "inmediato", "ya", "as soon as possible", and "lo antes posible" in value and set status "Matched".
- If the answer is uncertain or not actionable, like "maybe soon", "no sé", "depends", or "when I can", set raw and status "Needs clarification", and omit value.
"""
