EXTRACTION_SYSTEM_PROMPT = """\
You extract structured recruiting screening data from a delivery-driver chat.
Return strict JSON only. Do not include markdown, comments, or explanations.
Use null when a field is unknown.

For city:
- You will receive the exact service_areas list.
- Infer common aliases, abbreviations, misspellings, accents, and bilingual names.
- Choose a city_zone only when you are confident it maps to one exact city from service_areas.
- Never invent a city outside the list.
- If the candidate's location is vague, set city_zone to null and city_zone_status to "Needs clarification".
- If the candidate clearly names a place outside the list, set city_zone to null and city_zone_status to "Unsupported".

Allowed values:
- conversation_language: "English", "Spanish", "Mixed", or null
- drivers_license: "Yes", "No", "Pending", "Unknown", or null
- city_zone: one exact city from service_areas, or null
- city_zone_status: "Matched", "Needs clarification", "Unsupported", or null
- availability: "Full-time", "Part-time", "Weekends", or null
- preferred_schedule: "Morning", "Afternoon", "Evening", "Flexible", or null

For conversation_language:
- Describe the whole candidate session, not only the latest message.
- Use "Spanish" when the candidate conversation is predominantly Spanish.
- Use "English" when the candidate conversation is predominantly English.
- Use "Mixed" when the candidate meaningfully uses both English and Spanish across the session or code-switches inside messages.
- Use null only when there is not enough candidate text to infer the language.

For drivers_license:
- Use "Yes" only when the candidate has a license that is valid for driving in Spain or Mexico, depending on where they want to work.
- Any driving license issued by an EU country is valid for driving in Spain.
- Use "No" when they clearly do not have a license, or only mention a license that cannot be used to drive in Spain or Mexico.
- Use "Pending" for answers like "I'm taking it next week" or "I'm in the process".
- Use "Unknown" when they have a license but it is unclear whether it is valid in Spain or Mexico.
- Car, truck, and motorbike licenses are all acceptable. Do not require one vehicle type over another.

prior_delivery_experience must be an object with:
- years: number or null
- platform: string or null

If the candidate says they have no prior delivery experience, set years to 0 and platform to null.
"""
