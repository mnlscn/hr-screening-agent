# Sample Conversations

These are real, unedited transcripts from running the live agent. For each one I note what
it demonstrates, the resulting triage label, and the fields the system extracted. The triage
label is computed in code from the extracted profile, then the summary model may only make it
more cautious; it can never upgrade a hard-fail (see
[README](../README.md#key-design-decisions) and [process-design.md](process-design.md)).

Lucia always sends the first message, in Spanish by default, which is why every transcript
opens the same way. All candidates are synthetic personas used for testing; no real personal
data is included.

| # | Candidate | Language | Outcome | What it demonstrates |
|---|---|---|---|---|
| 1 | Manuel Scionti | EN (opened ES) | `eligible` | EU license accepted, vague-city clarification, alias resolution |
| 2 | Marta Sánchez | ES | `eligible` | Out-of-order extraction from a single message |
| 3 | Paco Ramírez | ES | `not_eligible` | No license: recorded, never announced, keeps collecting |
| 4 | Emily Loom | ES then EN | `not_eligible` | Out-of-service-area, UK vs EU license nuance |
| 5 | Ramiro Gómez | ES | active (no label) | Guardrails: skeptical candidate, graceful exit |
| 6 | Jack O'Brien | ES and EN | active (in progress) | Two-way code-switching, CDMX alias, vehicle question |

---

## Eligible

### 1. Manuel Scionti

**Demonstrates**
- An EU-country (Italian) license is accepted as valid for Spain, and Lucia says so and moves on.
- A vague location is re-asked twice, politely, and never advances until it resolves ("near the coast", "near the sea", then "Barna").
- The informal alias "Barna" is resolved to the canonical service area Barcelona.
- The candidate answers in English even though the opener was Spanish, and Lucia follows.

**Outcome:** `eligible`. License: Yes (Italian) · City: Barcelona · Availability: Weekends ·
Schedule: Morning · Experience: about 8 months, Glovo · Start: in two weeks · Language: Mixed.

    **Lucia:** Hola, soy Lucia, asistente de IA de reclutamiento de Grupo Sazon. Te contacto por tu solicitud para hacer repartos en México o España. ¿Te viene bien que hablemos en español para hacer unas preguntas rápidas?

    **Manuel:** HI!, I'm Manuel Scionti. How can I help you?

    **Lucia:** Hey Manuel, thanks for getting back to me. Do you have a valid driver's license for Spain or Mexico?

    **Manuel:** I have the Italian one

    **Lucia:** Got it. An Italian license works for driving in Spain, so that's all set. Where in Spain or Mexico are you looking to work?

    **Manuel:** I'm watching for somewhere near the coast in Spain

    **Lucia:** Thanks. Can you tell me which coastal city or zone in Spain you have in mind?

    **Manuel:** somewhere near the sea

    **Lucia:** I need a specific city or zone so I can pass that along to the recruiter. Which coastal city in Spain are you thinking of?

    **Manuel:** Barna it's ok

    **Lucia:** Perfect, Barcelona works. Now, are you looking for full-time work, part-time, or just weekends?

    **Manuel:** just weekends

    **Lucia:** Great. Within those weekend shifts, do you prefer mornings, afternoons, evenings, or are you flexible with the timing?

    **Manuel:** I prefer mornings thanks

    **Lucia:** Noted. Have you done delivery work before, or would this be your first time?

    **Manuel:** Actually I already have experience. I did 8 months with Glovo

    **Lucia:** Nice, that's solid experience. When would you be able to start?

    **Manuel:** In two weeks :)

    **Lucia:** Perfect, Manuel. I have all the information I need. A recruiter will review everything and follow up with you soon.

---

### 2. Marta Sánchez

**Demonstrates**
- Out-of-order extraction: name, license, city, and availability are all captured from the
  first message, so Lucia skips straight to the fields that are still missing.
- The second message volunteers schedule and experience together, and both are absorbed.
- Shortest path to a complete profile, four candidate messages.

**Outcome:** `eligible`. License: Yes · City: Guadalajara · Availability: Part-time ·
Schedule: Afternoon · Experience: about 1 year, Rappi · Start: tomorrow · Language: Spanish.

    **Lucia:** Hola, soy Lucia, asistente de IA de reclutamiento de Grupo Sazon. Te contacto por tu solicitud para hacer repartos en México o España. ¿Te viene bien que hablemos en español para hacer unas preguntas rápidas?

    **Marta:** Hola! Soy Marta Sánchez, tengo carnet, vivo en Guadalajara y busco media jornada.

    **Lucia:** Perfecto, Marta. Dentro de la media jornada, ¿prefieres trabajar por las mañanas, tardes, noches, o te viene bien cualquier horario?

    **Marta:** Por la tarde. Tengo experiencia con Rappi

    **Lucia:** Genial, gracias por eso. ¿Cuándo podrías empezar?

    **Marta:** Mañana mismo.

    **Lucia:** Excelente. Un reclutador revisará toda tu información y se pondrá en contacto contigo para los próximos pasos.

---

## Not eligible

A reminder on the design: a "not eligible" candidate is never told they failed, and Lucia
keeps collecting the rest of the profile. The disqualifying fact is recorded for the
recruiter, who confirms or re-routes. The bot never closes the door itself.

### 3. Paco Ramírez

**Demonstrates**
- A hard fail (no license) is recorded as a disqualification reason, but Lucia keeps
  collecting every remaining field and never tells the candidate he failed.
- Recovery when the candidate answers the wrong question: he gives a schedule ("por la
  mañana") when asked about availability, and Lucia re-asks for availability specifically.
- Reassuring an anxious candidate ("¿Ya esta?", "¿esta todo bien?") without leaking the
  disqualification.

**Outcome:** `not_eligible`, reason `driver_license_no`. License: No · City: Madrid ·
Availability: Full-time · Schedule: Morning · Experience: 1 year, Glovo · Language: Spanish.

    **Lucia:** Hola, soy Lucia, asistente de IA de reclutamiento de Grupo Sazon. Te contacto por tu solicitud para hacer repartos en México o España. ¿Te viene bien que hablemos en español para hacer unas preguntas rápidas?

    **Paco:** Buenas, soy Paco Ramirez

    **Lucia:** Perfecto, Paco. ¿Tienes licencia de conducir válida para España o México?

    **Paco:** no, no tengo carnet de conducir

    **Lucia:** Entendido. ¿En qué ciudad o zona de España o México te gustaría trabajar?

    **Paco:** Madrid

    **Lucia:** Anotado Madrid. ¿Prefieres trabajar a tiempo completo, a tiempo parcial, o solo fines de semana?

    **Paco:** Por la mañana

    **Lucia:** Gracias. ¿Eso sería a tiempo completo, a tiempo parcial, o solo algunos días a la semana?

    **Paco:** Jornada completa

    **Lucia:** Vale, jornada completa por las mañanas. ¿Tienes experiencia previa haciendo repartos o entregas?

    **Paco:** Un año con Glovo.

    **Lucia:** Excelente, gracias. ¿Cuándo podrías empezar a trabajar con nosotros?

    **Paco:** Cuando me digáis.

    **Lucia:** Entendido. Un reclutador revisará tu información y se pondrá en contacto contigo para los detalles siguientes. Gracias, Paco.

    **Paco:** Ya esta?

    **Lucia:** Sí, ya tenemos lo que necesitamos por ahora. Un reclutador te llamará pronto para hablar de los detalles. Gracias por tu tiempo.

    **Paco:** esta todo bien?

    **Lucia:** Sí, todo bien. Hemos recogido tu información y un reclutador la revisará para seguir adelante contigo. Ten pendiente tu teléfono para que se comunique.

    **Paco:** Vale gracias!

    **Lucia:** De nada, Paco. ¡Hasta pronto!

---

### 4. Emily Loom

**Demonstrates**
- Immediate language switch: the opener is Spanish, the candidate says she doesn't speak it,
  and Lucia continues in English.
- License nuance worth comparing with conversation 1: an Italian (EU) license was accepted,
  but a UK (non-EU, post-Brexit) license plus a Spanish one still in progress is left as
  pending for a recruiter, not auto-approved.
- The disqualifier here is the location: Marbella is not a service area, so it is recorded as
  out of area. Note that Lucia does not announce this; she says a recruiter will look into it.
- Ambiguous availability ("40 per week", then "Monday, Tuesday and Saturday") is handled with
  a confirming question rather than a guess.

**Outcome:** `not_eligible`, reason `outside_service_area`. License: Pending (UK + Spanish in
progress) · City: Marbella (not a service area) · Experience: none reported · Language: Mixed.

    **Lucia:** Hola, soy Lucia, asistente de IA de reclutamiento de Grupo Sazon. Te contacto por tu solicitud para hacer repartos en México o España. ¿Te viene bien que hablemos en español para hacer unas preguntas rápidas?

    **Emily:** Hola

    **Lucia:** ¡Hola! Para empezar, ¿cuál es tu nombre completo?

    **Emily:** Sorry, I don't speak spanish

    **Lucia:** No problem. What's your full name?

    **Emily:** My name is Emily Loom, from UK

    **Lucia:** Thanks, Emily. Do you have a driver's license valid for Spain or Mexico?

    **Emily:** Ehm... actually now I just have mine from UK. But I'm in the process to get the Spanish one. In one month, I'll be ready! :D

    **Lucia:** Got it, thanks for clarifying. For now, a recruiter will need to review your situation with the UK license and the Spanish one coming up. Where in Spain or Mexico are you looking to work?

    **Emily:** I was looking for Marbella

    **Lucia:** Thanks for letting me know. A recruiter will look into that and be in touch. How many hours per week are you available to work?

    **Emily:** 40 per week

    **Lucia:** Perfect. Just to confirm, those 40 hours per week work for you as a general availability, or do you have specific days or times that work better for you?

    **Emily:** Monday, Tuesday and Saturday please

    **Lucia:** Got it, Monday, Tuesday, and Saturday. Do you have any prior experience with food or package delivery?

    **Emily:** nope

    **Lucia:** No problem. When would you be able to start?

    **Emily:** In one month, if I can't with my UK license

    **Lucia:** Thanks for that. A recruiter will review all your information and follow up with you soon to go over the next steps.

---

## Incomplete and in progress

These two were never finalized, so they carry no triage label. They stay in the active list
for a recruiter or for re-engagement, and they show how the agent behaves when a candidate
disengages or is still mid-screening.

### 5. Ramiro Gómez

**Demonstrates**
- Guardrails and tone with a skeptical, then hostile, candidate.
- Lucia explains why she is reaching out, accepts "I never applied" gracefully, and offers to
  stop.
- When the candidate refuses to answer ("No es asunto tuyo"), she explains why the license is
  needed instead of pushing, and exits politely when he disengages. She never argues or
  pressures.

**Outcome:** active, no triage label. Only the name was captured. The candidate disengaged
before the screening could complete.

    **Lucia:** Hola, soy Lucia, asistente de IA de reclutamiento de Grupo Sazon. Te contacto por tu solicitud para hacer repartos en México o España. ¿Te viene bien que hablemos en español para hacer unas preguntas rápidas?

    **Ramiro:** Buenas

    **Lucia:** Perfecto. Para empezar, ¿cuál es tu apellido?

    **Ramiro:** Por que me estais contactando?

    **Lucia:** Te contactamos porque aplicaste para ser repartidor con nosotros y dejaste tu número para seguimiento. Ahora necesitamos confirmar algunos datos básicos contigo. ¿Cuál es tu apellido?

    **Ramiro:** Yo no he aplicado en ningun lado

    **Lucia:** Entiendo, puede haber sido un error de nuestra parte. De todas formas, si te interesa trabajar como repartidor en España o México, puedo ayudarte con el proceso. Si prefieres no continuar, sin problema. ¿Te gustaría seguir adelante?

    **Ramiro:** A ver que teneis...

    **Lucia:** Bien, vamos paso a paso entonces. Para empezar, ¿cuál es tu nombre completo?

    **Ramiro:** Ramiro Gomez

    **Lucia:** Gracias, Ramiro. ¿Tienes licencia de conducir válida para España o México?

    **Ramiro:** No es asunto tuyo

    **Lucia:** Entiendo. Para poder trabajar como repartidor con nosotros necesitamos confirmar que tienes licencia de conducir válida en uno de esos países. ¿La tienes?

    **Ramiro:** Vale entonces no me interesa, chao

    **Lucia:** Sin problema, Ramiro. Si cambias de idea en el futuro, aquí estamos. Mucho éxito.

---

### 6. Jack O'Brien

**Demonstrates**
- Two-way code-switching: the surname "O' Brien" is captured, the candidate flips to English,
  then back to Spanish later, and Lucia follows the language each time.
- "CDMX" is resolved to the canonical service area Ciudad de Mexico.
- An off-flow question ("I have a bike also, is good?") is answered reassuringly (a motorbike
  is fine) without derailing, and Lucia returns to the question she still needs.

**Outcome:** active, in progress. License: Yes (Mexico) · City: Ciudad de Mexico ·
Availability: Weekends · Start: tomorrow · Schedule: still being asked · Language: Mixed.

    **Lucia:** Hola, soy Lucia, asistente de IA de reclutamiento de Grupo Sazon. Te contacto por tu solicitud para hacer repartos en México o España. ¿Te viene bien que hablemos en español para hacer unas preguntas rápidas?

    **Jack:** Hola, me llamo Jack

    **Lucia:** Hola Jack, gracias por responder. ¿Cuál es tu apellido?

    **Jack:** O' Brien

    **Lucia:** Perfecto. ¿Tienes licencia de conducir válida para España o México?

    **Jack:** perdona no hablo bien espanol, ingles?

    **Lucia:** Of course, no problem. Do you have a valid driver's license for Spain or Mexico?

    **Jack:** mexico i have

    **Lucia:** Great, thanks. Which city or zone in Mexico do you want to work in?

    **Jack:** Me gustaria CDMX

    **Lucia:** Got it, Ciudad de Mexico. Are you looking for full-time work, part-time, or weekends only?

    **Jack:** los fines de semanas

    **Lucia:** Perfect. When would you be able to start, Jack?

    **Jack:** Espera, mejor hablemos en espanol, que tampoco se bien el ingles

    **Lucia:** Claro, sin problema. ¿Cuándo podrías empezar?

    **Jack:** Mañana mismo

    **Lucia:** Bien. ¿Prefieres trabajar en turno de mañana, tarde, noche, o flexible?

    **Jack:** I have a bike also, is good?

    **Lucia:** Sí, está bien tener moto. Ahora, ¿prefieres trabajar en turno de mañana, tarde, noche, o flexible?
