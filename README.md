# Lucia — AI Candidate Screening Agent

An AI agent that screens delivery-driver applicants for **Grupo Sazón** over a messaging
interface. "Lucia" runs a short, bilingual (Spanish/English) chat, collects and validates
the seven required screening fields, and hands each candidate to a human recruiter with a
triage label (`eligible` / `not_eligible` / `needs_review`) and a structured HR summary.

The conversation design — stages, validation rules, edge cases, and outcomes — is documented
separately in **[docs/process-design.md](docs/process-design.md)** (Phase 1).

---

## What it does

- **Conversational screening** following a fixed seven-field flow, one question at a time.
- **Structured data extraction** — every turn is parsed into a validated candidate profile.
- **Bilingual ES/EN** with code-switching support; replies in the candidate's current language.
- **Triage + summary** for recruiters: a deterministic eligibility label plus an
  evidence-based HR summary.
- **HR dashboard** to browse, search, and filter candidates by triage/status/city.
- **Analytics**: completion rate, drop-off stage, screening funnel, duration, city map, and
  stale-candidate detection.

---

## Setup

**Prerequisites:** Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies (creates .venv from uv.lock)
uv sync

# 2. Provide your Anthropic API key
cp .env.example .env        # then edit .env and set ANTHROPIC_API_KEY
# (or export ANTHROPIC_API_KEY=... in your shell)

# 3. Run the app
uv run streamlit run src/screening/ui/app.py
```

The app opens with three tabs: **Candidate chat**, **HR dashboard**, and **Analytics**.
The HR dashboard and Analytics work without an API key; the chat requires one. Data persists
to a local SQLite file at `data/screening.sqlite3` (created on first run).

**Development:**

```bash
uv run pytest          # test suite (LLM calls are mocked — no key or network needed)
uv run ruff check src tests
uv run ruff format --check src tests
uv run ty check        # type checking
```

CI runs lint, format, type-check, and tests on every push/PR ([.github/workflows/ci.yml](.github/workflows/ci.yml)).

---

## Architecture

```
Candidate ──▶ Streamlit chat ──▶ ChatAgent ──▶ Anthropic (Haiku)  ─┐
                                     │                              │ streamed reply
                                     ├─▶ CandidateExtractor (Haiku) │
                                     │      └─▶ CandidateProfile ◀──┘
                                     ▼
                                  SQLite  ──▶ HR dashboard / Analytics
                                     ▲
              Recruiter "Finish" ────┴──▶ CandidateSummarizer (Sonnet) ─▶ label + HR summary
```

Each candidate turn triggers two LLM calls: an **extraction** pass that re-reads the
transcript into a structured profile, and a **chat** pass that streams Lucia's next message.
The profile's derived state (`missing_fields`, `clarification_fields`,
`disqualification_reasons`) decides which question comes next — see *Key design decisions*.

### Package layout

The code is organized in layers under `src/screening/`:

| Package | Responsibility |
|---|---|
| `screening.domain` | `CandidateProfile` (Pydantic) with validation, ES/EN normalization, and derived state; canonical service-area data helpers. |
| `screening.llm` | `ChatAgent` (conversation orchestration, streaming, memory pruning, error rollback), `CandidateExtractor` (transcript → structured JSON), `CandidateSummarizer` (triage label + HR summary), prompts, and LLM utilities. |
| `screening.application` | Session lifecycle workflows: start / resume / finalize. |
| `screening.persistence` | SQLite storage (candidates + messages, schema, indexes). |
| `screening.ui` | Streamlit app: chat, HR dashboard, analytics, sidebar, and their pure-logic helpers. |
| `screening.config` | Model IDs, token budgets, and the database path. |

The `ui` package separates rendering (`*.py`) from pure logic (`*_logic.py`) so the
analytics/dashboard computations are unit-tested without a running Streamlit.

---

## Key design decisions

1. **Deterministic flow, generative phrasing.** A rule-based layer decides *what* to ask
   next from the profile state; the LLM only decides *how* to phrase it. This keeps the
   screening exhaustive and predictable while the conversation stays natural, and makes the
   flow testable without the model.

2. **Two-model split (cost vs. quality).** `claude-haiku-4-5` handles the high-volume,
   latency-sensitive work (chat + per-turn extraction); `claude-sonnet-4-6` runs once per
   candidate to write the recruiter summary, where quality matters most. This keeps
   per-screening cost low while spending on the one output a human reads.

3. **Separate, non-destructive extraction.** The structured profile is rebuilt from the full
   transcript each turn and *merged* with prior values, so a later vague answer never erases
   a value captured earlier, and out-of-order answers are absorbed.

4. **Triage computed in code; the LLM can only add caution.** The eligibility label is
   derived deterministically from the profile. The summarizer may downgrade toward
   `needs_review` but can never upgrade a recorded hard-fail to `eligible`; on disagreement
   it falls back to `needs_review`. Lucia herself never accepts or rejects anyone.

5. **SQLite over flat files.** A single-file database with a real schema, indexes, and a
   messages table — zero infrastructure for the demo, while giving proper querying for the
   dashboard and a clean migration path to Postgres.

6. **Bilingual validation layer.** Pydantic validators normalize the long tail of ES/EN
   phrasings to canonical values and reject anything that can't be mapped (e.g. a city not in
   the service areas), rather than letting the model guess.

7. **Resilient by default.** A failed stream rolls back the turn (no half-written state); a
   failed extraction is non-fatal and the chat continues; a failed summary is recorded and
   the candidate is marked `needs_review`; conversation history is pruned to a token budget.

---

## Potential improvements

With more time, in rough priority order:

- **Fold extraction into the chat call** via tool-use/structured output. Today each turn
  makes two calls and re-reads the whole transcript (cost grows with conversation length) —
  the main thing to optimize before scaling to ~10K candidates/week.
- **Async re-engagement.** Stale candidates are detected and surfaced to recruiters today;
  the designed automated nudge (≈24h / ≈72h follow-ups) needs a scheduler the synchronous
  chat doesn't have yet.
- **Scale the datastore.** Postgres with WAL/connection pooling to handle concurrent writes;
  add structured logging, metrics, and tracing for observability.
- **FAQ / RAG** so Lucia can actually answer candidate questions (pay ranges, process)
  instead of deferring everything to a recruiter.
- **Robustness:** the shared Anthropic client now tunes the SDK's retry count and request
  timeout, and chat errors are mapped to candidate-facing messages — but a drop mid-stream
  still loses the turn. Add a resend affordance that preserves the candidate's message, and
  log request IDs for incident tracing.
- **Privacy & access:** authentication on the dashboard, a PII retention policy, and storing
  candidate JSON without `ensure_ascii` so accented names aren't escaped.
- **Quality evals.** The current tests verify plumbing (LLM calls mocked); add golden-
  transcript evals that exercise extraction and conversation quality against a live model.
- **Voice agent** (bonus tier) and an **ATS integration** API spec.
- Remove the unused `disqualified` candidate status (triage currently lives on `bot_label`).
