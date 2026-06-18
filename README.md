# Screening

Run the Streamlit UI:

```bash
uv run streamlit run src/screening/ui/app.py
```

Set `ANTHROPIC_API_KEY` in your environment or `.env` before starting the app.

## Package layout

- `screening.application`: candidate-session workflows.
- `screening.domain`: candidate models and service-area data helpers.
- `screening.llm`: chat agent, prompts, extraction, summaries, and LLM utilities.
- `screening.persistence`: SQLite storage.
- `screening.ui`: Streamlit app, dashboard, analytics, and UI state.
