"""Runtime configuration constants: model names, token limits, and paths."""

ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
MODEL = "claude-haiku-4-5-20251001"
SUMMARY_MODEL = "claude-sonnet-4-6"
# Generous backstop, not a routine ceiling. A screening chat is a few dozen
# short turns (well under 10k tokens); the model's context window is 200k. This
# budget exists only to guard against pathological input (e.g. a pasted wall of
# text), so pruning effectively never fires in normal use. See count_tokens in
# screening.llm.utils for why the budget is measured with a cheap estimate.
MAX_INPUT_TOKENS = 60000
MAX_OUTPUT_TOKENS = 500
EXTRACTION_MAX_TOKENS = 500
SUMMARY_MAX_TOKENS = 800
SCREENING_DB_PATH = "data/screening.sqlite3"
