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

# Anthropic client resilience, shared by the chat, extraction, and summary
# calls. The SDK already retries transient failures (429, 5xx, connection
# errors) with exponential backoff; we raise the count above the default of 2
# and cut the default 10-minute request timeout so an interactive turn fails
# fast instead of hanging. Note: a drop *mid-stream* cannot be retried
# transparently and still surfaces to the caller (see ChatAgent._stream_response).
MAX_RETRIES = 4
REQUEST_TIMEOUT_SECONDS = 90.0
EXTRACTION_MAX_TOKENS = 500
SUMMARY_MAX_TOKENS = 800

SCREENING_DB_PATH = "data/screening.sqlite3"

SCREENING_LOG_LEVEL = "INFO"
SCREENING_LOG_PATH = "logs/screening.jsonl"
SCREENING_LOG_ROTATION = "10 MB"
SCREENING_LOG_RETENTION = "14 days"
SCREENING_SHOW_LOG_PANEL_ENV = "SCREENING_SHOW_LOG_PANEL"
