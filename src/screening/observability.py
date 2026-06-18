"""Privacy-aware Loguru setup and structured logging helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger as _logger

from screening.config import (
    SCREENING_LOG_LEVEL,
    SCREENING_LOG_PATH,
    SCREENING_LOG_RETENTION,
    SCREENING_LOG_ROTATION,
)


logger = _logger
logger.remove()

_configured = False
_sink_ids: list[int] = []

_SENSITIVE_FIELD_NAMES = frozenset(
    {
        "api_key",
        "completion",
        "content",
        "full_name",
        "hr_summary",
        "message",
        "messages",
        "profile",
        "prompt",
        "raw_city_zone",
        "raw_drivers_license",
        "response",
        "summary",
        "transcript",
        "user_input",
    }
)


def configure_logging(
    *,
    log_path: str | Path | None = None,
    level: str | None = None,
    include_stderr: bool = True,
    force: bool = False,
) -> None:
    """Configure Loguru sinks for local app observability.

    The setup is idempotent by default so Streamlit reruns do not duplicate
    sinks. Use ``force=True`` in tests or one-off scripts that need to replace
    an existing sink.

    Args:
        log_path (str | Path | None): Optional file path override. Defaults to
            ``SCREENING_LOG_PATH`` or the environment variable of the same name.
        level (str | None): Optional log level override. Defaults to
            ``SCREENING_LOG_LEVEL`` or the environment variable of the same
            name.
        include_stderr (bool): Whether to add a readable stderr sink.
        force (bool): Whether to replace previously configured sinks.
    """
    global _configured

    if _configured and not force:
        return

    if force:
        _remove_configured_sinks()

    log_level = level or os.getenv("SCREENING_LOG_LEVEL", SCREENING_LOG_LEVEL)
    resolved_log_path = Path(
        log_path or os.getenv("SCREENING_LOG_PATH", SCREENING_LOG_PATH)
    )
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)

    if include_stderr:
        _sink_ids.append(
            logger.add(
                sys.stderr,
                level=log_level,
                format=(
                    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                    "<level>{level: <8}</level> | {message} | {extra}"
                ),
                backtrace=False,
                diagnose=False,
            )
        )

    _sink_ids.append(
        logger.add(
            resolved_log_path,
            level=log_level,
            rotation=os.getenv("SCREENING_LOG_ROTATION", SCREENING_LOG_ROTATION),
            retention=os.getenv("SCREENING_LOG_RETENTION", SCREENING_LOG_RETENTION),
            serialize=True,
            backtrace=False,
            diagnose=False,
        )
    )
    _configured = True


def bind_context(**fields: object):
    """Return a logger bound with sanitized structured fields.

    Args:
        **fields (object): Structured fields to attach to subsequent log calls.

    Returns:
        The shared Loguru logger with sanitized fields bound.
    """
    return logger.bind(**sanitize_log_fields(fields))


def sanitize_log_fields(fields: dict[str, object]) -> dict[str, object]:
    """Sanitize structured fields before they are attached to log records.

    Args:
        fields (dict[str, object]): Raw fields intended for a log record.

    Returns:
        dict[str, object]: JSON-serializable fields with sensitive payload keys
            redacted.
    """
    return {
        key: "<redacted>" if key in _SENSITIVE_FIELD_NAMES else _safe_value(value)
        for key, value in fields.items()
    }


def exception_metadata(error: BaseException) -> dict[str, object]:
    """Build safe exception metadata for structured logs.

    Args:
        error (BaseException): Exception to describe.

    Returns:
        dict[str, object]: Exception class, message, and Anthropic response
            metadata when available.
    """
    metadata: dict[str, object] = {
        "exception_type": error.__class__.__name__,
        "exception_message": _safe_exception_message(error),
    }

    status_code = getattr(error, "status_code", None)
    if status_code is not None:
        metadata["anthropic_status_code"] = status_code

    request_id = _request_id_from(error)
    if request_id is not None:
        metadata["anthropic_request_id"] = request_id

    return sanitize_log_fields(metadata)


def message_list_metadata(messages: list[object]) -> dict[str, int]:
    """Return count-only metadata for a message list.

    Args:
        messages (list[object]): Anthropic-style message objects.

    Returns:
        dict[str, int]: Message count and total character count, without
            including the message text itself.
    """
    return {
        "message_count": len(messages),
        "message_char_count": sum(
            _message_content_length(message) for message in messages
        ),
    }


def anthropic_response_metadata(response: object) -> dict[str, object]:
    """Extract safe metadata from an Anthropic response object.

    Args:
        response (object): Anthropic SDK response, or a test double.

    Returns:
        dict[str, object]: Message id, token usage, and request id when present.
    """
    metadata: dict[str, object] = {}

    message_id = getattr(response, "id", None)
    if message_id is not None:
        metadata["anthropic_message_id"] = message_id

    request_id = _request_id_from(response)
    if request_id is not None:
        metadata["anthropic_request_id"] = request_id

    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if input_tokens is not None:
        metadata["anthropic_input_tokens"] = input_tokens
    if output_tokens is not None:
        metadata["anthropic_output_tokens"] = output_tokens

    return sanitize_log_fields(metadata)


def profile_state_metadata(profile: object) -> dict[str, object]:
    """Return derived profile state that is safe for logs.

    Args:
        profile (object): CandidateProfile or compatible test double.

    Returns:
        dict[str, object]: Derived state fields only; no raw candidate values.
    """
    return sanitize_log_fields(
        {
            "is_complete": getattr(profile, "is_complete", None),
            "is_disqualified": getattr(profile, "is_disqualified", None),
            "missing_fields": getattr(profile, "missing_fields", None),
            "clarification_fields": getattr(profile, "clarification_fields", None),
            "disqualification_reasons": getattr(
                profile,
                "disqualification_reasons",
                None,
            ),
        }
    )


def _remove_configured_sinks() -> None:
    global _configured

    for sink_id in _sink_ids:
        try:
            logger.remove(sink_id)
        except ValueError:
            pass
    _sink_ids.clear()
    _configured = False


def _safe_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, set):
        return sorted(_safe_value(item) for item in value)
    if isinstance(value, tuple | list):
        return [_safe_value(item) for item in value]
    if isinstance(value, dict):
        return sanitize_log_fields({str(key): item for key, item in value.items()})
    return str(value)


def _safe_exception_message(error: BaseException) -> str:
    if error.__class__.__name__ == "ValidationError":
        return "Validation failed"
    message = str(error)
    return message if len(message) <= 500 else f"{message[:500]}..."


def _message_content_length(message: object) -> int:
    if not isinstance(message, dict):
        return 0
    content = message.get("content")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(str(block)) for block in content)
    return 0


def _request_id_from(value: object) -> str | None:
    for attribute in ("request_id", "_request_id"):
        request_id = getattr(value, attribute, None)
        if request_id is not None:
            return str(request_id)

    response = getattr(value, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    for header_name in ("request-id", "x-request-id", "anthropic-request-id"):
        try:
            request_id = headers.get(header_name)
        except AttributeError:
            return None
        if request_id is not None:
            return str(request_id)
    return None
