"""Tests for privacy-aware Loguru configuration."""

import json
from pathlib import Path

from screening.observability import configure_logging, logger


def test_configure_logging_writes_jsonl_and_is_idempotent(tmp_path: Path):
    log_path = tmp_path / "screening.jsonl"

    configure_logging(log_path=log_path, include_stderr=False, force=True)
    configure_logging(log_path=log_path, include_stderr=False)

    logger.bind(event="observability_test", candidate_id="candidate-1").info(
        "Observability test event"
    )

    records = _read_records(log_path)
    assert len(records) == 1
    assert records[0]["message"] == "Observability test event"
    assert records[0]["extra"]["event"] == "observability_test"
    assert records[0]["extra"]["candidate_id"] == "candidate-1"


def _read_records(log_path: Path) -> list[dict]:
    return [
        json.loads(line)["record"]
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
