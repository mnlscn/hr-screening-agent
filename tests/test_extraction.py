"""Tests for CandidateExtractor: LLM response parsing and profile updates."""

import json
from typing import Any, cast

from screening.llm.extraction import CandidateExtractor
from screening.domain.models import CandidateProfile


def test_extractor_output_shape_includes_conversation_language():
    extractor = CandidateExtractor(client=cast(Any, object()))

    payload = json.loads(
        extractor._build_prompt(
            messages=[{"role": "user", "content": "I would prefer English"}],
            current_profile=CandidateProfile(),
        )
    )

    assert payload["required_output_shape"]["conversation_language"] is None
