from typing import Any, cast

from anthropic.types import MessageParam

from screening.main import finalize_candidate_session
from screening.models import CandidateProfile, DeliveryExperience
from screening.storage import create_candidate, load_candidate, load_candidate_messages
from screening.summary import CandidateSummary


class FakeAgent:
    def __init__(self, candidate_id: str):
        self.candidate_id = candidate_id
        self.profile = CandidateProfile(
            full_name="Maria Garcia",
            drivers_license="Yes",
            raw_city_zone="Madrid",
            city_zone="Madrid",
            city_zone_status="Matched",
            conversation_language="Spanish",
            availability="Full-time",
            preferred_schedule="Morning",
            prior_delivery_experience=DeliveryExperience(years=2, platform="Glovo"),
            start_date="next Monday",
        )
        self.messages = [
            cast(MessageParam, {"role": "assistant", "content": "Hola"}),
            cast(MessageParam, {"role": "user", "content": "Soy Maria Garcia"}),
        ]
        self.client = object()
        self.last_extraction_error = None


class FakeSummarizer:
    def __init__(self):
        self.seen_messages: list[MessageParam] | None = None

    def summarize(self, profile, transcript, *, extraction_failed=False):
        self.seen_messages = list(transcript)
        return CandidateSummary(
            bot_label="eligible",
            hr_summary="Maria is complete, polite, and ready for HR review.",
        )


class FailingSummarizer:
    def summarize(self, profile, transcript, *, extraction_failed=False):
        raise ValueError("bad summary JSON")


def test_finalize_candidate_session_saves_before_summarizing(tmp_path):
    db_path = tmp_path / "screening.sqlite3"
    candidate_id = create_candidate(db_path=db_path)
    agent = FakeAgent(candidate_id)
    summarizer = FakeSummarizer()

    finalized = finalize_candidate_session(
        cast(Any, agent),
        db_path=db_path,
        summarizer=cast(Any, summarizer),
    )
    loaded = load_candidate(candidate_id, db_path=db_path)

    assert finalized.summary_status == "completed"
    assert loaded is not None
    assert loaded.status == "completed"
    assert loaded.bot_label == "eligible"
    assert loaded.summary_status == "completed"
    assert loaded.hr_summary == "Maria is complete, polite, and ready for HR review."
    assert load_candidate_messages(candidate_id, db_path=db_path) == agent.messages
    assert summarizer.seen_messages == agent.messages


def test_finalize_candidate_session_records_summary_failure(tmp_path):
    db_path = tmp_path / "screening.sqlite3"
    candidate_id = create_candidate(db_path=db_path)
    agent = FakeAgent(candidate_id)

    finalized = finalize_candidate_session(
        cast(Any, agent),
        db_path=db_path,
        summarizer=cast(Any, FailingSummarizer()),
    )
    loaded = load_candidate(candidate_id, db_path=db_path)

    assert finalized.summary_status == "failed"
    assert loaded is not None
    assert loaded.status == "completed"
    assert loaded.bot_label == "needs_review"
    assert loaded.summary_status == "failed"
    assert loaded.summary_error == "bad summary JSON"
