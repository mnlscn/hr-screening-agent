import sqlite3
from typing import cast
from uuid import UUID

from anthropic.types import MessageParam

from screening.models import CandidateProfile, DeliveryExperience
from screening.storage import (
    append_candidate_message,
    append_candidate_messages,
    create_candidate,
    init_database,
    load_agent_state,
    load_candidate,
    load_candidate_messages,
    list_candidates,
    mark_candidate_summary_pending,
    replace_candidate_messages,
    save_candidate_profile,
    save_candidate_summary,
    save_candidate_summary_failure,
    save_candidate_session,
)


def test_init_database_is_idempotent(tmp_path):
    db_path = tmp_path / "screening.sqlite3"

    init_database(db_path)
    init_database(db_path)

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        migrations = connection.execute(
            "SELECT version FROM schema_migrations"
        ).fetchall()

    assert {"candidates", "messages", "schema_migrations"}.issubset(tables)
    assert migrations == [(1,), (2,)]


def test_create_candidate_generates_id_and_empty_profile(tmp_path):
    db_path = tmp_path / "screening.sqlite3"

    candidate_id = create_candidate(db_path=db_path)

    UUID(candidate_id)
    loaded = load_candidate(candidate_id, db_path=db_path)

    assert loaded is not None
    assert loaded.id == candidate_id
    assert loaded.status == "active"
    assert loaded.hr_summary is None
    assert loaded.bot_label is None
    assert loaded.summary_status is None
    assert loaded.profile.model_dump(mode="json") == CandidateProfile().model_dump(
        mode="json"
    )


def test_list_candidates_returns_empty_list_for_empty_database(tmp_path):
    db_path = tmp_path / "screening.sqlite3"

    assert list_candidates(db_path=db_path) == []


def test_list_candidates_returns_newest_candidates_with_summary_fields(tmp_path):
    db_path = tmp_path / "screening.sqlite3"
    older_id = create_candidate(
        CandidateProfile(full_name="Older Candidate"),
        db_path=db_path,
    )
    newer_id = create_candidate(
        CandidateProfile(
            full_name="Newer Candidate",
            drivers_license="Yes",
            raw_city_zone="Madrid",
            city_zone="Madrid",
            city_zone_status="Matched",
            availability="Full-time",
            preferred_schedule="Morning",
            prior_delivery_experience=DeliveryExperience(years=1, platform="Glovo"),
            start_date="tomorrow",
        ),
        db_path=db_path,
    )
    save_candidate_summary(
        newer_id,
        db_path=db_path,
        hr_summary="Ready for HR review.",
        bot_label="eligible",
        model="claude-sonnet-4-6",
    )

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE candidates SET updated_at = ? WHERE id = ?",
            ("2026-01-01T00:00:00+00:00", older_id),
        )
        connection.execute(
            "UPDATE candidates SET updated_at = ? WHERE id = ?",
            ("2026-01-02T00:00:00+00:00", newer_id),
        )

    candidates = list_candidates(db_path=db_path)

    assert [candidate.id for candidate in candidates] == [newer_id, older_id]
    assert candidates[0].profile.full_name == "Newer Candidate"
    assert candidates[0].profile.city_zone == "Madrid"
    assert candidates[0].hr_summary == "Ready for HR review."
    assert candidates[0].bot_label == "eligible"
    assert candidates[0].summary_status == "completed"
    assert candidates[0].updated_at == "2026-01-02T00:00:00+00:00"


def test_candidate_profile_round_trips_and_updates_queryable_columns(tmp_path):
    db_path = tmp_path / "screening.sqlite3"
    candidate_id = create_candidate(db_path=db_path)
    profile = CandidateProfile(
        full_name="Maria Garcia",
        raw_drivers_license="yes",
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

    save_candidate_profile(candidate_id, profile, db_path=db_path)

    loaded = load_candidate(candidate_id, db_path=db_path)
    assert loaded is not None
    assert loaded.status == "completed"
    assert loaded.completed_at is not None
    assert loaded.profile.model_dump(mode="json") == profile.model_dump(mode="json")

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT full_name,
                   city_zone,
                   delivery_experience_years,
                   delivery_experience_platform,
                   is_complete
            FROM candidates
            WHERE id = ?
            """,
            (candidate_id,),
        ).fetchone()

    assert row == ("Maria Garcia", "Madrid", 2.0, "Glovo", 1)


def test_messages_preserve_order_when_replaced_and_appended(tmp_path):
    db_path = tmp_path / "screening.sqlite3"
    candidate_id = create_candidate(db_path=db_path)
    messages = [
        cast(MessageParam, {"role": "assistant", "content": "Hola"}),
        cast(MessageParam, {"role": "user", "content": "Estoy lista"}),
    ]

    replace_candidate_messages(candidate_id, messages, db_path=db_path)

    assert load_candidate_messages(candidate_id, db_path=db_path) == messages

    replacement = [cast(MessageParam, {"role": "user", "content": "Resume"})]
    replace_candidate_messages(candidate_id, replacement, db_path=db_path)
    append_candidate_message(
        candidate_id,
        cast(MessageParam, {"role": "assistant", "content": "Seguimos"}),
        db_path=db_path,
    )
    append_candidate_messages(
        candidate_id,
        [cast(MessageParam, {"role": "user", "content": "Vale"})],
        db_path=db_path,
    )

    assert load_candidate_messages(candidate_id, db_path=db_path) == [
        cast(MessageParam, {"role": "user", "content": "Resume"}),
        cast(MessageParam, {"role": "assistant", "content": "Seguimos"}),
        cast(MessageParam, {"role": "user", "content": "Vale"}),
    ]


def test_save_candidate_session_creates_loadable_agent_state(tmp_path):
    db_path = tmp_path / "screening.sqlite3"
    profile = CandidateProfile(full_name="Luis Perez")
    messages = [
        cast(MessageParam, {"role": "assistant", "content": "Hola"}),
        cast(MessageParam, {"role": "user", "content": "Soy Luis Perez"}),
    ]

    saved = save_candidate_session(profile, messages, db_path=db_path)
    state = load_agent_state(saved.candidate_id, db_path=db_path)

    assert state is not None
    assert state.candidate_id == saved.candidate_id
    assert state.profile.model_dump(mode="json") == profile.model_dump(mode="json")
    assert state.messages == messages
    assert state.status == "active"


def test_candidate_summary_fields_round_trip(tmp_path):
    db_path = tmp_path / "screening.sqlite3"
    candidate_id = create_candidate(db_path=db_path)

    mark_candidate_summary_pending(
        candidate_id,
        db_path=db_path,
        model="claude-sonnet-4-6",
    )
    pending = load_candidate(candidate_id, db_path=db_path)

    assert pending is not None
    assert pending.summary_status == "pending"
    assert pending.summary_model == "claude-sonnet-4-6"

    save_candidate_summary(
        candidate_id,
        db_path=db_path,
        hr_summary="Maria has a complete profile and replied clearly.",
        bot_label="eligible",
        model="claude-sonnet-4-6",
    )
    loaded = load_candidate(candidate_id, db_path=db_path)

    assert loaded is not None
    assert loaded.hr_summary == "Maria has a complete profile and replied clearly."
    assert loaded.bot_label == "eligible"
    assert loaded.summary_status == "completed"
    assert loaded.summary_model == "claude-sonnet-4-6"
    assert loaded.summary_error is None
    assert loaded.summary_generated_at is not None


def test_candidate_summary_failure_sets_needs_review(tmp_path):
    db_path = tmp_path / "screening.sqlite3"
    candidate_id = create_candidate(db_path=db_path)

    save_candidate_summary_failure(
        candidate_id,
        db_path=db_path,
        error="invalid JSON",
        model="claude-sonnet-4-6",
    )
    loaded = load_candidate(candidate_id, db_path=db_path)

    assert loaded is not None
    assert loaded.bot_label == "needs_review"
    assert loaded.summary_status == "failed"
    assert loaded.summary_error == "invalid JSON"
    assert loaded.summary_generated_at is not None


def test_disqualified_profile_does_not_mark_candidate_disqualified(tmp_path):
    db_path = tmp_path / "screening.sqlite3"
    candidate_id = create_candidate(db_path=db_path)
    profile = CandidateProfile(drivers_license="No")

    save_candidate_profile(candidate_id, profile, db_path=db_path)
    loaded = load_candidate(candidate_id, db_path=db_path)

    assert loaded is not None
    assert loaded.profile.is_disqualified
    assert loaded.status == "active"


def test_version_one_database_migrates_summary_columns(tmp_path):
    db_path = tmp_path / "screening.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );
            INSERT INTO schema_migrations (version, applied_at)
            VALUES (1, '2026-01-01T00:00:00+00:00');

            CREATE TABLE candidates (
                id TEXT PRIMARY KEY,
                full_name TEXT,
                raw_drivers_license TEXT,
                drivers_license TEXT,
                raw_city_zone TEXT,
                city_zone TEXT,
                city_zone_status TEXT,
                conversation_language TEXT,
                availability TEXT,
                preferred_schedule TEXT,
                delivery_experience_years REAL,
                delivery_experience_platform TEXT,
                start_date TEXT,
                is_complete INTEGER NOT NULL,
                is_disqualified INTEGER NOT NULL,
                missing_fields TEXT NOT NULL,
                clarification_fields TEXT NOT NULL,
                disqualification_reasons TEXT NOT NULL,
                profile_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN ('active', 'completed', 'disqualified')
                ),
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL
                    REFERENCES candidates(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('assistant', 'user')),
                content_text TEXT NOT NULL,
                message_json TEXT NOT NULL,
                UNIQUE (candidate_id, position)
            );
            """
        )

    init_database(db_path)

    with sqlite3.connect(db_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(candidates)")
        }
        migrations = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()

    assert {
        "hr_summary",
        "bot_label",
        "summary_status",
        "summary_model",
        "summary_error",
        "summary_generated_at",
    }.issubset(columns)
    assert migrations == [(1,), (2,)]
