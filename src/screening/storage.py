import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

from anthropic.types import MessageParam

from screening.config import SCREENING_DB_PATH
from screening.models import CandidateProfile


SCHEMA_VERSION = 2
ACTIVE_STATUS = "active"
COMPLETED_STATUS = "completed"
DISQUALIFIED_STATUS = "disqualified"
VALID_STATUSES = {ACTIVE_STATUS, COMPLETED_STATUS, DISQUALIFIED_STATUS}
FINAL_STATUSES = {COMPLETED_STATUS, DISQUALIFIED_STATUS}
PENDING_SUMMARY_STATUS = "pending"
COMPLETED_SUMMARY_STATUS = "completed"
FAILED_SUMMARY_STATUS = "failed"
VALID_SUMMARY_STATUSES = {
    PENDING_SUMMARY_STATUS,
    COMPLETED_SUMMARY_STATUS,
    FAILED_SUMMARY_STATUS,
}
VALID_BOT_LABELS = {"eligible", "not_eligible", "needs_review"}


@dataclass(frozen=True)
class StoredCandidate:
    id: str
    profile: CandidateProfile
    status: str
    started_at: str
    updated_at: str
    completed_at: str | None
    hr_summary: str | None
    bot_label: str | None
    summary_status: str | None
    summary_model: str | None
    summary_error: str | None
    summary_generated_at: str | None


@dataclass(frozen=True)
class CandidateConversationState:
    candidate_id: str
    profile: CandidateProfile
    messages: list[MessageParam]
    status: str
    started_at: str
    updated_at: str
    completed_at: str | None
    hr_summary: str | None
    bot_label: str | None
    summary_status: str | None
    summary_model: str | None
    summary_error: str | None
    summary_generated_at: str | None


@dataclass(frozen=True)
class SavedCandidateSession:
    candidate_id: str
    db_path: Path


def init_database(db_path: str | Path = SCREENING_DB_PATH) -> Path:
    database_path = Path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with _connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS candidates (
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
                completed_at TEXT,
                hr_summary TEXT,
                bot_label TEXT,
                summary_status TEXT,
                summary_model TEXT,
                summary_error TEXT,
                summary_generated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL
                    REFERENCES candidates(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('assistant', 'user')),
                content_text TEXT NOT NULL,
                message_json TEXT NOT NULL,
                UNIQUE (candidate_id, position)
            );

            CREATE INDEX IF NOT EXISTS idx_candidates_status
                ON candidates(status);
            CREATE INDEX IF NOT EXISTS idx_candidates_city_zone
                ON candidates(city_zone);
            CREATE INDEX IF NOT EXISTS idx_candidates_is_complete
                ON candidates(is_complete);
            CREATE INDEX IF NOT EXISTS idx_candidates_is_disqualified
                ON candidates(is_disqualified);
            CREATE INDEX IF NOT EXISTS idx_messages_candidate_position
                ON messages(candidate_id, position);
            """
        )
        _migrate_database(connection)

    return database_path


def create_candidate(
    profile: CandidateProfile | None = None,
    *,
    db_path: str | Path = SCREENING_DB_PATH,
    candidate_id: str | None = None,
    status: str | None = None,
) -> str:
    database_path = init_database(db_path)
    profile = profile or CandidateProfile()
    candidate_id = candidate_id or str(uuid4())
    now = _utc_now()
    candidate_status = _normalize_status(status, profile)
    columns = {
        "id": candidate_id,
        **_profile_columns(profile),
        "status": candidate_status,
        "started_at": now,
        "updated_at": now,
        "completed_at": now if candidate_status in FINAL_STATUSES else None,
    }

    column_names = ", ".join(columns)
    placeholders = ", ".join(f":{name}" for name in columns)
    with _connect(database_path) as connection:
        connection.execute(
            f"INSERT INTO candidates ({column_names}) VALUES ({placeholders})",
            columns,
        )

    return candidate_id


def load_candidate(
    candidate_id: str,
    *,
    db_path: str | Path = SCREENING_DB_PATH,
) -> StoredCandidate | None:
    database_path = init_database(db_path)
    with _connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT id,
                   profile_json,
                   status,
                   started_at,
                   updated_at,
                   completed_at,
                   hr_summary,
                   bot_label,
                   summary_status,
                   summary_model,
                   summary_error,
                   summary_generated_at
            FROM candidates
            WHERE id = ?
            """,
            (candidate_id,),
        ).fetchone()

    if row is None:
        return None

    return StoredCandidate(
        id=str(row["id"]),
        profile=CandidateProfile.model_validate(json.loads(row["profile_json"])),
        status=str(row["status"]),
        started_at=str(row["started_at"]),
        updated_at=str(row["updated_at"]),
        completed_at=cast(str | None, row["completed_at"]),
        hr_summary=cast(str | None, row["hr_summary"]),
        bot_label=cast(str | None, row["bot_label"]),
        summary_status=cast(str | None, row["summary_status"]),
        summary_model=cast(str | None, row["summary_model"]),
        summary_error=cast(str | None, row["summary_error"]),
        summary_generated_at=cast(str | None, row["summary_generated_at"]),
    )


def save_candidate_profile(
    candidate_id: str,
    profile: CandidateProfile,
    *,
    db_path: str | Path = SCREENING_DB_PATH,
    status: str | None = None,
) -> None:
    database_path = init_database(db_path)
    candidate_status = _normalize_status(status, profile)
    now = _utc_now()
    completed_at = now if candidate_status in FINAL_STATUSES else None
    columns = {
        **_profile_columns(profile),
        "status": candidate_status,
        "updated_at": now,
        "completed_at": completed_at,
        "candidate_id": candidate_id,
    }

    with _connect(database_path) as connection:
        result = connection.execute(
            """
            UPDATE candidates
            SET full_name = :full_name,
                raw_drivers_license = :raw_drivers_license,
                drivers_license = :drivers_license,
                raw_city_zone = :raw_city_zone,
                city_zone = :city_zone,
                city_zone_status = :city_zone_status,
                conversation_language = :conversation_language,
                availability = :availability,
                preferred_schedule = :preferred_schedule,
                delivery_experience_years = :delivery_experience_years,
                delivery_experience_platform = :delivery_experience_platform,
                start_date = :start_date,
                is_complete = :is_complete,
                is_disqualified = :is_disqualified,
                missing_fields = :missing_fields,
                clarification_fields = :clarification_fields,
                disqualification_reasons = :disqualification_reasons,
                profile_json = :profile_json,
                status = :status,
                updated_at = :updated_at,
                completed_at = CASE
                    WHEN :completed_at IS NULL THEN NULL
                    ELSE COALESCE(completed_at, :completed_at)
                END
            WHERE id = :candidate_id
            """,
            columns,
        )

    if result.rowcount == 0:
        raise KeyError(f"candidate not found: {candidate_id}")


def replace_candidate_messages(
    candidate_id: str,
    messages: list[MessageParam],
    *,
    db_path: str | Path = SCREENING_DB_PATH,
) -> None:
    database_path = init_database(db_path)
    with _connect(database_path) as connection:
        _require_candidate(connection, candidate_id)
        connection.execute(
            "DELETE FROM messages WHERE candidate_id = ?", (candidate_id,)
        )
        for position, message in enumerate(messages):
            _insert_message(connection, candidate_id, position, message)


def append_candidate_message(
    candidate_id: str,
    message: MessageParam,
    *,
    db_path: str | Path = SCREENING_DB_PATH,
) -> None:
    append_candidate_messages(candidate_id, [message], db_path=db_path)


def append_candidate_messages(
    candidate_id: str,
    messages: list[MessageParam],
    *,
    db_path: str | Path = SCREENING_DB_PATH,
) -> None:
    if not messages:
        return

    database_path = init_database(db_path)
    with _connect(database_path) as connection:
        _require_candidate(connection, candidate_id)
        next_position = connection.execute(
            """
            SELECT COALESCE(MAX(position) + 1, 0)
            FROM messages
            WHERE candidate_id = ?
            """,
            (candidate_id,),
        ).fetchone()[0]
        for offset, message in enumerate(messages):
            _insert_message(
                connection, candidate_id, int(next_position) + offset, message
            )


def load_candidate_messages(
    candidate_id: str,
    *,
    db_path: str | Path = SCREENING_DB_PATH,
) -> list[MessageParam]:
    database_path = init_database(db_path)
    with _connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT message_json
            FROM messages
            WHERE candidate_id = ?
            ORDER BY position
            """,
            (candidate_id,),
        ).fetchall()

    return [cast(MessageParam, json.loads(row["message_json"])) for row in rows]


def load_agent_state(
    candidate_id: str,
    *,
    db_path: str | Path = SCREENING_DB_PATH,
) -> CandidateConversationState | None:
    candidate = load_candidate(candidate_id, db_path=db_path)
    if candidate is None:
        return None

    return CandidateConversationState(
        candidate_id=candidate.id,
        profile=candidate.profile,
        messages=load_candidate_messages(candidate.id, db_path=db_path),
        status=candidate.status,
        started_at=candidate.started_at,
        updated_at=candidate.updated_at,
        completed_at=candidate.completed_at,
        hr_summary=candidate.hr_summary,
        bot_label=candidate.bot_label,
        summary_status=candidate.summary_status,
        summary_model=candidate.summary_model,
        summary_error=candidate.summary_error,
        summary_generated_at=candidate.summary_generated_at,
    )


def save_candidate_session(
    profile: CandidateProfile,
    transcript: list[MessageParam],
    *,
    candidate_id: str | None = None,
    db_path: str | Path = SCREENING_DB_PATH,
    status: str | None = None,
) -> SavedCandidateSession:
    database_path = init_database(db_path)
    if candidate_id is None:
        candidate_id = create_candidate(
            profile,
            db_path=database_path,
            status=status,
        )
    else:
        save_candidate_profile(
            candidate_id,
            profile,
            db_path=database_path,
            status=status,
        )

    replace_candidate_messages(candidate_id, transcript, db_path=database_path)
    return SavedCandidateSession(candidate_id=candidate_id, db_path=database_path)


def mark_candidate_summary_pending(
    candidate_id: str,
    *,
    db_path: str | Path = SCREENING_DB_PATH,
    model: str | None = None,
) -> None:
    database_path = init_database(db_path)
    now = _utc_now()

    with _connect(database_path) as connection:
        result = connection.execute(
            """
            UPDATE candidates
            SET summary_status = ?,
                summary_model = ?,
                summary_error = NULL,
                summary_generated_at = NULL,
                updated_at = ?
            WHERE id = ?
            """,
            (PENDING_SUMMARY_STATUS, model, now, candidate_id),
        )

    if result.rowcount == 0:
        raise KeyError(f"candidate not found: {candidate_id}")


def save_candidate_summary(
    candidate_id: str,
    *,
    hr_summary: str,
    bot_label: str,
    model: str,
    db_path: str | Path = SCREENING_DB_PATH,
) -> None:
    if bot_label not in VALID_BOT_LABELS:
        raise ValueError(f"invalid bot label: {bot_label}")

    database_path = init_database(db_path)
    now = _utc_now()
    with _connect(database_path) as connection:
        result = connection.execute(
            """
            UPDATE candidates
            SET hr_summary = ?,
                bot_label = ?,
                summary_status = ?,
                summary_model = ?,
                summary_error = NULL,
                summary_generated_at = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                hr_summary,
                bot_label,
                COMPLETED_SUMMARY_STATUS,
                model,
                now,
                now,
                candidate_id,
            ),
        )

    if result.rowcount == 0:
        raise KeyError(f"candidate not found: {candidate_id}")


def save_candidate_summary_failure(
    candidate_id: str,
    *,
    error: str,
    model: str | None,
    db_path: str | Path = SCREENING_DB_PATH,
) -> None:
    database_path = init_database(db_path)
    now = _utc_now()
    with _connect(database_path) as connection:
        result = connection.execute(
            """
            UPDATE candidates
            SET bot_label = ?,
                summary_status = ?,
                summary_model = ?,
                summary_error = ?,
                summary_generated_at = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                "needs_review",
                FAILED_SUMMARY_STATUS,
                model,
                error,
                now,
                now,
                candidate_id,
            ),
        )

    if result.rowcount == 0:
        raise KeyError(f"candidate not found: {candidate_id}")


def _connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _migrate_database(connection: sqlite3.Connection) -> None:
    applied_versions = {
        int(row["version"])
        for row in connection.execute("SELECT version FROM schema_migrations")
    }

    if 1 not in applied_versions:
        _record_schema_migration(connection, 1)

    if 2 not in applied_versions:
        _migrate_to_v2(connection)
        _record_schema_migration(connection, 2)


def _record_schema_migration(
    connection: sqlite3.Connection,
    version: int,
) -> None:
    connection.execute(
        """
        INSERT OR IGNORE INTO schema_migrations (version, applied_at)
        VALUES (?, ?)
        """,
        (version, _utc_now()),
    )


def _migrate_to_v2(connection: sqlite3.Connection) -> None:
    existing_columns = {
        str(row["name"]) for row in connection.execute("PRAGMA table_info(candidates)")
    }
    summary_columns = {
        "hr_summary": "TEXT",
        "bot_label": "TEXT",
        "summary_status": "TEXT",
        "summary_model": "TEXT",
        "summary_error": "TEXT",
        "summary_generated_at": "TEXT",
    }
    for column_name, column_type in summary_columns.items():
        if column_name not in existing_columns:
            connection.execute(
                f"ALTER TABLE candidates ADD COLUMN {column_name} {column_type}"
            )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _normalize_status(status: str | None, profile: CandidateProfile) -> str:
    candidate_status = status or _status_from_profile(profile)
    if candidate_status not in VALID_STATUSES:
        raise ValueError(f"invalid candidate status: {candidate_status}")
    return candidate_status


def _status_from_profile(profile: CandidateProfile) -> str:
    if profile.is_complete:
        return COMPLETED_STATUS
    return ACTIVE_STATUS


def _profile_columns(profile: CandidateProfile) -> dict[str, object]:
    experience = profile.prior_delivery_experience
    return {
        "full_name": profile.full_name,
        "raw_drivers_license": profile.raw_drivers_license,
        "drivers_license": profile.drivers_license,
        "raw_city_zone": profile.raw_city_zone,
        "city_zone": profile.city_zone,
        "city_zone_status": profile.city_zone_status,
        "conversation_language": profile.conversation_language,
        "availability": profile.availability,
        "preferred_schedule": profile.preferred_schedule,
        "delivery_experience_years": experience.years if experience else None,
        "delivery_experience_platform": experience.platform if experience else None,
        "start_date": profile.start_date,
        "is_complete": int(profile.is_complete),
        "is_disqualified": int(profile.is_disqualified),
        "missing_fields": _json_dumps(profile.missing_fields),
        "clarification_fields": _json_dumps(profile.clarification_fields),
        "disqualification_reasons": _json_dumps(profile.disqualification_reasons),
        "profile_json": _json_dumps(profile.model_dump(mode="json")),
    }


def _require_candidate(connection: sqlite3.Connection, candidate_id: str) -> None:
    row = connection.execute(
        "SELECT 1 FROM candidates WHERE id = ?",
        (candidate_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"candidate not found: {candidate_id}")


def _insert_message(
    connection: sqlite3.Connection,
    candidate_id: str,
    position: int,
    message: MessageParam,
) -> None:
    role = str(message["role"])
    if role not in {"assistant", "user"}:
        raise ValueError(f"invalid message role: {role}")

    content = message["content"]
    content_text = content if isinstance(content, str) else _json_dumps(content)
    connection.execute(
        """
        INSERT INTO messages (
            candidate_id,
            position,
            role,
            content_text,
            message_json
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            candidate_id,
            position,
            role,
            content_text,
            _json_dumps(message),
        ),
    )
