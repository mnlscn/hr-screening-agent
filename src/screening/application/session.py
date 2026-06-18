"""Candidate session lifecycle: start, save, resume, and finalize screenings."""

import os
from dataclasses import dataclass
from pathlib import Path

from screening.llm.agent import ChatAgent
from screening.config import ANTHROPIC_API_KEY_ENV, SCREENING_DB_PATH, SUMMARY_MODEL
from screening.domain.models import ACTIVE_STATUS, COMPLETED_STATUS
from screening.persistence.storage import (
    SavedCandidateSession,
    create_candidate,
    mark_candidate_summary_pending,
    save_candidate_session,
    save_candidate_summary,
    save_candidate_summary_failure,
)
from screening.llm.summary import CandidateSummarizer


@dataclass(frozen=True)
class FinalizedCandidateSession:
    """Result of finalizing a candidate screening session.

    Attributes:
        candidate_id (str): Identifier of the finalized candidate.
        db_path (Path): Path to the database where the session was persisted.
        summary_status (str): Outcome of summary generation, either "completed"
            or "failed".
    """

    candidate_id: str
    db_path: Path
    summary_status: str


def start_candidate_session(
    *,
    api_key: str | None = None,
    db_path: str | Path = SCREENING_DB_PATH,
) -> ChatAgent:
    """Create a new candidate record and return a started chat agent.

    Args:
        api_key (str | None): Anthropic API key; falls back to the
            ``ANTHROPIC_API_KEY`` environment variable when None.
        db_path (str | Path): Path to the screening database.

    Returns:
        ChatAgent: A started agent bound to the new candidate, already saved.
    """
    candidate_id = create_candidate(db_path=db_path)
    agent = ChatAgent(
        api_key=_resolve_api_key(api_key),
        candidate_id=candidate_id,
    )
    agent.start()
    save_current_session(agent, db_path=db_path)
    return agent


def resume_candidate_session(
    candidate_id: str,
    *,
    api_key: str | None = None,
    db_path: str | Path = SCREENING_DB_PATH,
) -> ChatAgent:
    """Rebuild a chat agent from a previously stored candidate session.

    Args:
        candidate_id (str): Identifier of the candidate to resume.
        api_key (str | None): Anthropic API key; falls back to the
            ``ANTHROPIC_API_KEY`` environment variable when None.
        db_path (str | Path): Path to the screening database.

    Returns:
        ChatAgent: An agent restored with the candidate's profile and
            transcript.
    """
    return ChatAgent.from_candidate_id(
        candidate_id,
        api_key=_resolve_api_key(api_key),
        db_path=db_path,
    )


def save_current_session(
    agent: ChatAgent,
    *,
    db_path: str | Path = SCREENING_DB_PATH,
) -> SavedCandidateSession:
    """Persist the agent's current profile and transcript as an active session.

    Updates ``agent.candidate_id`` with the identifier returned by storage.

    Args:
        agent (ChatAgent): The agent whose state should be saved.
        db_path (str | Path): Path to the screening database.

    Returns:
        SavedCandidateSession: The saved candidate identifier and database path.
    """
    saved_session = save_candidate_session(
        profile=agent.profile,
        transcript=agent.messages,
        candidate_id=agent.candidate_id,
        db_path=db_path,
        status=ACTIVE_STATUS,
    )
    agent.candidate_id = saved_session.candidate_id
    return saved_session


def finalize_candidate_session(
    agent: ChatAgent,
    *,
    db_path: str | Path = SCREENING_DB_PATH,
    summarizer: CandidateSummarizer | None = None,
) -> FinalizedCandidateSession:
    """Complete a screening: persist it and generate the HR summary.

    Saves the session with completed status, then generates and stores the
    candidate summary.

    Args:
        agent (ChatAgent): The agent whose session should be finalized.
        db_path (str | Path): Path to the screening database.
        summarizer (CandidateSummarizer | None): Summarizer to use; a default
            one backed by the agent's client is created when None.

    Returns:
        FinalizedCandidateSession: The finalized candidate identifier, database
            path, and summary status.
    """
    saved_session = save_candidate_session(
        profile=agent.profile,
        transcript=agent.messages,
        candidate_id=agent.candidate_id,
        db_path=db_path,
        status=COMPLETED_STATUS,
    )
    agent.candidate_id = saved_session.candidate_id
    summary_status = _generate_and_store_summary(
        agent,
        saved_session,
        summarizer=summarizer,
    )
    return FinalizedCandidateSession(
        candidate_id=saved_session.candidate_id,
        db_path=saved_session.db_path,
        summary_status=summary_status,
    )


def _generate_and_store_summary(
    agent: ChatAgent,
    saved_session: SavedCandidateSession,
    *,
    summarizer: CandidateSummarizer | None = None,
) -> str:
    """Generate the candidate summary and persist it or its failure.

    Marks the summary as pending, attempts generation, and stores either the
    completed summary or a failure record.

    Args:
        agent (ChatAgent): The agent providing the profile and transcript.
        saved_session (SavedCandidateSession): The persisted session to attach
            the summary to.
        summarizer (CandidateSummarizer | None): Summarizer to use; a default
            one backed by the agent's client is created when None.

    Returns:
        str: "completed" when the summary was stored, or "failed" when
            generation raised an error.
    """
    mark_candidate_summary_pending(
        saved_session.candidate_id,
        db_path=saved_session.db_path,
        model=SUMMARY_MODEL,
    )
    summarizer = summarizer or CandidateSummarizer(agent.client)

    try:
        summary = summarizer.summarize(
            agent.profile,
            agent.messages,
            extraction_failed=agent.last_extraction_error is not None,
        )
    except Exception as error:
        save_candidate_summary_failure(
            saved_session.candidate_id,
            error=str(error),
            model=SUMMARY_MODEL,
            db_path=saved_session.db_path,
        )
        return "failed"

    save_candidate_summary(
        saved_session.candidate_id,
        hr_summary=summary.hr_summary,
        bot_label=summary.bot_label,
        model=summary.model,
        db_path=saved_session.db_path,
    )
    return "completed"


def _resolve_api_key(api_key: str | None) -> str | None:
    """Resolve the API key, falling back to the environment variable.

    Args:
        api_key (str | None): Explicitly provided API key, if any.

    Returns:
        str | None: The provided key, or the ``ANTHROPIC_API_KEY`` environment
            variable value when none was given.
    """
    return api_key if api_key is not None else os.getenv(ANTHROPIC_API_KEY_ENV)
