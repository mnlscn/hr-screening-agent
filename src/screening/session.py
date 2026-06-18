import os
from dataclasses import dataclass
from pathlib import Path

from screening.agent import ChatAgent
from screening.config import ANTHROPIC_API_KEY_ENV, SCREENING_DB_PATH, SUMMARY_MODEL
from screening.storage import (
    ACTIVE_STATUS,
    COMPLETED_STATUS,
    SavedCandidateSession,
    create_candidate,
    mark_candidate_summary_pending,
    save_candidate_session,
    save_candidate_summary,
    save_candidate_summary_failure,
)
from screening.summary import CandidateSummarizer


@dataclass(frozen=True)
class FinalizedCandidateSession:
    candidate_id: str
    db_path: Path
    summary_status: str


def start_candidate_session(
    *,
    api_key: str | None = None,
    db_path: str | Path = SCREENING_DB_PATH,
) -> ChatAgent:
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
    return api_key if api_key is not None else os.getenv(ANTHROPIC_API_KEY_ENV)
