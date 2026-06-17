import json
from datetime import UTC, datetime
from pathlib import Path

from screening.config import CANDIDATES_JSONL_PATH
from screening.models import CandidateProfile


def append_candidate_session(
    profile: CandidateProfile,
    transcript: list[dict[str, str]],
    path: str = CANDIDATES_JSONL_PATH,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "profile": profile.model_dump(mode="json"),
        "transcript": transcript,
        "is_complete": profile.is_complete,
        "is_disqualified": profile.is_disqualified,
        "disqualification_reasons": profile.disqualification_reasons,
    }

    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True))
        file.write("\n")

    return output_path

