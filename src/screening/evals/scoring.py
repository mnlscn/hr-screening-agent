"""Pure scoring and case loading for extraction-quality evals."""

from __future__ import annotations

import io
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

from anthropic.types import MessageParam
from rich.console import Console, Group, RenderableType
from rich.table import Table
from rich.text import Text

from screening.domain.models import CandidateProfile


SOFT_FIELDS = frozenset(
    {
        "full_name",
        "start_date",
        "prior_delivery_experience.platform",
    }
)
NUMERIC_FIELDS = frozenset({"prior_delivery_experience.years"})
CATEGORICAL_FIELDS = frozenset(
    {
        "drivers_license",
        "full_name_status",
        "city_zone",
        "city_zone_status",
        "availability",
        "availability_status",
        "preferred_schedule",
        "preferred_schedule_status",
        "prior_delivery_experience_status",
        "start_date_status",
        "conversation_language",
    }
)
DERIVED_FIELDS = frozenset(
    {
        "is_complete",
        "is_disqualified",
        "disqualification_reasons",
        "missing_fields",
        "clarification_fields",
    }
)
RAW_FIELD_PREFIX = "raw_"
NESTED_EXPERIENCE_FIELDS = frozenset({"years", "platform"})
ALLOWED_EXPECTED_FIELDS = (
    CATEGORICAL_FIELDS
    | SOFT_FIELDS
    | NUMERIC_FIELDS
    | frozenset({"prior_delivery_experience"})
)


@dataclass(frozen=True)
class EvalCase:
    """A single golden extraction case loaded from JSONL."""

    id: str
    bucket: str
    transcript: list[MessageParam]
    expected: dict[str, Any]
    current_profile: CandidateProfile


@dataclass(frozen=True)
class FieldResult:
    """Result for one asserted field in one eval case."""

    case_id: str
    bucket: str
    field: str
    expected: Any
    got: Any
    correct: bool
    soft: bool = False
    error: str | None = None


@dataclass(frozen=True)
class CaseResult:
    """Scored result for one eval case run."""

    case_id: str
    bucket: str
    fields: tuple[FieldResult, ...]
    error: str | None = None

    @property
    def passed(self) -> bool:
        """Return True when every asserted field matched and no error occurred."""
        return self.error is None and all(field.correct for field in self.fields)


@dataclass(frozen=True)
class Accuracy:
    """Correct/total accuracy for a group of assertions."""

    correct: int
    total: int

    @property
    def rate(self) -> float:
        """Return accuracy as a float from 0.0 to 1.0."""
        if self.total == 0:
            return 0.0
        return self.correct / self.total


@dataclass(frozen=True)
class EvalSummary:
    """Aggregated extraction-eval metrics."""

    case_count: int
    passed_cases: int
    field_accuracy: dict[str, Accuracy]
    bucket_accuracy: dict[str, Accuracy]
    overall_accuracy: Accuracy
    confusion_matrices: dict[str, dict[str, dict[str, int]]]
    failures: tuple[FieldResult, ...]

    @property
    def case_pass_rate(self) -> float:
        """Return case pass rate as a float from 0.0 to 1.0."""
        if self.case_count == 0:
            return 0.0
        return self.passed_cases / self.case_count


def load_cases(path: str | Path | None = None) -> list[EvalCase]:
    """Load extraction eval cases from bundled JSONL or a custom path.

    Args:
        path (str | Path | None): Optional JSONL file. When omitted, loads the
            packaged ``extraction_cases.jsonl`` resource.

    Returns:
        list[EvalCase]: Validated eval cases.

    Raises:
        ValueError: If a case is malformed or case IDs are duplicated.
    """
    if path is None:
        resource = files("screening.evals").joinpath("extraction_cases.jsonl")
        text = resource.read_text(encoding="utf-8")
        source = "screening.evals/extraction_cases.jsonl"
    else:
        case_path = Path(path)
        text = case_path.read_text(encoding="utf-8")
        source = str(case_path)

    cases = _load_cases_from_text(text, source=source)
    seen: set[str] = set()
    duplicates: set[str] = set()
    for case in cases:
        if case.id in seen:
            duplicates.add(case.id)
        seen.add(case.id)
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(f"{source}: duplicate case id(s): {duplicate_list}")
    return cases


def score_case(
    case_id: str,
    bucket: str,
    expected: Mapping[str, Any],
    extracted: CandidateProfile,
) -> CaseResult:
    """Score one extracted profile against asserted expected fields.

    Args:
        case_id (str): Eval case identifier.
        bucket (str): Eval bucket name.
        expected (Mapping[str, Any]): Expected normalized fields.
        extracted (CandidateProfile): Profile returned by the extractor.

    Returns:
        CaseResult: Per-field scoring details for the case.
    """
    flattened = flatten_expected(expected, source=f"case {case_id}")
    extracted_data = extracted.model_dump(mode="json")
    fields = []
    for field, expected_value in flattened.items():
        got = _field_value(extracted_data, field)
        correct = _values_match(field, expected_value, got)
        fields.append(
            FieldResult(
                case_id=case_id,
                bucket=bucket,
                field=field,
                expected=expected_value,
                got=got,
                correct=correct,
                soft=field in SOFT_FIELDS,
            )
        )
    return CaseResult(case_id=case_id, bucket=bucket, fields=tuple(fields))


def score_error(
    case_id: str,
    bucket: str,
    expected: Mapping[str, Any],
    error: Exception,
) -> CaseResult:
    """Convert an extraction exception into failed field assertions.

    Args:
        case_id (str): Eval case identifier.
        bucket (str): Eval bucket name.
        expected (Mapping[str, Any]): Expected normalized fields.
        error (Exception): Extraction/API/validation exception.

    Returns:
        CaseResult: Failed case result carrying the exception text.
    """
    error_text = f"{type(error).__name__}: {error}"
    fields = [
        FieldResult(
            case_id=case_id,
            bucket=bucket,
            field=field,
            expected=expected_value,
            got=None,
            correct=False,
            soft=field in SOFT_FIELDS,
            error=error_text,
        )
        for field, expected_value in flatten_expected(
            expected,
            source=f"case {case_id}",
        ).items()
    ]
    return CaseResult(
        case_id=case_id,
        bucket=bucket,
        fields=tuple(fields),
        error=error_text,
    )


def aggregate(results: Iterable[CaseResult]) -> EvalSummary:
    """Aggregate case results into field, bucket, pass-rate, and confusion metrics.

    Args:
        results (Iterable[CaseResult]): Scored case results.

    Returns:
        EvalSummary: Aggregated metrics and failure details.
    """
    case_results = tuple(results)
    field_totals: dict[str, int] = {}
    field_correct: dict[str, int] = {}
    bucket_totals: dict[str, int] = {}
    bucket_correct: dict[str, int] = {}
    confusion: dict[str, dict[str, dict[str, int]]] = {}
    failures: list[FieldResult] = []

    for case in case_results:
        for field in case.fields:
            field_totals[field.field] = field_totals.get(field.field, 0) + 1
            bucket_totals[field.bucket] = bucket_totals.get(field.bucket, 0) + 1

            if field.correct:
                field_correct[field.field] = field_correct.get(field.field, 0) + 1
                bucket_correct[field.bucket] = bucket_correct.get(field.bucket, 0) + 1
            else:
                failures.append(field)

            if field.field in CATEGORICAL_FIELDS:
                expected_label = _label(field.expected)
                got_label = _label(field.got)
                field_matrix = confusion.setdefault(field.field, {})
                expected_row = field_matrix.setdefault(expected_label, {})
                expected_row[got_label] = expected_row.get(got_label, 0) + 1

    field_accuracy = {
        field: Accuracy(correct=field_correct.get(field, 0), total=total)
        for field, total in sorted(field_totals.items())
    }
    bucket_accuracy = {
        bucket: Accuracy(correct=bucket_correct.get(bucket, 0), total=total)
        for bucket, total in sorted(bucket_totals.items())
    }
    overall_total = sum(field_totals.values())
    overall_correct = sum(field_correct.values())

    return EvalSummary(
        case_count=len(case_results),
        passed_cases=sum(1 for case in case_results if case.passed),
        field_accuracy=field_accuracy,
        bucket_accuracy=bucket_accuracy,
        overall_accuracy=Accuracy(correct=overall_correct, total=overall_total),
        confusion_matrices=confusion,
        failures=tuple(failures),
    )


def report_renderable(summary: EvalSummary) -> RenderableType:
    """Build a rich renderable for the eval summary.

    Args:
        summary (EvalSummary): Aggregated eval metrics.

    Returns:
        RenderableType: A grouped set of tables and headings, printable to a
            rich ``Console`` (with color on a terminal).
    """
    sections: list[RenderableType] = [
        _header(summary),
        _accuracy_table("Per-field accuracy", summary.field_accuracy),
        _accuracy_table("Per-bucket accuracy", summary.bucket_accuracy),
        *_confusion_tables(summary.confusion_matrices),
        _failure_table(summary.failures),
    ]
    spaced: list[RenderableType] = []
    for index, section in enumerate(sections):
        if index:
            spaced.append(Text(""))
        spaced.append(section)
    return Group(*spaced)


def format_report(summary: EvalSummary) -> str:
    """Render the eval summary as plain text (no color).

    Used for the ``--out`` workflow and tests. The live runner prints
    :func:`report_renderable` to a real terminal console for color.

    Args:
        summary (EvalSummary): Aggregated eval metrics.

    Returns:
        str: Human-readable report text.
    """
    console = Console(
        file=io.StringIO(),
        width=100,
        force_terminal=False,
        no_color=True,
    )
    console.print(report_renderable(summary))
    return cast(io.StringIO, console.file).getvalue()


def _rate_style(rate: float) -> str:
    """Return a rich style for an accuracy rate: green perfect, yellow high, red."""
    if rate >= 1.0:
        return "green"
    if rate >= 0.9:
        return "yellow"
    return "red"


def _header(summary: EvalSummary) -> RenderableType:
    cases = summary.case_pass_rate
    fields = summary.overall_accuracy.rate
    stats = Text.assemble(
        ("Cases: ", "dim"),
        (
            f"{summary.passed_cases}/{summary.case_count} ({_percent(cases)})",
            _rate_style(cases),
        ),
        ("    Field assertions: ", "dim"),
        (
            f"{summary.overall_accuracy.correct}/{summary.overall_accuracy.total} "
            f"({_percent(fields)})",
            _rate_style(fields),
        ),
    )
    return Group(Text("Extraction Eval Report", style="bold"), stats)


def flatten_expected(
    expected: Mapping[str, Any],
    *,
    source: str = "expected",
) -> dict[str, Any]:
    """Validate and flatten expected fields from a case definition.

    Args:
        expected (Mapping[str, Any]): Expected fields from JSON or tests.
        source (str): Human-readable source label for error messages.

    Returns:
        dict[str, Any]: Flattened field mapping.

    Raises:
        ValueError: If a field is unsupported or unsafe to score.
    """
    flattened: dict[str, Any] = {}
    for field, value in expected.items():
        _validate_expected_field(field, source=source)
        if field == "prior_delivery_experience":
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"{source}: prior_delivery_experience must be an object"
                )
            for nested_field, nested_value in value.items():
                if nested_field not in NESTED_EXPERIENCE_FIELDS:
                    raise ValueError(
                        f"{source}: unsupported experience field {nested_field!r}"
                    )
                flattened[f"prior_delivery_experience.{nested_field}"] = nested_value
        else:
            flattened[field] = value

    if not flattened:
        raise ValueError(f"{source}: expected must assert at least one field")
    return flattened


def _load_cases_from_text(text: str, *, source: str) -> list[EvalCase]:
    cases = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        location = f"{source}:{line_number}"
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{location}: invalid JSON: {error}") from error
        cases.append(_parse_case(payload, location=location))
    if not cases:
        raise ValueError(f"{source}: no cases found")
    return cases


def _parse_case(payload: Any, *, location: str) -> EvalCase:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{location}: case must be a JSON object")

    required = {"id", "bucket", "transcript", "expected"}
    missing = required - set(payload)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{location}: missing required field(s): {missing_list}")

    case_id = payload["id"]
    bucket = payload["bucket"]
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"{location}: id must be a non-empty string")
    if not isinstance(bucket, str) or not bucket.strip():
        raise ValueError(f"{location}: bucket must be a non-empty string")

    expected = payload["expected"]
    if not isinstance(expected, Mapping):
        raise ValueError(f"{location}: expected must be an object")
    expected_dict = dict(expected)
    flatten_expected(expected_dict, source=location)

    current_profile_payload = payload.get("current_profile")
    if current_profile_payload is None:
        current_profile = CandidateProfile()
    elif isinstance(current_profile_payload, Mapping):
        current_profile = CandidateProfile.model_validate(current_profile_payload)
    else:
        raise ValueError(f"{location}: current_profile must be an object or null")

    return EvalCase(
        id=case_id,
        bucket=bucket,
        transcript=_parse_transcript(payload["transcript"], location=location),
        expected=expected_dict,
        current_profile=current_profile,
    )


def _parse_transcript(value: Any, *, location: str) -> list[MessageParam]:
    if not isinstance(value, list):
        raise ValueError(f"{location}: transcript must be a list")

    messages: list[MessageParam] = []
    for index, message in enumerate(value):
        message_location = f"{location}:transcript[{index}]"
        if not isinstance(message, Mapping):
            raise ValueError(f"{message_location}: message must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"assistant", "user"}:
            raise ValueError(f"{message_location}: role must be assistant or user")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{message_location}: content must be a non-empty string")
        messages.append(cast(MessageParam, {"role": role, "content": content}))
    if not messages:
        raise ValueError(f"{location}: transcript must contain at least one message")
    return messages


def _validate_expected_field(field: str, *, source: str) -> None:
    if field.startswith(RAW_FIELD_PREFIX):
        raise ValueError(f"{source}: raw fields are not scored: {field!r}")
    if field in DERIVED_FIELDS:
        raise ValueError(f"{source}: derived fields are not scored: {field!r}")
    if field not in ALLOWED_EXPECTED_FIELDS:
        raise ValueError(f"{source}: unsupported expected field: {field!r}")


def _field_value(data: Mapping[str, Any], field: str) -> Any:
    if field.startswith("prior_delivery_experience."):
        experience = data.get("prior_delivery_experience")
        if not isinstance(experience, Mapping):
            return None
        nested_field = field.rsplit(".", maxsplit=1)[1]
        return experience.get(nested_field)
    return data.get(field)


def _values_match(field: str, expected: Any, got: Any) -> bool:
    if field in SOFT_FIELDS:
        return _normalize_text(expected) == _normalize_text(got)
    if field in NUMERIC_FIELDS:
        return _normalize_number(expected) == _normalize_number(got)
    return expected == got


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip().casefold()


def _normalize_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _label(value: Any) -> str:
    if value is None:
        return "<null>"
    return str(value)


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _accuracy_table(title: str, values: Mapping[str, Accuracy]) -> Table:
    table = Table(title=title, title_justify="left", title_style="bold")
    table.add_column("Name")
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Accuracy", justify="right")
    if not values:
        table.add_row("(none)", "", "", "")
        return table
    for name, accuracy in values.items():
        table.add_row(
            name,
            str(accuracy.correct),
            str(accuracy.total),
            Text(_percent(accuracy.rate), style=_rate_style(accuracy.rate)),
        )
    return table


def _confusion_tables(
    matrices: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> list[RenderableType]:
    heading: RenderableType = Text("Confusion matrices", style="bold")
    if not matrices:
        return [heading, Text("  (none)", style="dim")]
    renderables: list[RenderableType] = [heading]
    renderables.extend(
        _confusion_grid(field, matrices[field]) for field in sorted(matrices)
    )
    return renderables


def _confusion_grid(field: str, matrix: Mapping[str, Mapping[str, int]]) -> Table:
    expected_labels = sorted(matrix)
    predicted_labels = sorted(
        {got for row in matrix.values() for got in row}.union(expected_labels)
    )
    table = Table(title=field, title_justify="left", title_style="bold cyan")
    table.add_column("expected ↓ / predicted →", style="dim")
    for label in predicted_labels:
        table.add_column(label, justify="right")
    for expected in expected_labels:
        cells = [Text(expected)]
        for predicted in predicted_labels:
            count = matrix[expected].get(predicted, 0)
            cells.append(_confusion_cell(count, correct=predicted == expected))
        table.add_row(*cells)
    return table


def _confusion_cell(count: int, *, correct: bool) -> Text:
    if count == 0:
        return Text("·", style="dim")
    if correct:
        return Text(str(count), style="green")
    return Text(str(count), style="bold red")


def _failure_table(failures: Iterable[FieldResult]) -> Table:
    failure_list = tuple(failures)
    table = Table(title="Failures", title_justify="left", title_style="bold red")
    for column in ("Case", "Bucket", "Field", "Expected", "Got", "Error"):
        table.add_column(column)
    if not failure_list:
        table.add_row("(none)", "", "", "", "", "")
        return table
    for failure in failure_list:
        table.add_row(
            failure.case_id,
            failure.bucket,
            failure.field,
            _json_value(failure.expected),
            _json_value(failure.got),
            failure.error or "",
        )
    return table


def _json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
