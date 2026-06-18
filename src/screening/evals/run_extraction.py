"""Manual runner for live extraction-quality evals."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from screening.config import (
    ANTHROPIC_API_KEY_ENV,
    MAX_RETRIES,
    REQUEST_TIMEOUT_SECONDS,
)
from screening.evals.scoring import (
    CaseResult,
    EvalCase,
    EvalSummary,
    aggregate,
    load_cases,
    report_renderable,
    score_case,
    score_error,
)
from screening.llm.extraction import CandidateExtractor


def main(argv: list[str] | None = None) -> int:
    """Run the live extraction eval and print a report.

    Args:
        argv (list[str] | None): Optional CLI arguments for tests or embedding.

    Returns:
        int: Process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    load_dotenv()
    api_key = os.getenv(ANTHROPIC_API_KEY_ENV)
    if not api_key:
        parser.error(
            f"{ANTHROPIC_API_KEY_ENV} is required. Set it in .env or your shell."
        )

    cases = _filter_cases(
        load_cases(args.cases),
        buckets=args.bucket,
        case_ids=args.case_id,
    )
    if not cases:
        parser.error("no eval cases matched the provided filters")

    client = Anthropic(
        api_key=api_key,
        max_retries=MAX_RETRIES,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    extractor = CandidateExtractor(client)
    results, raw_runs = _run_cases(
        extractor,
        cases=cases,
        samples=args.samples,
        workers=args.workers,
    )
    summary = aggregate(results)

    Console().print(report_renderable(summary))
    if args.out is not None:
        _write_results(args.out, summary=summary, runs=raw_runs)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run live CandidateExtractor quality evals.",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        help="Path to a JSONL case file. Defaults to the bundled seed cases.",
    )
    parser.add_argument(
        "--bucket",
        action="append",
        help="Bucket to run. May be provided more than once.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        help="Specific case ID to run. May be provided more than once.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Repeat each case N times and aggregate all runs. Defaults to 1.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent extraction calls. Defaults to 8.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional path for raw JSON results.",
    )
    return parser


def _filter_cases(
    cases: list[EvalCase],
    *,
    buckets: list[str] | None,
    case_ids: list[str] | None,
) -> list[EvalCase]:
    bucket_filter = set(buckets or [])
    case_id_filter = set(case_ids or [])
    return [
        case
        for case in cases
        if (not bucket_filter or case.bucket in bucket_filter)
        and (not case_id_filter or case.id in case_id_filter)
    ]


def _run_cases(
    extractor: CandidateExtractor,
    *,
    cases: list[EvalCase],
    samples: int,
    workers: int,
) -> tuple[list[CaseResult], list[dict[str, Any]]]:
    work = [
        (sample_index, case) for sample_index in range(1, samples + 1) for case in cases
    ]
    completed: list[tuple[int, CaseResult, dict[str, Any]]] = []

    with (
        ThreadPoolExecutor(max_workers=workers) as executor,
        _progress_bar() as progress,
    ):
        task = progress.add_task("Extracting", total=len(work))
        futures = {
            executor.submit(_run_one, extractor, sample_index, case): order_index
            for order_index, (sample_index, case) in enumerate(work)
        }
        for future in as_completed(futures):
            order_index = futures[future]
            result, raw_run = future.result()
            completed.append((order_index, result, raw_run))
            progress.advance(task)

    completed.sort(key=lambda item: item[0])
    results = [result for _, result, _ in completed]
    raw_runs = [raw_run for _, _, raw_run in completed]
    return results, raw_runs


def _progress_bar() -> Progress:
    """Build a live progress bar that renders to stderr.

    Rendering to stderr keeps stdout reserved for the final report, so a
    redirected or piped run captures only the report.

    Returns:
        Progress: A configured rich progress bar.
    """
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=Console(stderr=True),
    )


def _run_one(
    extractor: CandidateExtractor,
    sample_index: int,
    case: EvalCase,
) -> tuple[CaseResult, dict[str, Any]]:
    profile_payload: dict[str, Any] | None = None
    try:
        profile = extractor.extract(case.transcript, case.current_profile)
    except Exception as error:
        result = score_error(case.id, case.bucket, case.expected, error)
        error_text = result.error
    else:
        result = score_case(case.id, case.bucket, case.expected, profile)
        error_text = None
        profile_payload = profile.model_dump(mode="json")

    raw_run = {
        "sample": sample_index,
        "case_id": case.id,
        "bucket": case.bucket,
        "passed": result.passed,
        "error": error_text,
        "extracted": profile_payload,
        "fields": [_field_result_to_dict(field) for field in result.fields],
    }
    return result, raw_run


def _write_results(
    path: Path,
    *,
    summary: EvalSummary,
    runs: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": _summary_to_dict(summary),
        "runs": runs,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _summary_to_dict(summary: EvalSummary) -> dict[str, Any]:
    return {
        "case_count": summary.case_count,
        "passed_cases": summary.passed_cases,
        "case_pass_rate": summary.case_pass_rate,
        "overall_accuracy": _accuracy_to_dict(summary.overall_accuracy),
        "field_accuracy": {
            field: _accuracy_to_dict(accuracy)
            for field, accuracy in summary.field_accuracy.items()
        },
        "bucket_accuracy": {
            bucket: _accuracy_to_dict(accuracy)
            for bucket, accuracy in summary.bucket_accuracy.items()
        },
        "confusion_matrices": summary.confusion_matrices,
        "failures": [_field_result_to_dict(field) for field in summary.failures],
    }


def _accuracy_to_dict(accuracy) -> dict[str, float | int]:
    return {
        "correct": accuracy.correct,
        "total": accuracy.total,
        "rate": accuracy.rate,
    }


def _field_result_to_dict(field) -> dict[str, Any]:
    return {
        "case_id": field.case_id,
        "bucket": field.bucket,
        "field": field.field,
        "expected": field.expected,
        "got": field.got,
        "correct": field.correct,
        "soft": field.soft,
        "error": field.error,
    }


if __name__ == "__main__":
    raise SystemExit(main())
