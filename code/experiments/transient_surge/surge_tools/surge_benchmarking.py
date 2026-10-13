"""Run [a, b) cases from JSONL with PSO assignments selected by matching index.

The simulation engine is unchanged. Cases run sequentially; replications
within a case can run in parallel. Output is one COMPLETE strict-JSON case
record per line. Call under an if __name__ == '__main__' guard for workers.
"""
from __future__ import annotations

from collections.abc import Iterator
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .surge_case_analysis import (
    SCHEMA_VERSION,
    Surge_phase_costs,
    Surge_phase_costs_from_case,
    _case_row_arguments,
    _integer,
)


def iter_benchmark_results(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read complete records lazily from the runner's JSONL output.

    Raises on malformed/incomplete lines instead of ignoring interrupted data.
    Compatible with aggregate_case_records(list(iter_benchmark_results(path))).
    """
    def reject_constant(token):
        raise ValueError(f"Non-standard JSON number: {token}")

    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, parse_constant=reject_constant)
            except ValueError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"Unsupported case record at {path}:{line_number}.")
            try:
                _integer(record["inputs"]["case_id"], "case_id")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Missing/invalid case ID at {path}:{line_number}.") from exc
            yield record


def _load_case_rows(path: Path):
    rows, physical_lines = [], []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid input JSON at {path}:{line_number}.") from exc
            if not isinstance(case, dict):
                raise ValueError(f"Expected a case object at {path}:{line_number}.")
            rows.append(case)
            physical_lines.append(line_number)
    if not rows:
        raise ValueError(f"No case records found in {path}.")
    # Match the input normalization in the user's original benchmark function.
    return pd.json_normalize(rows), rows, physical_lines


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def Surge_benchmarking(
    a, b, logFile="fullRangeResultsFullFlexNew.json", *,
    assignmentFile="PSO service priority benchmarking large cases.json",
    outputFile="surge_benchmark_results.jsonl",
    assignmentColumn="assignment_matrix",
    assignmentOrient=None,
    assignmentLines=False,
    maxQueue=500,
    replications=35,
    nCores=None,
    startSeed=12345,
    print_report=True,
    append=False,
    overwrite=False,
    return_records=False,
    **simulation_options,
) -> dict[str, Any]:
    """Simulate original case indices a <= i < b using PSO_OPT.loc[i].

    The case file is JSON Lines (one object per line), even with a .json suffix.
    assignmentFile defaults to the pandas JSON format read by pd.read_json in
    the original code. Set assignmentOrient / assignmentLines explicitly for
    other pandas JSON layouts. There is NO positional fallback to iloc[i].

    Each complete simulation record is written to outputFile and flushed
    before starting the next case. Do not write concurrently to this file.
    Existing output is protected unless append=True or overwrite=True.
    Append rejects duplicate case IDs, input-file changes, and invalid JSONL.

    Returns a small manifest containing case summaries and output location.
    Set return_records=True to ALSO retain all full records in memory. The
    full replication arrays are always saved, regardless of this flag.

    Defaults inherited from Surge_phase_costs:
      100k expected arrivals: 10k warmup, 20k baseline, 10k surge, 60k recovery;
      surge multiplier 3, report blocks 10k, fine bins <=100 baseline units;
      fastest-service-first priorities and preemptive service.
    Pass simulation_options to override these or recovery/inventory settings.
    No database is required, and no GA/PSO optimization is rerun.
    """
    a = _integer(a, "a")
    b = _integer(b, "b")
    if b <= a:
        raise ValueError("Require 0 <= a < b; b is exclusive.")
    if append and overwrite:
        raise ValueError("append and overwrite cannot both be True.")

    case_path = Path(logFile).expanduser().resolve()
    assignment_path = Path(assignmentFile).expanduser().resolve()
    output_path = Path(outputFile).expanduser().resolve()
    if output_path in (case_path, assignment_path):
        raise ValueError("Output must not replace either input file.")
    for path in (case_path, assignment_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists() and not (append or overwrite):
        raise FileExistsError(
            f"{output_path} already exists; choose a new file, append=True, "
            "or explicitly overwrite=True."
        )

    df, original_rows, line_numbers = _load_case_rows(case_path)
    if b > len(df):
        raise ValueError(f"Requested range [{a}, {b}) exceeds {len(df)} input cases.")
    pso = pd.read_json(assignment_path, orient=assignmentOrient, lines=assignmentLines)
    if not isinstance(pso, pd.DataFrame) or assignmentColumn not in pso.columns:
        raise ValueError(f"Assignment file needs a '{assignmentColumn}' column.")
    if not pso.index.is_unique:
        raise ValueError("Assignment row labels must be unique.")

    # a/b refer to ORIGINAL zero-based case-row indices, not a filtered range.
    # caseId=i therefore keeps seeds stable when work is split into ranges.
    run_options = dict(simulation_options)
    forbidden = set(inspect.signature(Surge_phase_costs).parameters)
    forbidden = {key for key in forbidden if key in (
        "optimized_assignment", "m_skills", "n_servers", "failure_rates",
        "service_rates", "holding_costs", "penalty_cost", "maxQueue",
        "skill_costs", "machine_costs", "caseId", "case_name", "output_path",
    )}
    conflicting = forbidden.intersection(run_options)
    if conflicting:
        raise TypeError(f"These options are supplied per input case: {sorted(conflicting)}")
    run_options.update(replications=replications, nCores=nCores,
                       startSeed=startSeed, print_report=print_report)

    # Validate every selected row/assignment before spending time on a batch.
    selected = []
    for i in range(a, b):
        if i not in pso.index:
            raise ValueError(
                f"PSO assignment index {i} is missing. Match the original case "
                "row labels; do not rely on a differently ordered file."
            )
        row = df.loc[i].to_dict()
        matrix = pso.loc[i, assignmentColumn]
        try:
            arguments = _case_row_arguments(
                row, matrix, maxQueue=maxQueue,
                priorities=run_options.get("priorities", "fastest_service_first"),
            )
            inspect.signature(Surge_phase_costs).bind(
                **arguments, caseId=i, case_name=f"case_{i}", **run_options,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid case/assignment at original index {i}: {exc}") from exc
        selected.append((i, row, arguments["optimized_assignment"]))

    case_hash = _sha256(case_path)
    assignment_hash = _sha256(assignment_path)
    existing_ids = set()
    if append and output_path.exists():
        for record in iter_benchmark_results(output_path):
            previous_id = record["inputs"]["case_id"]
            if previous_id in existing_ids:
                raise ValueError(f"Output already contains duplicate case ID {previous_id}.")
            existing_ids.add(previous_id)
            source = record.get("source", {})
            if (source.get("case_file_sha256") != case_hash
                    or source.get("assignment_file_sha256") != assignment_hash):
                raise ValueError("Append requires the same case and assignment file contents.")
        duplicates = existing_ids.intersection(range(a, b))
        if duplicates:
            raise ValueError(f"Output already contains requested case IDs: {sorted(duplicates)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "output_file": str(output_path),
        "requested_range": {"start_inclusive": a, "stop_exclusive": b},
        "completed_case_ids": [],
        "summaries": [],
        "records": [] if return_records else None,
    }
    mode = "a" if append else ("w" if overwrite else "x")
    with output_path.open(mode, encoding="utf-8") as stream:
        # A valid last record need not have a trailing newline.
        if append and output_path.stat().st_size:
            with output_path.open("rb") as reader:
                reader.seek(-1, os.SEEK_END)
                if reader.read(1) not in (b"\n", b"\r"):
                    stream.write("\n")
        for i, row, assignment in selected:
            print(f"Running surge case {i} ({i-a+1}/{b-a}); "
                  f"{replications} replications", flush=True)
            source = {
                "case_file": str(case_path),
                "case_file_sha256": case_hash,
                "case_row_index": i,
                "case_file_line": line_numbers[i],
                "assignment_file": str(assignment_path),
                "assignment_file_sha256": assignment_hash,
                "assignment_column": assignmentColumn,
                "assignment_row_index": i,
                "index_matching": "df.loc[i] and PSO_OPT.loc[i]; never positional fallback",
                "assignment_origin": "precomputed PSO assignment; not reoptimized",
                "original_input_case": original_rows[i],
            }
            try:
                record = Surge_phase_costs_from_case(
                    row, assignment, caseId=i, case_name=f"case_{i}",
                    maxQueue=maxQueue, source=source, **run_options,
                )
                # Serialize fully BEFORE writing so encoding errors cannot
                # leave half a JSON object. Power/process failure still can;
                # the reader detects malformed lines instead of skipping them.
                encoded = json.dumps(record, allow_nan=False, ensure_ascii=False,
                                     separators=(",", ":"))
                stream.write(encoded + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            except Exception as exc:
                raise RuntimeError(
                    f"Case {i} failed. Earlier completed records remain in {output_path}. "
                    "Inspect the cause; do not interpret this as a completed batch."
                ) from exc
            manifest["completed_case_ids"].append(i)
            manifest["summaries"].append({
                "case_id": i,
                "baseline_queue_mean": record["summary"]["baseline_queue"]["mean"],
                "surge_queue_mean": record["summary"]["surge_queue"]["mean"],
                "recovered_fraction": record["summary"]["recovered_fraction"],
                "mean_recovery_arrival_units_recovered_only": record["summary"]
                    ["recovery_arrival_units_recovered_only"]["mean"],
            })
            if return_records:
                manifest["records"].append(record)
            print(f"Saved complete case {i} to {output_path}", flush=True)
    return manifest
