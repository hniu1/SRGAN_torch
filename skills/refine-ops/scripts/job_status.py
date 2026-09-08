#!/usr/bin/env python3
"""Query live and historical Slurm state for exact job IDs and emit JSON."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from typing import Any


def run(command: list[str]) -> tuple[str, str | None]:
    executable = command[0]
    if shutil.which(executable) is None:
        return "", f"{executable} is not available"
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "", f"{executable} failed: {exc}"
    if process.returncode:
        message = process.stderr.strip() or f"exit code {process.returncode}"
        return process.stdout, f"{executable}: {message}"
    return process.stdout, None


def rows(text: str, fields: list[str]) -> list[dict[str, str]]:
    parsed = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = line.rstrip().split("|")
        if len(values) < len(fields):
            values.extend([""] * (len(fields) - len(values)))
        parsed.append(dict(zip(fields, values[: len(fields)])))
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_ids", nargs="+", help="exact numeric Slurm job IDs")
    args = parser.parse_args()
    invalid = [job_id for job_id in args.job_ids if not job_id.isdigit()]
    if invalid:
        print(json.dumps({"status": "invalid", "errors": [f"invalid job ID: {item}" for item in invalid]}, indent=2))
        return 2

    job_list = ",".join(args.job_ids)
    live_text, live_error = run(
        ["squeue", "-h", "-j", job_list, "-o", "%i|%j|%T|%M|%R"]
    )
    history_text, history_error = run(
        [
            "sacct",
            "-n",
            "-X",
            "-j",
            job_list,
            "--parsable2",
            "--format=JobIDRaw,JobName,State,ExitCode,Elapsed,Start,End",
        ]
    )
    errors = [error for error in (live_error, history_error) if error]
    report: dict[str, Any] = {
        "status": "ok" if not errors else "partial" if live_text or history_text else "unavailable",
        "requested_job_ids": args.job_ids,
        "live": rows(live_text, ["job_id", "job_name", "state", "elapsed", "reason_or_node"]),
        "accounting": rows(
            history_text,
            ["job_id", "job_name", "state", "exit_code", "elapsed", "start", "end"],
        ),
        "errors": errors,
    }
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if report["status"] in ("ok", "partial") else 3


if __name__ == "__main__":
    raise SystemExit(main())
