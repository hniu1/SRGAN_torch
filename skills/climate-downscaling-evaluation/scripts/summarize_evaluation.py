#!/usr/bin/env python3
"""Render a ClimateSwin evaluation_summary.json as Markdown or normalized JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def normalize(summary: dict[str, Any], source: Path) -> dict[str, Any]:
    rows = []
    metadata = summary.get("variable_metadata", {})
    for variable in summary.get("variables", summary.get("metrics", {}).keys()):
        metrics = summary.get("metrics", {}).get(variable, {})
        model = metrics.get("model", {})
        baseline = metrics.get("bilinear_baseline", {})
        rows.append(
            {
                "variable": variable,
                "units": metadata.get(variable, {}).get("units", "unknown"),
                "bias": model.get("bias"),
                "mae": model.get("mae"),
                "rmse": model.get("rmse"),
                "bilinear_mae": baseline.get("mae"),
                "mae_improvement_percent": metrics.get("mae_improvement_percent"),
                "count": metrics.get("count"),
            }
        )
    return {
        "source": str(source),
        "checkpoint": summary.get("checkpoint"),
        "data_dir": summary.get("data_dir"),
        "split": summary.get("split"),
        "days": summary.get("days"),
        "parameters": summary.get("parameters"),
        "temperature_order_enforced": summary.get("temperature_order_enforced"),
        "temperature_order_violation_fraction": summary.get("temperature_order_violation_fraction"),
        "metrics": rows,
        "spatial_statistics": summary.get("spatial_statistics"),
        "spatial_comparison_plots": summary.get("spatial_comparison_plots", []),
    }


def display(value: Any, digits: int = 5) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ClimateSwin evaluation",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Data: `{report['data_dir']}`",
        f"- Split/days: {report['split']} / {report['days']}",
        f"- Parameters: {report['parameters']}",
        f"- Temperature order enforced: {report['temperature_order_enforced']}",
        f"- Pre-enforcement violation fraction: {display(report['temperature_order_violation_fraction'])}",
        "",
        "| Variable | Units | Bias | MAE | RMSE | Bilinear MAE | MAE improvement | Count |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["metrics"]:
        improvement = row["mae_improvement_percent"]
        lines.append(
            "| {variable} | {units} | {bias} | {mae} | {rmse} | {bilinear} | {improvement} | {count} |".format(
                variable=row["variable"],
                units=row["units"],
                bias=display(row["bias"]),
                mae=display(row["mae"]),
                rmse=display(row["rmse"]),
                bilinear=display(row["bilinear_mae"]),
                improvement="n/a" if improvement is None else f"{improvement:.2f}%",
                count=display(row["count"]),
            )
        )
    if report["spatial_statistics"]:
        lines.extend(["", f"Spatial statistics: `{report['spatial_statistics']}`"])
    if report["spatial_comparison_plots"]:
        lines.extend(["", f"Spatial plots: {len(report['spatial_comparison_plots'])}"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()

    try:
        with args.summary.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read {args.summary}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(raw, dict) or not isinstance(raw.get("metrics"), dict):
        print("error: expected a JSON object containing a metrics object", file=sys.stderr)
        return 2

    report = normalize(raw, args.summary)
    if args.format == "json":
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
