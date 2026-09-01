#!/usr/bin/env python3
"""Plot ClimateSwin training and validation losses from history.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("artifacts/runs/climateswin_v1/history.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/runs/climateswin_v1/training_loss_curves.png"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = [
        json.loads(line)
        for line in args.history.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"No training records found in {args.history}")

    epochs = [int(record["epoch"]) for record in records]
    train_total = [record["train"]["total"] for record in records]
    validation_total = [record["validation"]["total"] for record in records]
    best_index = min(range(len(records)), key=validation_total.__getitem__)
    best_epoch = epochs[best_index]
    best_loss = validation_total[best_index]

    colors = {"tmin_data": "#2878b5", "tmax_data": "#d95319", "prcp_data": "#3c9d4e"}
    labels = {"tmin_data": "tmin", "tmax_data": "tmax", "prcp_data": "precipitation"}
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    axis = axes[0, 0]
    axis.plot(epochs, train_total, label="training", linewidth=2)
    axis.plot(epochs, validation_total, label="validation", linewidth=2)
    axis.scatter([best_epoch], [best_loss], color="black", zorder=4)
    axis.annotate(
        f"best: epoch {best_epoch}\n{best_loss:.5f}",
        (best_epoch, best_loss),
        xytext=(-70, 30),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "black"},
    )
    axis.set_title("Total objective")
    axis.set_ylabel("Normalized loss")
    axis.legend()

    for axis, split, title in (
        (axes[0, 1], "validation", "Validation data terms"),
        (axes[1, 0], "train", "Training data terms"),
    ):
        for key in ("tmin_data", "tmax_data", "prcp_data"):
            axis.plot(
                epochs,
                [record[split][key] for record in records],
                label=labels[key],
                color=colors[key],
                linewidth=2,
            )
        axis.set_title(title)
        axis.set_ylabel("Normalized loss")
        axis.legend()

    axis = axes[1, 1]
    axis.plot(
        epochs,
        [record["validation"]["precipitation_conservation"] for record in records],
        label="precipitation conservation",
        linewidth=2,
        color="#7b4ab5",
    )
    axis.plot(
        epochs,
        [record["validation"]["temperature_order"] for record in records],
        label="temperature order",
        linewidth=1.5,
        color="#8c6d31",
    )
    axis.set_yscale("log")
    axis.set_title("Validation physical constraints")
    axis.set_ylabel("Loss (log scale)")
    axis.legend()

    for axis in axes.flat:
        axis.axvline(63.5, color="0.65", linestyle="--", linewidth=1)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    figure.suptitle(
        "ClimateSwin multivariable training\nDashed line: walltime interruption and resume",
        fontsize=15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
