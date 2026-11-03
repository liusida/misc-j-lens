# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Plot every centuries-contribution table row as a horizontal bar."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT = (
    ROOT
    / "artifacts/runs/j/legacy"
    / "five_prompt_centuries"
    / "centuries_all_pair_contributions.csv"
)
OUTPUT = (
    ROOT
    / "artifacts/runs/j/legacy"
    / "five_prompt_centuries"
    / "centuries_contribution_by_row.png"
)


def main() -> None:
    with INPUT.open(newline="") as handle:
        records = list(csv.DictReader(handle))

    contributions = np.array(
        [float(row["contribution_to_centuries_logit"]) for row in records]
    )
    contributions.sort()
    row_rank = np.arange(1, len(contributions) + 1)
    colors = np.where(contributions >= 0, "#2878B5", "#D9534F")

    fig, ax = plt.subplots(figsize=(10, 12), constrained_layout=True)
    ax.barh(
        row_rank,
        contributions,
        height=1.0,
        color=colors,
        edgecolor="none",
        rasterized=True,
    )
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_xlabel("Contribution to the ‘ centuries’ logit")
    ax.set_ylabel("Contribution-table row (sorted by contribution)")
    ax.set_title(
        f"All {len(contributions):,} source–target pair contributions\n"
        "Five-prompt J-Lens, query activation $h_{12,17}$"
    )
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.text(
        0.01,
        0.99,
        f"negative rows: {(contributions < 0).sum():,}\n"
        f"positive rows: {(contributions > 0).sum():,}\n"
        f"sum: {contributions.sum():+.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#bbbbbb"},
    )

    fig.savefig(OUTPUT, dpi=220)
    print(f"Saved {OUTPUT}")
    print(f"Rows: {len(contributions)}")
    print(f"Range: {contributions.min():+.8f} to {contributions.max():+.8f}")
    print(f"Sum: {contributions.sum():+.8f}")


if __name__ == "__main__":
    main()
