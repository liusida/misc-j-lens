# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Plot the distribution of all centuries-logit contributions."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = (
    ROOT
    / "j_lens_output"
    / "five_prompt_centuries"
    / "centuries_all_pair_contributions.csv"
)
OUTPUT = (
    ROOT
    / "j_lens_output"
    / "five_prompt_centuries"
    / "centuries_contribution_histogram.png"
)


def main() -> None:
    with INPUT.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    values = np.array(
        [float(row["contribution_to_centuries_logit"]) for row in records]
    )

    fig, (ax, ax_zoom) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        height_ratios=(2, 1),
        constrained_layout=True,
    )
    bins = np.linspace(values.min(), values.max(), 121)
    ax.hist(values, bins=bins, color="#2878B5", edgecolor="white", linewidth=0.35)
    ax.axvline(0, color="#222222", linewidth=1, label="zero")
    ax.axvline(values.mean(), color="#E07A1F", linewidth=1.5, label="mean")
    ax.axvline(np.median(values), color="#6A3D9A", linewidth=1.5, label="median")
    ax.set_xlabel("Contribution to the ‘ centuries’ logit")
    ax.set_ylabel("Number of source–target rows")
    ax.set_title(
        f"Distribution of {len(values):,} source–target pair contributions"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    # A near-zero view makes the negative and small-positive mass visible.
    central = values[np.abs(values) <= 0.015]
    zoom_bins = np.linspace(-0.015, 0.015, 101)
    ax_zoom.hist(
        central,
        bins=zoom_bins,
        color="#4C9F70",
        edgecolor="white",
        linewidth=0.3,
    )
    ax_zoom.axvline(0, color="#222222", linewidth=1)
    ax_zoom.set_xlim(-0.015, 0.015)
    ax_zoom.set_xlabel("Contribution (zoom around zero)")
    ax_zoom.set_ylabel("Rows")
    ax_zoom.grid(axis="y", alpha=0.25)

    stats = (
        f"mean = {values.mean():+.6f}\n"
        f"median = {np.median(values):+.6f}\n"
        f"negative = {(values < 0).sum():,}\n"
        f"positive = {(values > 0).sum():,}"
    )
    ax.text(
        0.98,
        0.96,
        stats,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#bbbbbb"},
    )

    fig.savefig(OUTPUT, dpi=220)
    print(f"Saved {OUTPUT}")
    print(f"Rows: {len(values)}")
    print(f"Mean: {values.mean():+.8f}")
    print(f"Median: {np.median(values):+.8f}")


if __name__ == "__main__":
    main()
