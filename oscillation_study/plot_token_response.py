# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Render the saved top-token response with target-token tick labels."""
import csv
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--folder", type=Path, default=Path(__file__).resolve().parent / "output")
args = parser.parse_args()
folder = args.folder
summary = json.loads((folder / "summary.json").read_text())
with (folder / "responses.csv").open(newline="") as handle:
    rows = [r for r in csv.DictReader(handle) if r["method"] == "top_token"]
positions = [int(r["target_position"]) for r in rows]
values = [float(r["projection_on_average_transport"]) for r in rows]
labels = [f"{r['target_token']!r}\n{r['target_position']}" for r in rows]
fig, ax = plt.subplots(figsize=(17, 5.5), layout="constrained")
ax.plot(positions, values, marker="o", markersize=4, linewidth=1.8)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xticks(positions, labels, rotation=65, ha="right", fontsize=9)
ax.set_xlabel("Target token and absolute token position")
ax.set_ylabel("Signed projection onto average transport")
ax.set_title(
    f"{summary['decoded_token']!r} direction: downstream response from "
    f"layer {summary['layer']}, position {summary['position']} ({summary['query_token']!r})"
)
ax.grid(axis="y", alpha=0.2)
for extension in ["png", "pdf"]:
    path = folder / f"top_token_response_token_ticks.{extension}"
    fig.savefig(path, dpi=200)
    print(path)
