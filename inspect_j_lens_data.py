# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.7"]
# ///
"""Inspect one scalar path through the saved, unaveraged J-Lens data."""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default="j_lens_output/one_prompt_layer.pt")
    parser.add_argument("--output-dim", type=int, default=100)
    parser.add_argument("--input-dim", type=int, default=200)
    args = parser.parse_args()

    data = torch.load(args.artifact, map_location="cpu", weights_only=False)
    raw = data["raw_grads_before_position_mean"].float()
    averaged = data["j_prompt_after_position_mean"]
    positions = data["valid_positions"].tolist()
    tokens = data["tokens"]
    i, j = args.output_dim, args.input_dim
    values = raw[i, positions, j]

    print(f"G[{i}, t, {j}] before averaging over valid source position t:")
    for position, value in zip(positions, values.tolist(), strict=True):
        print(
            f"  t={position:02d} token={tokens[position]!r:14s} "
            f"value={value:+.8f}"
        )
    manual_mean = values.mean()
    stored_mean = averaged[i, j]
    print(f"\nmanual mean       = {manual_mean.item():+.10f}")
    print(f"stored J[{i},{j}] = {stored_mean.item():+.10f}")
    print(f"absolute error    = {(manual_mean - stored_mean).abs().item():.3e}")


if __name__ == "__main__":
    main()
