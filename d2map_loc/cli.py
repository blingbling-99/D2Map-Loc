"""Command-line interface for selecting a compact descriptor/XYZ map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .selector import select_map


def _load(path: Path) -> np.ndarray:
    if path.suffix != ".npy":
        raise ValueError(f"only non-pickled .npy input is accepted: {path}")
    return np.load(path, allow_pickle=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select a fixed-budget descriptor-diverse map subset."
    )
    parser.add_argument("--descriptors", type=Path, required=True, help="[N,D] .npy")
    parser.add_argument("--xyz", type=Path, help="optional row-aligned [N,3] .npy")
    parser.add_argument("--ids", type=Path, help="optional row-aligned [N] .npy")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, ...")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    descriptors = _load(args.descriptors)
    aligned = {}
    if args.xyz is not None:
        aligned["xyz"] = _load(args.xyz)
    if args.ids is not None:
        aligned["ids"] = _load(args.ids)

    indices, selected = select_map(
        descriptors,
        args.budget,
        device=args.device,
        aligned_arrays=aligned,
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    paths = {"indices": output / "selected_indices.npy"}
    np.save(paths["indices"], indices, allow_pickle=False)
    for name, array in selected.items():
        paths[name] = output / f"selected_{name}.npy"
        np.save(paths[name], array, allow_pickle=False)

    manifest = {
        "algorithm": "D2Map-Loc normalized-cosine k-center greedy",
        "budget": int(args.budget),
        "device": args.device,
        "input_descriptor_shape": list(descriptors.shape),
        "outputs": {
            name: {
                "file": path.name,
                "sha256": _sha256(path),
                "shape": list(np.load(path, allow_pickle=False).shape),
            }
            for name, path in sorted(paths.items())
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
