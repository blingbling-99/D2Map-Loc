"""Core D2Map-Loc selector.

The selector only consumes completed map descriptors. Query images, poses,
correctness labels, and localization errors are deliberately absent from the
interface.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F


def normalize_descriptors(descriptors: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalized float32 descriptors.

    Zero rows remain zero. Non-finite or non-matrix inputs are rejected.
    """

    values = np.asarray(descriptors, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("descriptors must be a finite [N, D] array")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("descriptors must have non-zero N and D")
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return np.ascontiguousarray(
        values / np.maximum(norm, np.finfo(np.float32).tiny)
    )


@torch.inference_mode()
def select_indices(
    descriptors: np.ndarray,
    budget: int,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Select exactly ``budget`` rows using normalized-cosine k-center greedy.

    Initialization chooses the descriptor least aligned with the normalized
    global descriptor mean. Each later step chooses the row whose maximum
    cosine similarity to the selected set is smallest. ``torch.argmin`` makes
    exact ties resolve to the smallest original row index.

    Args:
        descriptors: Completed map descriptors with shape ``[N, D]``.
        budget: Number of original rows to retain, satisfying ``1 <= K <= N``.
        device: PyTorch device used for selection, e.g. ``"cpu"`` or ``"cuda"``.

    Returns:
        A contiguous int64 array of selected original row indices in selection
        order. Descriptor values are used only to choose membership and are not
        modified in the caller's map.
    """

    normalized = normalize_descriptors(descriptors)
    count = len(normalized)
    target = int(budget)
    if target <= 0 or target > count:
        raise ValueError("budget must satisfy 1 <= budget <= N")

    vectors = torch.as_tensor(normalized, dtype=torch.float32, device=device)
    center = F.normalize(vectors.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
    center_similarity = vectors @ center
    first = int(torch.argmin(center_similarity).item())

    order = torch.empty(target, dtype=torch.int64, device=device)
    order[0] = first
    maximum_similarity = vectors @ vectors[first]
    maximum_similarity[first] = torch.inf

    for position in range(1, target):
        selected = torch.argmin(maximum_similarity)
        order[position] = selected
        similarity = vectors @ vectors[selected]
        maximum_similarity = torch.maximum(maximum_similarity, similarity)
        maximum_similarity[order[: position + 1]] = torch.inf

    result = np.ascontiguousarray(order.cpu().numpy(), dtype=np.int64)
    if len(np.unique(result)) != target:
        raise AssertionError("selection contains duplicate rows")
    return result


def select_map(
    descriptors: np.ndarray,
    budget: int,
    *,
    device: str | torch.device = "cpu",
    aligned_arrays: Mapping[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Select indices and apply them jointly to row-aligned map arrays.

    Pass XYZ coordinates, Gaussian IDs, or other per-landmark arrays through
    ``aligned_arrays``. Every array must have the same first dimension as the
    descriptor matrix. Original values are copied unchanged.
    """

    values = np.asarray(descriptors)
    indices = select_indices(values, budget, device=device)
    arrays: dict[str, np.ndarray] = {"descriptors": np.ascontiguousarray(values[indices])}

    for name, array in (aligned_arrays or {}).items():
        if name == "descriptors":
            raise ValueError("'descriptors' is a reserved aligned-array name")
        aligned = np.asarray(array)
        if aligned.ndim == 0 or aligned.shape[0] != values.shape[0]:
            raise ValueError(f"aligned array {name!r} must have first dimension N")
        arrays[name] = np.ascontiguousarray(aligned[indices])
    return indices, arrays
