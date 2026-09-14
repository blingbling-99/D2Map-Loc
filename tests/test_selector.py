import inspect

import numpy as np
import pytest
import torch

from d2map_loc import normalize_descriptors, select_indices, select_map


def test_normalization_is_float32_finite_and_rowwise():
    source = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float64)
    result = normalize_descriptors(source)
    assert result.dtype == np.float32
    assert np.allclose(result[0], [0.6, 0.8])
    assert np.array_equal(result[1], [0.0, 0.0])


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_selection_is_deterministic_unique_and_exact(device):
    descriptors = np.random.default_rng(12).normal(size=(20, 8)).astype(np.float32)
    first = select_indices(descriptors, 15, device=device)
    second = select_indices(descriptors.copy(), 15, device=device)
    assert np.array_equal(first, second)
    assert first.dtype == np.int64
    assert len(first) == len(np.unique(first)) == 15


def test_ties_resolve_to_smallest_original_row():
    descriptors = np.asarray([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    assert select_indices(descriptors, 4).tolist() == [0, 2, 1, 3]


def test_small_frozen_reference_order_guards_algorithm_definition():
    descriptors = np.random.default_rng(2027).normal(size=(9, 4)).astype(np.float32)
    assert select_indices(descriptors, 5).tolist() == [3, 8, 0, 2, 7]


def test_joint_selection_preserves_descriptor_xyz_identity():
    descriptors = np.random.default_rng(7).normal(size=(30, 5)).astype(np.float32)
    xyz = np.arange(90, dtype=np.float32).reshape(30, 3)
    ids = np.arange(30, dtype=np.int64) + 100
    indices, compact = select_map(
        descriptors, 11, aligned_arrays={"xyz": xyz, "ids": ids}
    )
    assert np.array_equal(compact["descriptors"], descriptors[indices])
    assert np.array_equal(compact["xyz"], xyz[indices])
    assert np.array_equal(compact["ids"], ids[indices])


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        select_indices(np.empty((0, 4), dtype=np.float32), 1)
    with pytest.raises(ValueError):
        select_indices(np.ones((4, 2), dtype=np.float32), 5)
    with pytest.raises(ValueError):
        select_indices(np.asarray([[np.nan, 0.0]], dtype=np.float32), 1)
    with pytest.raises(ValueError):
        select_map(
            np.ones((4, 2), dtype=np.float32),
            2,
            aligned_arrays={"descriptors": np.ones((4, 3), dtype=np.float32)},
        )


def test_public_selector_has_no_query_or_label_inputs():
    forbidden = {"query", "pose", "ground_truth", "label", "error", "correctness"}
    assert not set(inspect.signature(select_indices).parameters) & forbidden
