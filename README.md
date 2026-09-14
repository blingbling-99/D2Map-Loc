# D²Map-Loc

Official core implementation of **D²Map-Loc: Descriptor-Diverse Map Selection
for Budgeted Camera Localization** (ICASSP 2027 submission).

D²Map-Loc is a deterministic, training-free selector for retaining exactly
`K` original descriptor/XYZ pairs from a completed localization map. It covers
normalized descriptor space while leaving descriptor values, coordinates, the
Gaussian representation, and the localization backend unchanged.

This public repository intentionally contains the **core method only**. It does
not redistribute 7-Scenes, ULF-Loc, pretrained weights, Gaussian maps, paper
artifacts, internal experiment logs, or evaluation caches.

## Method

Given map descriptors `D` with shape `[N, d]`, D²Map-Loc:

1. L2-normalizes each descriptor for selection only.
2. Selects the row least aligned with the normalized global descriptor mean.
3. Repeatedly adds the unselected row with the smallest maximum cosine
   similarity to the selected set.
4. Uses the resulting indices to subset descriptors and XYZ coordinates
   jointly, preserving their original identities and values.

Exact ties use the smallest original row index. The implementation stores one
coverage value per row and does not materialize an `N × N` similarity matrix.
Its complexity is `O(KNd)` time and `O(Nd + N + K)` space.

## Installation

Python 3.9 or newer is required.

```bash
git clone https://github.com/blingbling-99/D2Map-Loc.git
cd D2Map-Loc
python -m pip install -e .
```

Install the CUDA-enabled PyTorch build appropriate for your system before the
editable installation if GPU selection is needed.

## Python API

```python
import numpy as np
from d2map_loc import select_map

descriptors = np.load("map_descriptors.npy")  # [N, D]
xyz = np.load("map_xyz.npy")                  # [N, 3]

indices, compact = select_map(
    descriptors,
    budget=10_000,
    device="cuda",
    aligned_arrays={"xyz": xyz},
)

np.save("selected_indices.npy", indices)
np.save("compact_descriptors.npy", compact["descriptors"])
np.save("compact_xyz.npy", compact["xyz"])
```

## Command line

Inputs use NumPy's non-pickled `.npy` format. `--xyz` and `--ids` are optional,
but when supplied they are subset with exactly the same selected rows.

```bash
d2map-loc \
  --descriptors map_descriptors.npy \
  --xyz map_xyz.npy \
  --ids gaussian_ids.npy \
  --budget 10000 \
  --device cuda \
  --output-dir outputs/scene_name_10k
```

The output directory contains selected indices and arrays plus `manifest.json`
with shapes and SHA-256 hashes. The directory must not already exist, which
helps prevent accidental overwrites.

## Reproducibility boundary

Selection consumes map descriptors only. It does not accept query images,
query poses, correctness labels, or localization errors. For exact replay, keep
the original input row order, dtype, PyTorch version, and compute device fixed.
The paper experiments used float32 descriptors and a CUDA device.

Run the public regression tests with:

```bash
python -m pip install -e ".[test]"
pytest -q
```

## Using D²Map-Loc with ULF-Loc

Export the completed ULF-Loc searchable descriptor rows and their aligned XYZ
coordinates as `.npy` arrays, run D²Map-Loc once offline, then pass the selected
descriptor/XYZ arrays to the unchanged ULF-Loc matching and localization
stages. ULF-Loc itself remains an external dependency and is not vendored here.

## Citation

The final proceedings metadata will be added after acceptance. Until then,
please cite the submission:

```bibtex
@inproceedings{liu2027d2maploc,
  title     = {D$^{2}$Map-Loc: Descriptor-Diverse Map Selection for Budgeted Camera Localization},
  author    = {Liu, Shanshan and Zhang, Tianshuo and Wang, Wei},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2027}
}
```

GitHub also reads the repository's [`CITATION.cff`](CITATION.cff) and exposes a
“Cite this repository” entry.

## Acknowledgements

The paper evaluates D²Map-Loc with the frozen ULF-Loc localization backend.
Please also follow ULF-Loc's repository and paper citation requirements when
using that system.

## License

This core implementation is released under the [MIT License](LICENSE).
