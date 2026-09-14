"""D2Map-Loc: descriptor-diverse fixed-budget map selection."""

from .selector import normalize_descriptors, select_indices, select_map

__all__ = ["normalize_descriptors", "select_indices", "select_map"]
__version__ = "0.1.0"
