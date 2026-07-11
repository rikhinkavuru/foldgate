"""foldgate.io -- parse released predictions into unified Prediction records.

Currently: Runs N' Poses (`rnp.py`). Native AF3/Boltz/Chai + FoldBench adapters
land here next; confirm field names against a real output file before parsing
(see CLAUDE.md -- AF3 emits no PDE, Boltz uses NPZ, etc.).
"""

from .foldbench import load_foldbench
from .rnp import RNP_METHODS, RMSD_THRESHOLD_A, RNPData, load_rnp

__all__ = ["RNPData", "load_rnp", "RNP_METHODS", "RMSD_THRESHOLD_A", "load_foldbench"]
