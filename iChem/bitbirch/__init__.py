r"""BitBIRCH-Lean, a high-throughput, memory-efficient implementation of BitBIRCH

BitBIRCH-Lean is designed for high-thorouput clustering of huge molecular
libraries (of up to hundreds of milliones of molecules).
"""
from .cluster import cluster
from .optimal_threshold import optimal_threshold
from bblean.bitbirch import BitBirch  # type: ignore
from bblean.fingerprints import pack_fingerprints, unpack_fingerprints  # type: ignore
from bblean.similarity import (  # type: ignore
    estimate_jt_std,
    jt_isim_medoid,
    jt_isim_packed,
    jt_sim_matrix_packed,
    jt_sim_packed,
    jt_stratified_sampling,
)

__all__ = [
    "BitBirch",
    "cluster",
    "estimate_jt_std",
    "jt_isim_medoid",
    "jt_isim_packed",
    "jt_sim_matrix_packed",
    "jt_sim_packed",
    "jt_stratified_sampling",
    "pack_fingerprints",
    "unpack_fingerprints",
]
