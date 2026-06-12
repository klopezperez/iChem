import numpy as np  # type: ignore
from typing import Union

from bblean.fingerprints import pack_fingerprints, unpack_fingerprints  # type: ignore

from ..utils.utils import binary_fps
from ..utils.utils import load_smiles as _load_smiles
from ..utils.utils import load_smiles_gzipped as _load_smiles_gzipped


class LibChem:
    """Container for a named representative SMILES library."""

    def __init__(
            self,
            name: str,
            smiles: Union[str, list],
    ):
        if not name:
            raise ValueError("Please specify a name for the library.")
        if not smiles:
            raise ValueError("Please provide SMILES data to load.")

        self.name = name
        self.fps_packed = None
        self.smiles = None
        self.n_molecules = 0
        self.flags = None

        self.load_smiles(smiles)

    def load_smiles(
            self,
            smiles: Union[str, list],
    ) -> None:
        """Load representative SMILES strings from a list or file."""
        if isinstance(smiles, str):
            if smiles.endswith(".smi.gz") or smiles.endswith(".gz"):
                loaded_smiles = _load_smiles_gzipped(smiles)
            else:
                loaded_smiles = _load_smiles(smiles)
        elif isinstance(smiles, list):
            loaded_smiles = smiles
        else:
            raise TypeError(
                f"smiles must be a string path or list, got {type(smiles)}"
            )

        if self.fps_packed is not None and len(loaded_smiles) != self.n_molecules:
            raise ValueError("Number of SMILES does not match number of fingerprints.")

        self.smiles = loaded_smiles
        self.n_molecules = len(self.smiles)
        self.set_flags()

    def load_fingerprints(
            self,
            fingerprints: Union[str, np.ndarray],
            packed: bool = True,
    ) -> None:
        """Load representative fingerprints from a .npy file or numpy array."""
        if isinstance(fingerprints, str):
            fps = np.load(fingerprints, mmap_mode="r")
        elif isinstance(fingerprints, np.ndarray):
            fps = fingerprints
        else:
            raise TypeError(
                "fingerprints must be a .npy path or numpy array, "
                f"got {type(fingerprints)}"
            )

        if fps.shape[0] != self.n_molecules:
            raise ValueError("Number of fingerprints does not match number of SMILES.")

        self.fps_packed = fps if packed else pack_fingerprints(fps)

    def set_flags(self) -> None:
        """Set one origin flag per representative molecule."""
        self.flags = [self.name] * self.n_molecules

    def generate_fingerprints(
            self,
            fp_type: str = "ECFP4",
            n_bits: int = 2048,
    ) -> None:
        """Generate packed binary fingerprints for the representative SMILES."""
        if self.smiles is None:
            raise ValueError("SMILES data not loaded.")

        fps, invalid = binary_fps(
            self.smiles,
            fp_type=fp_type,
            n_bits=n_bits,
            return_invalid=True,
            packed=True,
        )

        if invalid:
            raise ValueError(
                f"{len(invalid)} invalid SMILES were found during fingerprint generation."
            )

        self.fps_packed = fps

    def get_fingerprints(
            self,
            packed: bool = True,
    ) -> np.ndarray:
        """Retrieve representative fingerprints in packed or unpacked format."""
        if self.fps_packed is None:
            raise ValueError("Fingerprints not loaded or generated.")

        if packed:
            return self.fps_packed
        return unpack_fingerprints(self.fps_packed)

    def get_flags(self) -> list:
        """Retrieve origin flags for the representative molecules."""
        if self.flags is None:
            raise ValueError("Flags not computed.")
        return self.flags

    def empty_fingerprints(self) -> None:
        """Delete fingerprints from memory."""
        self.fps_packed = None
