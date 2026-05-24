"""Internal CLI helper functions."""

from pathlib import Path


def load_smiles(input_path: Path) -> list:
    """Load SMILES from .smi or .smi.gz files."""
    from .utils.utils import load_multiple_smiles, load_smiles as _load_smiles, load_smiles_gzipped

    if input_path.is_file():
        if input_path.suffix == '.smi':
            return _load_smiles(input_path)
        elif input_path.suffix == '.gz':
            return load_smiles_gzipped(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
    elif input_path.is_dir():
        # Load both .smi and .smi.gz files from directory
        smiles = []
        smiles.extend(load_multiple_smiles(input_path, gzipped=False))
        smiles.extend(load_multiple_smiles(input_path, gzipped=True))
        return smiles
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")


def get_smi_files(input_path: Path) -> list[Path]:
    """Get list of .smi and .smi.gz files from input path."""
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        smi_files = sorted(input_path.glob('*.smi')) + sorted(input_path.glob('*.smi.gz'))
        return smi_files
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")


def get_output_path(input_file: Path, fp_type: str, suffix: str, out_dir: Path = None) -> Path:
    """Generate output filename based on input file and fingerprint type.
    
    Args:
        input_file: Input .smi or .smi.gz file
        fp_type: Fingerprint type (e.g., 'ECFP4', 'RDKIT')
        suffix: Output file suffix (e.g., '.npy')
        out_dir: Output directory (defaults to input file's parent)
    
    Returns:
        Output path with format: <stem>_<fp_type><suffix>
    """
    if out_dir is None:
        out_dir = input_file.parent
    
    stem = input_file.stem
    if stem.endswith('.smi'):
        stem = stem[:-4]
    
    return out_dir / f"{stem}_{fp_type}{suffix}"
