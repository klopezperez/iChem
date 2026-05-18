import numpy as np  # type: ignore
from rdkit import Chem, DataStructs  # type: ignore
from rdkit.Chem import Descriptors, rdFingerprintGenerator, MACCSkeys  # type: ignore
from multiprocessing import Pool, cpu_count  # type: ignore
from .._config import CPU_CORES  # type: ignore
from .utils import smiles_standarization

"""
This module contains utility functions for the iChem package regarding
fingerprint generation, and pairwise similarity calculations using RDKit.
"""


def _get_generator(fp_type: str, n_bits: int):
    """Helper function to get the appropriate fingerprint generator"""
    if fp_type == 'RDKIT':
        return rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5, fpSize=n_bits)
    elif fp_type == 'ECFP4':
        return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    elif fp_type == 'ECFP6':
        return rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=n_bits)
    elif fp_type == 'AP':
        return rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
    elif fp_type == 'TT':
        return rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits)
    elif fp_type == 'MACCS':
        class MACCSGen:
            def GetFingerprintAsNumPy(self, mol):
                fp = np.zeros((167,), dtype=np.uint8)
                DataStructs.ConvertToNumpyArray(
                    MACCSkeys.GenMACCSKeys(mol), fp
                )
                return fp

            def GetCountFingerprintAsNumPy(self, mol):
                fp = np.zeros((167,), dtype=np.uint8)
                DataStructs.ConvertToNumpyArray(
                    MACCSkeys.GenMACCSKeys(mol), fp
                )
                patterns = MACCSkeys.smartsPatts
                for i in range(1, 167):
                    if fp[i] == 1:
                        matches = mol.GetSubstructMatches(
                            Chem.MolFromSmarts(patterns[i][0])
                        )
                        fp[i] = max(len(matches), 1)
                return fp

        return MACCSGen()
    else:
        raise ValueError(f'Invalid fingerprint type: {fp_type}')


def binary_fps(smiles: list,
               fp_type: str = 'RDKIT',
               n_bits: int = 2048,
               return_invalid: bool = False,
               standarize: bool = False,
               packed: bool = False):
    """This function generates binary fingerprints for the dataset.

    Parallelized across CPU cores.

    Parameters:
        smiles: list of SMILES strings
        fp_type: type of fingerprint to generate
            ['RDKIT', 'ECFP4', 'ECFP6', 'AP', 'TT', or 'MACCS']
        n_bits: number of bits for the fingerprint (ignored for MACCS)
        return_invalid: whether to return invalid SMILES indices
        standarize: whether to standardize molecules
        packed: whether to return packed fingerprints
            (not supported for MACCS)

    Returns:
        fingerprints: numpy array of fingerprints
        and list of invalid SMILES indices if return_invalid is True
    """

    # Divide the smiles into chunks and generate fingerprints for each chunk
    # in parallel
    n_cpus = min(cpu_count(), CPU_CORES)
    smiles_chunks = np.array_split(smiles, n_cpus)

    # Create list of (chunk, chunk_offset, ...) tuples for parallel processing
    chunk_tasks = []
    offset = 0
    for chunk in smiles_chunks:
        chunk_tasks.append(
            (chunk, offset, fp_type, n_bits, return_invalid,
             standarize, packed)
        )
        offset += len(chunk)

    with Pool(n_cpus) as pool:
        results = pool.starmap(_binary_fps, chunk_tasks)

    # Concatenate the results from all chunks
    if return_invalid:
        fps = np.concatenate([res[0] for res in results])
        invalid_indices = [idx for res in results for idx in res[1]]
        if invalid_indices:
            print(
                f"Warning: {len(invalid_indices)} invalid SMILES found at "
                f"indices: {invalid_indices}"
            )
            print(
                "There might be issues in the order of the valid "
                "fingerprints due to the invalid SMILES. Consider checking "
                "the invalid SMILES and their indices."
            )
        return fps, invalid_indices
    else:
        fps = np.concatenate(results)
        return fps


def _binary_fps(smiles: list,
                chunk_offset: int = 0,
                fp_type: str = 'RDKIT',
                n_bits: int = 2048,
                return_invalid: bool = False,
                standarize: bool = False,
                packed: bool = False) -> np.ndarray:
    """
    This function generates binary fingerprints for the dataset.

    Parameters:
    -----------
    smiles: list of SMILES strings
    fp_type: type of fingerprint to generate
        ['RDKIT', 'ECFP4', 'ECFP6', or 'MACCS']
    n_bits: number of bits for the fingerprint
    return_invalid: whether to return invalid SMILES indices
    packed: whether to return packed fingerprints

    Returns:
    --------
    fingerprints: numpy array of fingerprints
    and list of invalid SMILES indices if return_invalid is True
    """
    # Generate the fingerprints
    fps_gen = _get_generator(fp_type, n_bits)

    # MACCS does not support packed output; enforce unpacked
    if fp_type == 'MACCS' and packed:
        print(
            'Warning: packed=True is not supported for MACCS; '
            'using unpacked (packed=False).'
        )
        packed = False

    # Determine fingerprint size
    if fp_type == 'MACCS':
        fp_size = 167
    else:
        fp_size = n_bits if not packed else n_bits // 8

    # Pre-allocate numpy array for all fingerprints
    fingerprints = np.empty((len(smiles), fp_size), dtype=np.uint8)
    valid_idx = 0
    invalid_smiles = []

    for k, smi in enumerate(smiles):
        # Generate the mol object
        try:
            mol = Chem.MolFromSmiles(smi)
            if standarize:
                mol = smiles_standarization(mol)
        except Exception:
            print('Invalid SMILES: ', smi)
            invalid_smiles.append(k)
            continue

        try:
            # Generate the fingerprint and store directly in array
            fingerprint = fps_gen.GetFingerprintAsNumPy(mol)
            if packed:
                fingerprint = np.packbits(fingerprint)
            fingerprints[valid_idx] = fingerprint
            valid_idx += 1
        except Exception:
            print('Error generating fingerprint for SMILES: ', smi)
            invalid_smiles.append(k)

    # Trim array to only include valid fingerprints
    fingerprints = fingerprints[:valid_idx]

    if return_invalid:
        adjusted_invalid = [idx + chunk_offset for idx in invalid_smiles]
        return fingerprints, adjusted_invalid
    else:
        return fingerprints


def count_fps(smiles: list,
              fp_type: str = 'RDKIT',
              n_bits: int = 2048,
              return_invalid: bool = True) -> np.ndarray:
    """
    This function generates count-based fingerprints for the dataset.

    Parallelized across CPU cores.

    Parameters:
    -----------
    smiles: list of SMILES strings
    fp_type: type of fingerprint to generate ['RDKIT', 'ECFP4', 'ECFP6']
    n_bits: number of bits for the fingerprint
    return_invalid: whether to return invalid SMILES indices

    Returns:
    --------
    fingerprints: numpy array of count fingerprints
    and list of invalid SMILES indices if return_invalid is True
    """

    smiles_chunks = np.array_split(smiles, cpu_count())

    # Create list of (chunk, chunk_offset, ...) tuples for parallel processing
    chunk_tasks = []
    offset = 0
    for chunk in smiles_chunks:
        chunk_tasks.append((chunk, offset, fp_type, n_bits, return_invalid))
        offset += len(chunk)

    with Pool(cpu_count()) as pool:
        results = pool.starmap(_count_fps, chunk_tasks)

    if return_invalid:
        fps = np.concatenate([res[0] for res in results])
        invalid_indices = [idx for res in results for idx in res[1]]
        if invalid_indices:
            print(
                f"Warning: {len(invalid_indices)} invalid SMILES found at "
                f"indices: {invalid_indices}"
            )
            print(
                "There might be issues in the order of the valid "
                "fingerprints due to the invalid SMILES. Consider checking "
                "the invalid SMILES and their indices."
            )
        return fps, invalid_indices
    else:
        fps = np.concatenate(results)
        return fps


def _count_fps(smiles: list,
               chunk_offset: int = 0,
               fp_type: str = 'RDKIT',
               n_bits: int = 2048,
               return_invalid: bool = True) -> np.ndarray:
    """
    This function generates count-based fingerprints for the dataset.

    Parameters:
    -----------
    smiles: list of SMILES strings
    fp_type: type of fingerprint to generate ['RDKIT', 'ECFP4', 'ECFP6']
    n_bits: number of bits for the fingerprint
    return_invalid: whether to return invalid SMILES indices

    Returns:
    --------
    fingerprints: numpy array of count fingerprints
    and list of invalid SMILES indices if return_invalid is True
    """
    # Generate the fingerprint generator
    fps_gen = _get_generator(fp_type, n_bits)

    # Determine fingerprint size and dtype
    # (counts can exceed 255 for some types)
    if fp_type == 'MACCS':
        fp_size = 167
        dtype = np.uint8
    else:
        fp_size = n_bits
        dtype = np.uint16  # smaller than int64, avoids overflow vs uint8

    # Pre-allocate numpy array for all fingerprints
    fingerprints = np.empty((len(smiles), fp_size), dtype=dtype)
    valid_idx = 0
    invalid_smiles = []

    for k, smi in enumerate(smiles):
        # Generate the mol object
        try:
            mol = Chem.MolFromSmiles(smi)
        except Exception:
            print('Invalid SMILES: ', smi)
            invalid_smiles.append(k)
            continue

        try:
            # Generate the count fingerprint and store directly in array
            fingerprint = fps_gen.GetCountFingerprintAsNumPy(mol)
            fingerprints[valid_idx] = fingerprint
            valid_idx += 1
        except Exception:
            print('Error generating fingerprint for SMILES: ', smi)
            invalid_smiles.append(k)

    # Trim array to only include valid fingerprints
    fingerprints = fingerprints[:valid_idx]

    if return_invalid:
        adjusted_invalid = [idx + chunk_offset for idx in invalid_smiles]
        return fingerprints, adjusted_invalid
    else:
        return fingerprints


def real_fps(smiles, return_invalid: bool = False):
    """
    This function generates real number fingerprints for the dataset.

    Based on RDKit descriptors. Skips corrupted smiles strings.

    Parameters:
    -----------
    smiles: list of SMILES strings
    return_invalid: whether to return invalid SMILES indices

    Returns:
    --------
    fingerprints: numpy array of fingerprints
    and list of invalid SMILES indices if return_invalid is True
    """
    fps = []
    invalid_smiles = []
    for k, smi in enumerate(smiles):
        # Generate the mol object
        try:
            mol = Chem.MolFromSmiles(smi)
            try:
                des = []
                for nm, fn in Descriptors._descList:
                    val = fn(mol)
                    des.append(val)
                fps.append(des)
            except Exception:
                print('Error computing descriptor: ', nm)
                invalid_smiles.append(k)
                continue
        except Exception:
            print('Invalid SMILES: ', smi)
            invalid_smiles.append(k)

    # Convert to numpy array
    fps = np.array(fps)
    if return_invalid:
        return fps, invalid_smiles
    else:
        return fps
