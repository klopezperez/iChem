import numpy as np # type: ignore
import pandas as pd # type: ignore
from ..iSIM import calculate_isim
from ..iSIM.real import pair_jt, pair_rr, pair_sm
from rdkit import Chem, DataStructs # type: ignore
from rdkit.Chem import Descriptors, rdFingerprintGenerator, MACCSkeys, SaltRemover # type: ignore
from multiprocessing import Pool, cpu_count # type: ignore
from .._config import CPU_CORES # type: ignore

"""
This module contains utility functions for the iChem package regarding fingerprint generation, and 
pairwise similarity calculations using RDKit functions.
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
                DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), fp)
                return fp
            def GetCountFingerprintAsNumPy(self, mol):
                fp = np.zeros((167,), dtype=np.uint8)
                DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), fp)
                patterns = MACCSkeys.smartsPatts
                for i in range(1, 167):
                    if fp[i] == 1:
                        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(patterns[i][0]))
                        fp[i] = max(len(matches), 1)
                return fp
        return MACCSGen()
    else:
        raise ValueError(f'Invalid fingerprint type: {fp_type}')

# NOTE: Fingerprint generation functions were moved to `iChem.utils.fingerprints`.
# The remaining utilities (normalization, pairwise comparisons, file loaders,
# and standardization helpers) remain in this module.

def minmax_norm(fps):
    """
    This function performs min-max normalization on the dataset. Required for the calculation of iSIM for real
    number fingerprints. Eliminates columns with NaN values or where all values are zero.

    Parameters:
    fps: numpy array of fingerprints

    Returns:
    fps: normalized numpy array of fingerprints
    """

    # Turn the array into a DataFrame
    df = pd.DataFrame(fps)

    # Normalize the data
    df_numeric = df.select_dtypes(include = [np.number])
    columns = df_numeric.columns

    for column in columns:
        min_prop = np.min(df[column])
        max_prop = np.max(df[column])

        if min_prop == max_prop:
            df = df.drop(column, axis = 1)
            continue

        try:
            df[column] = [(x - min_prop) / (max_prop - min_prop) for x in df[column]]
        except ZeroDivisionError:
            df.drop(column, axis = 1)

    df = df.dropna(axis = 'columns')

    # Return the normalized data as a numpy array
    return df.to_numpy()

def normalize_fps(fps):
    """
    This function performs min-max normalization on the dataset without using pandas.
    For columns with all zero values, the normalized value remains zero.

    Parameters:
    fps: numpy array of fingerprints

    Returns:
    fps: normalized numpy array of fingerprints
    """
    # Convert to numpy array if not already
    fps = np.array(fps, dtype=float)

    # Calculate the minimum and maximum for each column
    col_min = np.min(fps, axis=0)
    col_max = np.max(fps, axis=0)

    # Avoid division by zero: identify columns where min == max
    range_vals = col_max - col_min
    non_zero_range = range_vals != 0

    # Perform min-max normalization only for columns with non-zero range
    normalized_fps = np.zeros_like(fps, dtype=float)
    normalized_fps[:, non_zero_range] = (fps[:, non_zero_range] - col_min[non_zero_range]) / range_vals[non_zero_range]

    # Columns with zero range (all values are the same) remain zero
    return normalized_fps

def npy_to_rdkit(fps_np):
    """
    This function converts numpy array fingerprints to RDKit fingerprints.

    Parameters:
    fps_np: numpy array of fingerprints

    Returns:
    fp_rdkit: list of RDKit fingerprints
    """
    fp_len = len(fps_np[0])
    fp_rdkit = []
    for fp in fps_np:
        bitvect = DataStructs.ExplicitBitVect(fp_len)
        bitvect.SetBitsFromList(np.where(fp)[0].tolist())
        fp_rdkit.append(bitvect)
    
    return fp_rdkit


def rdkit_pairwise_sim(fingerprints, return_std: bool = False):
    """
    This function computes the pairwise similarity between all objects in the dataset using Jaccard-Tanimoto similarity.

    Parameters:
    fingerprints: list of fingerprints

    Returns:
    similarity: average similarity between all objects 
    and standard deviation if return_std is True
    """
    if type(fingerprints[0]) == np.ndarray:
        fingerprints = npy_to_rdkit(fingerprints)

    nfps = len(fingerprints)
    similarity = []

    for n in range(nfps - 1):
        sim = DataStructs.BulkTanimotoSimilarity(fingerprints[n], fingerprints[n+1:])
        similarity.extend([s for s in sim])

    if return_std:
        return np.mean(similarity), np.std(similarity)
    else:
        return np.mean(similarity)


def rdkit_pairwise_matrix(fingerprints: np.ndarray, fingerprints_2: np.ndarray = None):
    """
    This function computes the pairwise similarity between all objects in the dataset using Jaccard-Tanimoto similarity.

    Parameters:
    fingerprints: list of fingerprints

    Returns:
    similarity: matrix of similarity values
    """
    if type(fingerprints[0]) == np.ndarray:
        fingerprints = npy_to_rdkit(fingerprints)

    if fingerprints_2 is None:
        n = len(fingerprints)
        matrix = np.zeros((n, n))
        np.fill_diagonal(matrix, 1)  # Set diagonal values to 1

        # Fill the upper triangle directly while computing similarities
        for i in range(n - 1):
            sim = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[i + 1:])
            matrix[i, i + 1:] = sim
            matrix[i + 1:, i] = sim  # Mirror to the lower triangle

        return matrix
    else:
        if type(fingerprints_2[0]) == np.ndarray:
            fingerprints_2 = npy_to_rdkit(fingerprints_2)

        n1 = len(fingerprints)
        n2 = len(fingerprints_2)
        matrix = np.zeros((n1, n2))

        for i in range(n1):
            sim = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints_2)
            matrix[i, :] = sim
        
        return matrix
  
def pairwise_average(fingerprints: np.ndarray, n_ary: str = 'JT', return_std: bool = False):
    """
    This function computes the pairwise average similarity between all objects in the dataset.
    
    Parameters:
    fingerprints: numpy array of fingerprints
    n_ary: type of similarity to index compute
    
    Returns:
    average: average similarity between all objects
    and standard deviation if return_std is True
    """
   
    # Compute the pairwise similarities
    pairwise_sims = []
    for i in range(len(fingerprints) - 1):
        for j in range(i + 1, len(fingerprints)):
            pairwise_sims.append(calculate_isim(np.array([fingerprints[i], fingerprints[j]]), n_ary = n_ary))

    # Compute the average similarity
    average = np.mean(pairwise_sims)
    
    if return_std:
        return average, np.std(pairwise_sims)
    else:
        return average
    
def pairwise_average_real(fingerprints: np.ndarray, n_ary: str = 'JT', return_std: bool = False):
    """
    This function computes the pairwise average similarity between all objects in the dataset.
    
    Parameters:
    fingerprints: numpy array of fingerprints
    n_ary: type of similarity to index compute
    
    Returns:
    average: average similarity between all objects
    and standard deviation if return_std is True
    """
    # Normalize the fingerprints
    normalized_fps = minmax_norm(fingerprints)

    # Define the comparison function
    if n_ary == 'JT':
        compare = pair_jt
    elif n_ary == 'RR':
        compare = pair_rr
    elif n_ary == 'SM':
        compare = pair_sm
    else:
        print('Invalid similarity index: ', n_ary)
        exit(0)
    
    # Compute the pairwise similarities
    pairwise_sims = []
    for i in range(len(normalized_fps) - 1):
        for j in range(i + 1, len(normalized_fps)):
            pairwise_sims.append(compare(normalized_fps[i], normalized_fps[j]))

    # Compute the average similarity
    average = np.mean(pairwise_sims)
    
    if return_std:
        return average, np.std(pairwise_sims)
    else:
        return average
    
def load_smiles(file_path: str, standarize: bool = False) -> list:
    """
    This function loads SMILES strings from a file.
    
    Parameters:
    file_path: path to the file containing SMILES strings
    
    Returns:
    smiles: list of SMILES strings
    """
    with open(file_path, 'r') as f:
        smiles = [line.split('\t', 1)[0].split(' ')[0].strip() for line in f if line.strip()]

    if standarize:
        standardized_smiles = []
        for smi in smiles:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    standardized_mol = smiles_standarization(mol)
                    standardized_smiles.append(Chem.MolToSmiles(standardized_mol))
                else:
                    print('Invalid SMILES: ', smi)
            except Exception as e:
                print(f'Error processing SMILES {smi}: {e}')
        return standardized_smiles
        
    return smiles

def load_smiles_and_ids(file_path: str) -> tuple:
    """
    This function loads SMILES and IDs from a file.
    
    Parameters:
    file_path: path to the file containing SMILES strings
    
    Returns:
    smiles: list of SMILES strings
    ids: list of corresponding IDs (indices) for the SMILES strings
    """
    # Read SMILES and IDs from the file, each line has the format: "SMILES\tID"
    smiles = []
    ids = []

    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                smi, idx = line.split('\t', 1)
                smiles.append(smi.split(' ')[0].strip())
                ids.append(idx.strip())
        
    return smiles, ids

def load_multiple_smiles(dir_path: str,
                         standarize: bool = False,
                         gzipped: bool = False) -> list:
    """
    This function loads SMILES strings from multiple files in a directory.
    
    Parameters:
    -----------
    dir_path: str
        Path to the directory containing files with SMILES strings
    standarize: bool, default=False
        Whether to standardize SMILES strings
    gzipped: bool, default=False
        If True, load .smi.gz files; if False, load .smi files
    
    Returns:
    --------
    smiles: list
        List of SMILES strings
    """
    from pathlib import Path

    dir_path = Path(dir_path)
    file_pattern = '*.smi.gz' if gzipped else '*.smi'
    loader_func = load_smiles_gzipped if gzipped else load_smiles

    smiles_files = sorted(dir_path.glob(file_pattern))

    all_smiles = []
    for file in smiles_files:
        smiles = loader_func(file, standarize=standarize)
        all_smiles.extend(smiles)

    return all_smiles

def load_smiles_gzipped(file_path: str, standarize: bool = False) -> list:
    """
    This function loads SMILES strings from a gzipped file.
    
    Parameters:
    file_path: path to the gzipped file containing SMILES strings
    
    Returns:
    smiles: list of SMILES strings
    """
    import gzip

    smiles = []
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            if line.strip():
                smiles.append(line.split('\t', 1)[0].split(' ')[0].strip())

    if standarize:
        standardized_smiles = []
        for smi in smiles:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    standardized_mol = smiles_standarization(mol)
                    standardized_smiles.append(Chem.MolToSmiles(standardized_mol))
                else:
                    print('Invalid SMILES: ', smi)
            except Exception as e:
                print(f'Error processing SMILES {smi}: {e}')
        return standardized_smiles
        
    return smiles

def smiles_standarization(mol) -> Chem.Mol:
    """
    This function standardizes a molecule by removing salts and keeping the largest fragment.

    Parameters:
    mol: RDKit molecule object

    Returns:
    standardized_mol: standardized RDKit molecule object
    """
    remover = SaltRemover.SaltRemover()
    mol = remover.StripMol(mol)
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        largest_frag = max(frags, key=lambda m: m.GetNumAtoms())
        return largest_frag
    else:
        return mol
    