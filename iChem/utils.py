import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

"""
This module contains utility functions for the iChem package regarding fingerprint generation, and 
pairwise similarity calculations using RDKit functions.
"""
def binary_fps(smiles: list, fp_type: str = 'RDKIT', n_bits: int = 2048):
    """
    This function generates binary fingerprints for the dataset.
    
    Parameters:
    smiles: list of SMILES strings
    fp_type: type of fingerprint to generate ['RDKIT', 'ECFP4', 'ECFP6', or 'MACCS']
    n_bits: number of bits for the fingerprint
    
    Returns:
    fingerprints: numpy array of fingerprints
    """
    # Generate the fingerprints
    if fp_type == 'RDKIT':
       def generate_fp(mol, fp):
            DataStructs.cDataStructs.ConvertToNumpyArray(Chem.RDKFingerprint(mol), fp)
    elif fp_type == 'ECFP4':
        def generate_fp(mol, fp):
            DataStructs.cDataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits), fp)
    elif fp_type == 'ECFP6':
        def generate_fp(mol, fp):
            DataStructs.cDataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=n_bits), fp)
    elif fp_type == 'MACCS':
        def generate_fp(mol, fp):
            DataStructs.cDataStructs.ConvertToNumpyArray(Chem.rdMolDescriptors.GetMACCSKeysFingerprint(mol), fp)
    else:
        print('Invalid fingerprint type: ', fp_type)
        exit(0)

    fingerprints = []
    for smi in smiles:
        # Generate the mol object
        try:
          mol = Chem.MolFromSmiles(smi)
        except:
          print('Invalid SMILES: ', smi)
          exit(0)

        # Generate the fingerprint and append to the list
        fingerprint = np.array([])
        generate_fp(mol, fingerprint)
        fingerprints.append(fingerprint)
    
    fingerprints = np.array(fingerprints)

    return fingerprints

def real_fps(smiles):
    """
    This function generates real number fingerprints for the dataset based on RDKit descriptors.
    Skips corrupted smiles strings. 
    
    Parameters:
    smiles: list of SMILES strings
    
    Returns:
    fingerprints: numpy array of fingerprints
    """
    fps = []
    for smi in smiles:
        # Generate the mol object
        try:
          mol = Chem.MolFromSmiles(smi)
        except:
          print('Invalid SMILES: ', smi)
          exit(0)

        # Generate the fingerprint and append to the list
        des = []
        for nm, fn in Descriptors._descList:
            try: 
                val = fn(mol)
            except:
                print('Error computing descriptor: ', nm)
                val = 'NaN'
            des.append(val)

        fps.append(des)
    
    # Drop columns with NaN values
    fps = np.array(fps)
    fps = fps[:, ~np.isnan(fps).any(axis = 0)]
    
    return fps

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


def rdkit_pairwise_sim(fingerprints):
    """
    This function computes the pairwise similarity between all objects in the dataset using Jaccard-Tanimoto similarity.

    Parameters:
    fingerprints: list of fingerprints

    Returns:
    similarity: average similarity between all objects
    """
    if type(fingerprints[0]) == np.ndarray:
        fingerprints = npy_to_rdkit(fingerprints)

    nfps = len(fingerprints)
    similarity = []

    for n in range(nfps - 1):
        sim = DataStructs.BulkTanimotoSimilarity(fingerprints[n], fingerprints[n+1:])
        similarity.extend([s for s in sim])

    return np.mean(similarity)


def rdkit_pairwise_matrix(fingerprints):
    """
    This function computes the pairwise similarity between all objects in the dataset using Jaccard-Tanimoto similarity.

    Parameters:
    fingerprints: list of fingerprints

    Returns:
    similarity: matrix of similarity values
    """
    if type(fingerprints[0]) == np.ndarray:
        fingerprints = npy_to_rdkit(fingerprints)

    n = len(fingerprints)
    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, 1)  # Set diagonal values to 1

    # Fill the upper triangle directly while computing similarities
    for i in range(n - 1):
        sim = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[i + 1:])
        for j, s in enumerate(sim):
            matrix[i, i + 1 + j] = s  # Map similarities to the correct indices in the upper triangle

    return matrix