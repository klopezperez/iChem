import iChem.utils as utils
import iChem.iSIM as iSIM
from rdkit import DataStructs
import pytest
import numpy as np
import pandas as pd

fps = np.load('data/RDKIT_fps.npy')

# Test the npy_to_rdkit function
def test_npy_to_rdkit():
    # Convert numpy array to RDKit fingerprints
    fp_rdkit = utils.npy_to_rdkit(fps)
    assert type(fp_rdkit[0]) == DataStructs.cDataStructs.ExplicitBitVect

# Test the rdkit_pairwise_sim function
def test_rdkit_pairwise_sim():
    # Calculate the pairwise similarity
    value = utils.rdkit_pairwise_sim(fps)
    assert value == pytest.approx(0.198477)

# Test the rdkit_pairwise_matrix function
def test_rdkit_pairwise_matrix():
    # Calculate the pairwise matrix
    value = utils.rdkit_pairwise_matrix(fps)

    dimensions = value.shape
    assert dimensions == (119, 119)
    assert value.diagonal().sum() == 119
    assert value[0, 1] == iSIM.calculate_isim(np.array([fps[0], fps[1]]), n_ary="JT")

# Test the real_fps function and normalization
def test_real_fps():
    # Calculate the real fingerprints
    smiles = pd.read_csv('data/logP_data.csv')
    smiles = smiles['SMILES']
    fps = utils.real_fps(smiles)
    assert type(fps) == np.ndarray
    assert fps.shape[0] == 119
    assert np.nan not in fps

    # Calculate the minmax normalization
    value_norm = utils.minmax_norm(fps)
    assert value_norm.min() == 0
    assert value_norm.max() == 1
    assert value_norm.shape