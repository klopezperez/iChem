import iChem.iSIM.iSIM as iSIM
import numpy as np
import pytest

# Read the fingerprint 
fps = np.load('tests/data/RDKIT_fps.npy')

# Test the calculate_isim function
def test_calculate_isim():
    # Calculate the iSIM
    value = iSIM.calculate_isim(fps, n_ary="JT")
    assert value == pytest.approx(0.2135206)

# Test the calculate_medoid function
def test_calculate_medoid():
    # Calculate the medoid
    value = iSIM.calculate_medoid(fps)
    assert value == 63

# Test the calculate_outlier function
def test_calculate_outlier():
    # Calculate the outlier
    value = iSIM.calculate_outlier(fps)
    assert value == 80