from iChem.utils.fingerprints import binary_fps, real_fps, count_fps
from iChem.utils import load_smiles
import unittest

class TestFingerprints(unittest.TestCase):

    def test_binary_fps(self):
        smiles = load_smiles('tests/data/molecules.smi')
        fps = binary_fps(smiles, n_bits=2048,
                         fp_type='ECFP4',
                         packed=True,
                         return_invalid=False)
        self.assertEqual(fps.shape, (119, 2048/8))

        fps = binary_fps(smiles, n_bits=2048,
                         fp_type='ECFP4',
                         packed=False,
                         return_invalid=False)
        self.assertEqual(fps.shape, (119, 2048))

    def test_real_fps(self):
        smiles = load_smiles('tests/data/molecules.smi')
        fps = real_fps(smiles)
        self.assertEqual(fps.shape[0], 119)

    def test_count_fps(self):
        smiles = load_smiles('tests/data/molecules.smi')
        fps = count_fps(smiles)
        self.assertEqual(fps.shape[0], 119)