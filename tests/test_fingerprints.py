from iChem.utils.fingerprints import binary_fps, real_fps, count_fps, _binary_fps
from iChem.utils import load_smiles
from contextlib import redirect_stdout
import io
import unittest
import time

class TestFingerprints(unittest.TestCase):

    def test_binary_fps(self):
        smiles = load_smiles('tests/data/molecules.smi')
        fps = binary_fps(smiles, n_bits=2048,
                         fp_type='ECFP4',
                         packed=True,
                         return_invalid=False)
        self.assertEqual(fps.shape, (118, 2048/8))

        fps = binary_fps(smiles, n_bits=2048,
                         fp_type='ECFP4',
                         packed=False,
                         return_invalid=False)
        self.assertEqual(fps.shape, (118, 2048))

    def test_real_fps(self):
        smiles = load_smiles('tests/data/molecules.smi')
        fps = real_fps(smiles)
        self.assertEqual(fps.shape, (118, 217))

    def test_count_fps(self):
        smiles = load_smiles('tests/data/molecules.smi')
        fps = count_fps(smiles, n_bits=1024)
        self.assertEqual(fps.shape, (118, 1024))

    def test_invalid_smiles(self):
        smiles = load_smiles('tests/data/molecules_invalids.smi')
        fps, invalid_ids = binary_fps(smiles,
                         return_invalid=True,
                         packed=True)
        self.assertEqual(fps.shape, (118, 2048/8))
        self.assertEqual(len(invalid_ids), 3)
        self.assertListEqual(invalid_ids, [6, 65, 120])

    def test_invalid_fp_type(self):
        smiles = load_smiles('tests/data/molecules.smi')[:1]
        with self.assertRaises(ValueError):
            _binary_fps(smiles, fp_type='NOT_A_FP_TYPE')

    def test_maccs_binary_fps(self):
        smiles = load_smiles('tests/data/molecules.smi')
        capture = io.StringIO()
        with redirect_stdout(capture):
            fps = binary_fps(smiles, fp_type='MACCS', packed=True)
        self.assertEqual(fps.shape, (118, 167))

    def test_count_fps_invalid_smiles(self):
        smiles = load_smiles('tests/data/molecules_invalids.smi')
        fps, invalid_ids = count_fps(smiles, return_invalid=True)
        self.assertEqual(fps.shape[0], 118)
        self.assertEqual(fps.shape[1], 2048)
        self.assertEqual(len(invalid_ids), 3)
        self.assertListEqual(invalid_ids, [6, 65, 120])

    def test_real_fps_invalid_smiles(self):
        smiles = load_smiles('tests/data/molecules_invalids.smi')
        fps, invalid_ids = real_fps(smiles, return_invalid=True)
        self.assertEqual(fps.shape, (118, 217))
        self.assertEqual(len(invalid_ids), 3)
        self.assertListEqual(invalid_ids, [6, 65, 120])

    def test_time_binary_fps(self):
        smiles = load_smiles('tests/data/molecules.smi')
        smiles = smiles * 100
        start_time_parallel = time.time()
        fps = binary_fps(smiles, n_bits=2048,
                         fp_type='ECFP4',
                         packed=True,
                         return_invalid=False)
        end_time_parallel = time.time()
        time_parallel = end_time_parallel - start_time_parallel

        start_time_serial = time.time()
        fps_serial = _binary_fps(smiles, n_bits=2048,
                                 fp_type='ECFP4',
                                 packed=True,
                                 return_invalid=False)
        end_time_serial = time.time()
        time_serial = end_time_serial - start_time_serial
        self.assertTrue(time_serial - time_parallel > 0)
        self.assertTrue(time_serial / time_parallel > 1.5)
        self.assertListEqual(fps.tolist(), fps_serial.tolist())