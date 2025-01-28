import iChem.iSIM.sampling as sampling
import iChem.iSIM.iSIM as iSIM
import numpy as np

# Load fingerprints for the test
fingerprints = np.load('data/MACCS_fps.npy')

# Calculate comp sim
comp_sim = iSIM.calculate_comp_sim(fingerprints, n_ary = 'JT')

order = np.argsort(comp_sim)

print(order)