import numpy as np
import pytzer as pz

mols = np.array([0.5, 0.5])
ions = np.array(['Na', 'Cl'])
tots = np.array([1.0])
eles = np.array(['t_HSO4'])
mols, tots = pz.io.solution2solvent(mols, ions, tots, eles)

s = pz.io.salinity2mols(35)
