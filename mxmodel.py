import pytzer as pz
from autograd import numpy as np

mols = np.array([1.5, 1.5])
zs = np.array([1.0, -1.0])

Istr_pz = pz.model.Istr(mols, zs)
Istr_mx = pz.matrix.Istr(mols, zs)

