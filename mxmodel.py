import pytzer as pz
from autograd import numpy as np

# Define inputs
mols_mx = np.array([1.5, 1.5])
ions = np.array(['Na', 'Cl'])
tempK = np.array([298.15])
pres = np.array([10.1325])

# Calculate intermediates
mols_pz = np.vstack(mols_mx)
zs_pz = pz.props.charges(ions)[0]
zs_mx = zs_pz.ravel()
Aosm = pz.debyehueckel.Aosm_AW90(tempK, pres)[0][0]

# Calculate ionic strength
Istr_pz = pz.model.Istr(mols_pz, zs_pz)
Istr_mx = pz.matrix.Istr(mols_mx, zs_mx)

# Calculate Debye-Hueckel term
fG_pz = pz.model.fG(tempK, pres, Istr_pz, pz.cflibs.Seawater)
fG_mx = pz.matrix.fG(Aosm, Istr_mx)


