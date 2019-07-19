from autograd import numpy as np
import pytzer as pz

# -mu_i/(R*T) at 25 degC [P91 book, p. 445]
potentials = {
    'H2O': 95.6635,
    'Na': 105.651,
    'Cl': 52.955,
    'NaCl': 154.99,
}
mols = np.array([[1.0], [1.0]])*6.095
ions = np.array(['Na', 'Cl'])
tempK = np.array([298.15])
pres = np.array([10.10325])
lnksolNaCl = -potentials['NaCl'] + (potentials['Na'] + potentials['Cl'])
acfs = pz.model.acfs(mols, ions, tempK, pres)
lnksolNaCl_test = np.sum(np.log(acfs*mols))
