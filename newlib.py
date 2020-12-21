import numpy as np, pytzer as pz
from pytzer.libraries import Seawater as plib
from pytzer import unsymmetrical as unsym

params = plib.get_parameters(
    cations=["Na", "Ca"], anions=["Cl", "SO4"], neutrals=["tris"]
)


pz.model.func_J = unsym.none

molalities = np.array([1.0, 1.0, 1.0, 1.0])
charges = np.array([+1, -1, +2, -2])
args = pz.split_molalities_charges(molalities, charges)
gibbs = pz.Gibbs_nRT(*args, **params)
acf = pz.activity_coefficients(*args, **params)
print(acf)

plib.set_func_J(pz)
acf = pz.activity_coefficients(*args, **params)
print(acf)

plib.set_func_J(pz)
acf = pz.activity_coefficients(*args, **params)
print(acf)
