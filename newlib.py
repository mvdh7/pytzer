import numpy as np, pytzer as pz
from pytzer.libraries import Moller88
from pytzer import unsymmetrical as unsym

params = Moller88.get_parameters(
    cations=["Na", "Ca"], anions=["Cl", "SO4"], neutrals=["tris"]
)


pz.model.func_J = unsym.none

molalities = np.array([1.0, 1.0, 1.0, 1.0])
charges = np.array([+1, -1, +2, -2])
args = pz.split_molalities_charges(molalities, charges)
gibbs = pz.Gibbs_nRT(*args, **params)
acf = pz.activity_coefficients(*args, **params)
print(acf)

# import importlib
# pz.model = importlib.reload(pz.model)
# pz = importlib.reload(pz)
# pz.model.func_J = unsym.Harvie

# pz.update_func_J(pz, unsym.Harvie)
Moller88.set_func_J(pz)
acf = pz.activity_coefficients(*args, **params)
print(acf)

Moller88.set_func_J(pz)
acf = pz.activity_coefficients(*args, **params)
print(acf)
