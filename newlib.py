import jax
import numpy as np, pytzer as pz
from pytzer.libraries import Seawater as plib
from pytzer import unsymmetrical as unsym


pz.model.func_J = unsym.none

solutes = {
    "Na": 1.0,
    "Cl": 1.0,
    "Ca": 0.5,
    "SO4": 1.0,
    "Mg": 0.5,
    "tris": 1.0,
}

# molalities, charges = pz.get_molalities_charges(solutes)
args, ss = pz.get_pytzer_args(solutes)
params = plib.get_parameters(**ss)

gibbs = pz.model.Gibbs_nRT(*args, **params)
acf = pz.model.activity_coefficients(*args, **params)
print(acf)

acf2 = pz.activity_coefficients(solutes, verbose=False)

func_acf = jax.jit(lambda args: pz.activity_coefficients(*args, **params))

# plib.set_func_J(pz)
# acf = pz.activity_coefficients(*args, **params)
# print(acf)

# plib.set_func_J(pz)
# acf = pz.activity_coefficients(*args, **params)
# print(acf)
