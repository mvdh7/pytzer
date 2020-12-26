import jax, time
import pytzer as pz
from pytzer.libraries import Seawater as plib
from pytzer import unsymmetrical as unsym


pz.model.func_J = unsym.none

solutes = {
    "Na": 1.0,
    "Cl": 1.4,
    "Br": 0.3,
    "K": 0.3,
    "Ca": 0.5,
    "SO4": 1.0,
    "Mg": 0.5,
    "Sr": 0.2,
    "tris": 1.0,
    "CO2": 0.2,
}

# molalities, charges = pz.get_molalities_charges(solutes)
args, ss = pz.get_pytzer_args(solutes)
params = plib.get_parameters(**ss, verbose=False)

print("orig")
go = time.time()
gibbs = pz.model.Gibbs_nRT(*args, **params).item()
print(time.time() - go)
print("map")
go = time.time()
gibbs_map = pz.model.Gibbs_map(*args, **params).item()
print(time.time() - go)

print("orig")
go = time.time()
acfs = pz.model.log_activity_coefficients(*args, **params)
print(time.time() - go)
print("map")
go = time.time()
acfs_map = pz.model.log_activity_coefficients_map(*args, **params)
print(time.time() - go)


# acf = pz.model.activity_coefficients(*args, **params)
# print(acf)

# acf2 = pz.activity_coefficients(solutes, verbose=False)

# func_acf = jax.jit(lambda args: pz.activity_coefficients(*args, **params))

# plib.set_func_J(pz)
# acf = pz.activity_coefficients(*args, **params)
# print(acf)

# plib.set_func_J(pz)
# acf = pz.activity_coefficients(*args, **params)
# print(acf)
