import pytzer as pz
import jax
from jax import numpy as np

x = np.sqrt(0.5)

j__P75_eq46 = pz.unsymmetrical.P75_eq46(x).item()
jg__P75_eq46 = jax.jit(jax.grad(pz.unsymmetrical.P75_eq46))(x).item()

j__P75_eq47 = pz.unsymmetrical.P75_eq47(x).item()
jg__P75_eq47 = jax.jit(jax.grad(pz.unsymmetrical.P75_eq47))(x).item()

j__Harvie_raw, jg__Harvie_raw = pz.unsymmetrical._Harvie_raw(x)
j__Harvie_raw = j__Harvie_raw.item()
jg__Harvie_raw = jg__Harvie_raw.item()

j__Harvie = pz.unsymmetrical.Harvie(x).item()
jg__Harvie = jax.grad(pz.unsymmetrical.Harvie)(x).item()
# ^ Grad works but Jit doesn't
