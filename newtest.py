import pytzer as pz
import jax
from jax import numpy as np

x = np.sqrt(2)

j__P75_eq46 = pz.unsymmetrical.P75_eq46(x).item()
jg__P75_eq46 = jax.grad(pz.unsymmetrical.P75_eq46)(x).item()

j__P75_eq47 = pz.unsymmetrical.P75_eq47(x).item()
jg__P75_eq47 = jax.grad(pz.unsymmetrical.P75_eq47)(x).item()

j__Harvie, jg__Harvie = pz.unsymmetrical.Harvie(x)
j__Harvie = j__Harvie.item()
jg__Harvie = jg__Harvie.item()
