import pytzer as pz
import jax
from jax import numpy as np

x = np.sqrt(2)

j__P75_eq46 = pz.unsymmetrical.P75_eq46(x)
jg__P75_eq46 = jax.grad(pz.unsymmetrical.P75_eq46)(x)

j__P75_eq47 = pz.unsymmetrical.P75_eq47(x)
jg__P75_eq47 = jax.grad(pz.unsymmetrical.P75_eq47)(x)


j__Harvie = pz.unsymmetrical.Harvie(x)
