from collections import OrderedDict
import jax
from jax import numpy as np
import pytzer as pz

charges = OrderedDict({"Na": +1, "Cl": -1, "Mg": +2, "SO4": -2, "tris": 0,})

all_cations = set([s for s, c in charges.items() if c > 0])
all_anions = set([s for s, c in charges.items() if c < 0])
all_neutrals = set([s for s, c in charges.items() if c == 0])


@jax.jit
def ionic_strength(solutes):
    return np.sum(np.array([m * charges[s] ** 2 for s, m in solutes.items()])) / 2


@jax.jit
def Gibbs(solutes):
    Gex = ionic_strength(solutes)
    cations = [m for m in solutes.keys() if m in all_cations]
    anions = [m for m in solutes.keys() if m in all_anions]
    neutrals = [m for m in solutes.keys() if m in all_neutrals]
    for c in cations:
        for a in anions:
            Gex = Gex + solutes[c] * solutes[a]
    for n in neutrals:
        Gex = Gex + solutes[n]
    return Gex


solutes = OrderedDict({"Na": 1.5, "SO4": 0.75, "tris": 0.1,})

m_cats = np.array([1.5])
m_anis = np.array([0.75])
m_neus = np.array([0.1])
z_cats = np.array([+1])
z_anis = np.array([-2])
args = (m_cats, m_anis, m_neus, z_cats, z_anis)

istr = ionic_strength(solutes)
istr_g = jax.grad(ionic_strength)(solutes)

Gex = Gibbs(solutes)
Gex_g = jax.grad(Gibbs)(solutes)


params = pz.libraries.Seawater.get_parameters(solutes=solutes)

Gex_v0 = pz.model.Gibbs_nRT(*args, **params)
Gex_v1 = pz.model.Gibbs_map(*args, **params)
Gex_v2 = pz.model.Gibbs_dict_map(solutes, **params)
acfs_v1 = pz.model.log_activity_coefficients_map(*args, **params)
acfs_v2 = jax.grad(pz.model.Gibbs_dict_map)(solutes, **params)
