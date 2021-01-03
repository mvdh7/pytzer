import jax
from jax import numpy as np

charges = {
    "Na": +1,
    "Cl": -1,
    "Mg": +2,
    "SO4": -2,
    "tris": 0,
}

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
    
    

solutes = {
    "Na": 1.5,
    "SO4": 0.75,
    "tris": 0.1,
}


istr = ionic_strength(solutes)
istr_g = jax.grad(ionic_strength)(solutes)

Gex = Gibbs(solutes)
Gex_g = jax.grad(Gibbs)(solutes)
