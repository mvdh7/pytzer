# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2020  Matthew Paul Humphreys  (GNU GPLv3)
import numpy as np
from . import convert


def expand_solute_molalities(solute_molalities):
    """Get solutes, molalities and charges from the solute dict."""
    solutes = []
    molalities = []
    charges = []
    for solute, molality in solute_molalities.items():
        solutes.append(solute)
        molalities.append(molality)
        charges.append(convert.solute_to_charge[solute])
    return solutes, np.array(molalities), np.array(charges)


def split_solute_types(values, charges):
    """Split up input values into cations, anions and neutrals."""
    cations = np.compress(charges > 0, values)
    anions = np.compress(charges < 0, values)
    neutrals = np.compress(charges == 0, values)
    return cations, anions, neutrals


def get_pytzer_args(solute_molalities):
    solutes, molalities, charges = expand_solute_molalities(solute_molalities)
    m_cats, m_anis, m_neus = split_solute_types(molalities, charges)
    z_cats, z_anis = split_solute_types(charges, charges)[:2]
    pytzer_args = (m_cats, m_anis, m_neus, z_cats, z_anis)
    solutes_split = {}
    (
        solutes_split["cations"],
        solutes_split["anions"],
        solutes_split["neutrals"],
    ) = split_solute_types(solutes, charges)
    return pytzer_args, solutes_split
