# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
"""Define ionic properties."""
from autograd.numpy import array, concatenate, float_, unique, vstack

def charges(ions):
    """Find the charges on each of a list of ions."""
    # Define dict of charges
    #   Order: neutrals, then cations, then anions,
    #          and alphabetical within each section.
    z = {
        # Neutrals
        'glycerol': 0,
        'sucrose' : 0,
        'tris'    : 0,
        'urea'    : 0,
        # Cations
        'Ba'   : +2,
        'Ca'   : +2,
        'Cdjj' : +2,
        'Cojj' : +2,
        'Cs'   : +1,
        'Cujj' : +2,
        'H'    : +1,
        'K'    : +1,
        'La'   : +3,
        'Li'   : +1,
        'Mg'   : +2,
        'MgOH' : +1,
        'Na'   : +1,
        'NH4'  : +1,
        'Rb'   : +1,
        'Sr'   : +2,
        'trisH': +1,
        'UO2'  : +2,
        'Znjj' : +2,
        # Anions
        'BOH4' : -1,
        'Br'   : -1,
        'Cl'   : -1,
        'ClO4' : -1,
        'CO3'  : -2,
        'F'    : -1,
        'HSO4' : -1,
        'I'    : -1,
        'NO3'  : -1,
        'OH'   : -1,
        'S2O3' : -2,
        'SO4'  : -2,
    }
    # Get charges and lists of cation, anion and neutral solute names
    zs = vstack([float_(z[ion]) for ion in ions])
    cations  = [ion for ion in ions if z[ion] > 0]
    anions   = [ion for ion in ions if z[ion] < 0]
    neutrals = [ion for ion in ions if z[ion] == 0]
    return zs, cations, anions, neutrals

# Define electrolyte to ions conversion dict
_ele2ions = {
    't_HSO4': array(['HSO4', 'SO4']),
    't_Mg': array(['Mg', 'MgOH']),
    't_trisH': array(['trisH', 'tris']),
}

def getallions(eles, fixions):
    """Get all ions given list of electrolytes."""
    return unique(concatenate([
        fixions,
        concatenate([_ele2ions[ele] for ele in eles]),
        ['H', 'OH'],
    ]))
