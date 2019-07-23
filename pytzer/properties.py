# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
"""Define ionic properties."""
from autograd.numpy import array, concatenate, float_, unique, vstack

# Define dict of charges.
# Order: neutrals, cations, then anions, and alphabetical within each group.
_ion2charge = {
    # Neutrals
    'glycerol': 0,
    'sucrose': 0,
    'tris': 0,
    'urea': 0,
    # Cations
    'Ba': +2,
    'Ca': +2,
    'Cdjj': +2,
    'Cojj': +2,
    'Cs': +1,
    'Cujj': +2,
    'H': +1,
    'K': +1,
    'La': +3,
    'Li': +1,
    'Mg': +2,
    'MgOH': +1,
    'Na': +1,
    'NH4': +1,
    'Rb': +1,
    'Sr': +2,
    'trisH': +1,
    'UO2': +2,
    'Znjj': +2,
    # Anions
    'acetate': -1,
    'BOH4': -1,
    'Br': -1,
    'Cl': -1,
    'ClO4': -1,
    'CO3': -2,
    'F': -1,
    'HSO4': -1,
    'I': -1,
    'NO3': -1,
    'OH': -1,
    'S2O3': -2,
    'SO4': -2,
}

_ion2name = {
    'Ag': 'silver',
    'Aljjj': 'aluminium(III)',
    'AsO4': 'arsenate',
    'BF4': 'tetrafluoroborate',
    'BO2': 'oxido(oxo)borane',
    'Ba': 'barium',
    'Br': 'bromide',
    'BrO3': 'bromate',
    'Bu4N': 'tetrabutylammonium',
    'CO3': 'carbonate',
    'Ca': 'calcium',
    'Cdjj': 'cadmium(II)',
    'Ce': 'cerium',
    'Cl': 'chloride',
    'ClO3': 'chlorate',
    'ClO4': 'perchlorate',
    'CoCN6': 'Co(CN)6',
    'Coen3': 'tris(ethylenediamine)cobalt(III)',
    'Cojj': 'cobalt(II)',
    'Copn3': 'Copn3',
    'Cr': 'chromium',
    'CrO4': 'chromate',
    'Cs': 'caesium',
    'Cujj': 'copper(II)',
    'Et4N': 'tetraethylammonium',
    'Eu': 'europium',
    'F': 'fluoride',
    'Fejj': 'iron(II)',
    'FejjCN6': 'ferrocyanide',
    'FejjjCN6': 'ferricyanide',
    'Ga': 'gallium',
    'H': 'hydrogen',
    'H2AsO4': 'dihydrogen-arsenate',
    'H2PO4': 'dihydrogen-phosphate',
    'HAsO4': 'hydrogen-arsenate',
    'HCO3': 'bicarbonate',
    'HPO4': 'hydrogen-phosphate',
    'HSO4': 'bisulfate',
    'I': 'iodide',
    'IO3': 'iodate',
    'In': 'indium',
    'K': 'potassium',
    'La': 'lanthanum',
    'Li': 'lithium',
    'Me2H2N': 'Me2H2N',
    'Me3HN': 'Me3HN',
    'MeH3N': 'MeH3N',
    'MeN': 'MeN',
    'Me4N': 'tetramethylammonium',
    'Mg': 'magnesium',
    'MgOH': 'magnesium-hydroxide',
    'Mnjj': 'manganese(II)',
    'MoCN8': 'Mo(CN)8',
    'NH4': 'ammonium',
    'NO2': 'nitrite',
    'NO3': 'nitrate',
    'Na': 'sodium',
    'Nd': 'neodymium',
    'Nijj': 'nickel(II)',
    'OAc': 'OAc',
    'OH': 'hydroxide',
    'P2O7': 'diphosphate',
    'P3O10': 'triphosphate-pentaanion',
    'P3O9': 'trimetaphosphate',
    'PO4': 'phosphate',
    'Pbjj': 'lead(II)',
    'Pr': 'praeseodymium',
    'Pr4N': 'tetrapropylammonium',
    'PtCN4': 'platinocyanide',
    'PtF6': 'platinum-hexafluoride',
    'Rb': 'rubidium',
    'S2O3': 'thiosulfate',
    'SCN': 'thiocyanate',
    'SO4': 'sulfate',
    'Sm': 'samarium',
    'Sr': 'strontium',
    'Srjjj': 'strontium(III)',
    'Th': 'thorium',
    'Tl': 'thallium',
    'UO2': 'uranium-dioxide',
    'WCN8': 'W(CN)8',
    'Y': 'yttrium',
    'Znjj': 'zinc(II)',
}

# Define electrolyte to ions conversion dict
_ele2ions = {
    't_HSO4': array(['HSO4', 'SO4']),
    't_Mg': array(['Mg', 'MgOH']),
    't_trisH': array(['trisH', 'tris']),
}

def charges(ions):
    """Find the charges on each of a list of ions."""
    zs = vstack([float_(_ion2charge[ion]) for ion in ions])
    cations  = [ion for ion in ions if _ion2charge[ion] > 0]
    anions   = [ion for ion in ions if _ion2charge[ion] < 0]
    neutrals = [ion for ion in ions if _ion2charge[ion] == 0]
    return zs, cations, anions, neutrals

def getallions(eles, fixions):
    """Get all ions given list of electrolytes."""
    return unique(concatenate([
        fixions,
        concatenate([_ele2ions[ele] for ele in eles]),
        ['H', 'OH'],
    ]))
