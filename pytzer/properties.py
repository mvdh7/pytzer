# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
"""Define solute properties."""
from autograd.numpy import concatenate, float_, unique, vstack

# Define dict of charges.
# Order: neutrals, cations, then anions, and alphabetical within each group.
_ion2charge = {
    # Neutrals
    'BOH3': 0,
    'CO2': 0,
    'H3PO4': 0,
    'HF': 0,
    'glycerol': 0,
    'NH3': 0,
    'SO2': 0,
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
    'Fejj': +2,
    'Fejjj': +3,
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
    'AsO4': -2,
    'BOH4': -1,
    'Br': -1,
    'BrO3': -1,
    'Cl': -1,
    'ClO3': -1,
    'ClO4': -1,
    'CO3': -2,
    'F': -1,
    'H2AsO4': -1,
    'H2PO4': -1,
    'HAsO4': -2,
    'HCO3': -1,
    'HPO4': -2,
    'HS': -1,
    'HSO3': -1,
    'HSO4': -1,
    'I': -1,
    'NO2': -1,
    'NO3': -1,
    'OH': -1,
    'PO4': -3,
    'S2O3': -2,
    'SCN': -1,
    'SO3': -2,
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
    'Co(CN)6': 'Co(CN)6',
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
    'Fejj(CN)6': 'ferrocyanide',
    'FejjjCN6': 'ferricyanide',
    'Fejjj(CN)6': 'ferricyanide',
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
    'Mo(CN)8': 'Mo(CN)8',
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
    'Pt(CN)4': 'platinocyanide',
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
    'W(CN)8': 'W(CN)8',
    'Y': 'yttrium',
    'Znjj': 'zinc(II)',
}

# Define general electrolyte to ions conversion dict
_ele2ions = {
    'Ba(NO3)2': (('Ba', 'NO3'), (1, 2)),
    'CaCl2': (('Ca', 'Cl'), (1, 2)),
    'Cd(NO3)2': (('Cdjj', 'NO3'), (1, 2)),
    'Co(NO3)2': (('Cojj', 'NO3'), (1, 2)),
    'CsCl': (('Cs', 'Cl'), (1, 1)),
    'CuCl2': (('Cujj', 'Cl'), (1, 2)),
    'Cu(NO3)2': (('Cujj', 'NO3'), (1, 2)),
    'CuSO4': (('Cujj', 'SO4'), (1, 1)),
    'glycerol': (('glycerol',), (1,)),
    'H2SO4': (('HSO4', 'SO4', 'H', 'OH'), (0.5, 0.5, 1.5, 0.0)),
    'K2CO3': (('K', 'CO3'), (2, 1)),
    'K2SO4': (('K', 'SO4'), (2, 1)),
    'KCl': (('K', 'Cl'), (1, 1)),
    'KF': (('K', 'F'), (1, 1)),
    'LaCl3': (('La', 'Cl'), (1, 3)),
    'Li2SO4': (('Li', 'SO4'), (2, 1)),
    'LiCl': (('Li', 'Cl'), (1, 1)),
    'LiI': (('Li', 'I'), (1, 1)),
    'MgCl2': (('Mg', 'Cl'), (1, 2)),
    'Mg(ClO4)2': (('Mg', 'ClO4'), (1, 2)),
    'Mg(NO3)2': (('Mg', 'NO3'), (1, 2)),
    'MgSO4': (('Mg', 'SO4'), (1, 1)),
    'Na2S2O3': (('Na', 'S2O3'), (2, 1)),
    'Na2SO4': (('Na', 'SO4'), (2, 1)),
    'NaCl': (('Na', 'Cl'), (1, 1)),
    'NaF': (('Na', 'F'), (1, 1)),
    'NaOH': (('Na', 'OH'), (1, 1)),
    'RbCl': (('Rb', 'Cl'), (1, 1)),
    'SrCl2': (('Sr', 'Cl'), (1, 2)),
    'Sr(NO3)2': (('Sr', 'NO3'), (1, 2)),
    'sucrose': (('sucrose',), (1,)),
    'tris': (('tris',), (1,)),
    '(trisH)2SO4': (('trisH', 'SO4'), (2, 1)),
    'trisHCl': (('trisH', 'Cl'), (1, 1)),
    'UO2(NO3)2': (('UO2', 'NO3'), (1, 2)),
    'urea': (('urea',), (1,)),
    'Zn(ClO4)2': (('Znjj', 'ClO4'), (1, 2)),
    'Zn(NO3)2': (('Znjj', 'NO3'), (1, 2)),
}

# Define electrolyte to ions conversion dict for equilibria
_eq2ions = {
    't_HSO4': ('HSO4', 'SO4'),
    't_Mg': ('Mg', 'MgOH'),
    't_trisH': ('trisH', 'tris'),
}

def charges(ions):
    """Find the charges on each of a list of ions."""
    if len(ions) == 0:
        zs = float_([])
        cations = []
        anions = []
        neutrals = []
    else:
        zs = vstack([float_(_ion2charge[ion]) for ion in ions])
        cations = [ion for ion in ions if _ion2charge[ion] > 0]
        anions = [ion for ion in ions if _ion2charge[ion] < 0]
        neutrals = [ion for ion in ions if _ion2charge[ion] == 0]
    return zs, cations, anions, neutrals

def getallions(eles, fixions):
    """Get all ions given list of electrolytes for equilibria."""
    allions = concatenate([
        fixions,
        concatenate([_eq2ions[ele] for ele in eles]),
        ['H', 'OH'],
    ])
    if len(unique(allions)) < len(allions):
        allions = list(allions)
        allions.reverse()
        seen = set()
        seen_add = seen.add
        allions = [ion for ion in allions
            if not (ion in seen or seen_add(ion))]
        allions.reverse()
    return allions
