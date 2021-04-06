# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
"""Define solute properties."""
from . import convert
from autograd.numpy import concatenate, float_, unique, vstack

_ion2name = {
    "Ag": "silver",
    "Aljjj": "aluminium(III)",
    "AsO4": "arsenate",
    "BF4": "tetrafluoroborate",
    "BO2": "oxido(oxo)borane",
    "Ba": "barium",
    "Br": "bromide",
    "BrO3": "bromate",
    "Bu4N": "tetrabutylammonium",
    "CO3": "carbonate",
    "Ca": "calcium",
    "Cdjj": "cadmium(II)",
    "Ce": "cerium",
    "Cl": "chloride",
    "ClO3": "chlorate",
    "ClO4": "perchlorate",
    "CoCN6": "Co(CN)6",
    "Co(CN)6": "Co(CN)6",
    "Coen3": "tris(ethylenediamine)cobalt(III)",
    "Cojj": "cobalt(II)",
    "Copn3": "Copn3",
    "Cr": "chromium",
    "CrO4": "chromate",
    "Cs": "caesium",
    "Cujj": "copper(II)",
    "Et4N": "tetraethylammonium",
    "Eu": "europium",
    "F": "fluoride",
    "Fejj": "iron(II)",
    "FejjCN6": "ferrocyanide",
    "Fejj(CN)6": "ferrocyanide",
    "FejjjCN6": "ferricyanide",
    "Fejjj(CN)6": "ferricyanide",
    "Ga": "gallium",
    "H": "hydrogen",
    "H2AsO4": "dihydrogen-arsenate",
    "H2PO4": "dihydrogen-phosphate",
    "HAsO4": "hydrogen-arsenate",
    "HCO3": "bicarbonate",
    "HPO4": "hydrogen-phosphate",
    "HSO4": "bisulfate",
    "I": "iodide",
    "IO3": "iodate",
    "In": "indium",
    "K": "potassium",
    "La": "lanthanum",
    "Li": "lithium",
    "Me2H2N": "Me2H2N",
    "Me3HN": "Me3HN",
    "MeH3N": "MeH3N",
    "MeN": "MeN",
    "Me4N": "tetramethylammonium",
    "Mg": "magnesium",
    "MgOH": "magnesium-hydroxide",
    "Mnjj": "manganese(II)",
    "MoCN8": "Mo(CN)8",
    "Mo(CN)8": "Mo(CN)8",
    "NH4": "ammonium",
    "NO2": "nitrite",
    "NO3": "nitrate",
    "Na": "sodium",
    "Nd": "neodymium",
    "Nijj": "nickel(II)",
    "OAc": "OAc",
    "OH": "hydroxide",
    "P2O7": "diphosphate",
    "P3O10": "triphosphate-pentaanion",
    "P3O9": "trimetaphosphate",
    "PO4": "phosphate",
    "Pbjj": "lead(II)",
    "Pr": "praeseodymium",
    "Pr4N": "tetrapropylammonium",
    "PtCN4": "platinocyanide",
    "Pt(CN)4": "platinocyanide",
    "PtF6": "platinum-hexafluoride",
    "Rb": "rubidium",
    "S2O3": "thiosulfate",
    "SCN": "thiocyanate",
    "SO4": "sulfate",
    "Sm": "samarium",
    "Sr": "strontium",
    "Srjjj": "strontium(III)",
    "Th": "thorium",
    "Tl": "thallium",
    "UO2": "uranium-dioxide",
    "WCN8": "W(CN)8",
    "W(CN)8": "W(CN)8",
    "Y": "yttrium",
    "Znjj": "zinc(II)",
}

# Define general electrolyte to ions conversion dict
_ele2ions = {
    "Ba(NO3)2": (("Ba", "NO3"), (1, 2)),
    "CaCl2": (("Ca", "Cl"), (1, 2)),
    "Cd(NO3)2": (("Cdjj", "NO3"), (1, 2)),
    "Co(NO3)2": (("Cojj", "NO3"), (1, 2)),
    "CsCl": (("Cs", "Cl"), (1, 1)),
    "CuCl2": (("Cujj", "Cl"), (1, 2)),
    "Cu(NO3)2": (("Cujj", "NO3"), (1, 2)),
    "CuSO4": (("Cujj", "SO4"), (1, 1)),
    "glycerol": (("glycerol",), (1,)),
    "H2SO4": (("HSO4", "SO4", "H", "OH"), (0.5, 0.5, 1.5, 0.0)),
    "HBr": (("H", "Br"), (1, 1)),
    "HCl": (("H", "Cl"), (1, 1)),
    "HClO4": (("H", "ClO4"), (1, 1)),
    "HI": (("H", "I"), (1, 1)),
    "HNO3": (("H", "NO3"), (1, 1)),
    "KB(OH)4": (("K", "BOH4"), (1, 1)),
    "K2CO3": (("K", "CO3"), (2, 1)),
    "K2SO4": (("K", "SO4"), (2, 1)),
    "KBr": (("K", "Br"), (1, 1)),
    "KCl": (("K", "Cl"), (1, 1)),
    "KF": (("K", "F"), (1, 1)),
    "KI": (("K", "I"), (1, 1)),
    "KNO3": (("K", "NO3"), (1, 1)),
    "KOH": (("K", "OH"), (1, 1)),
    "LaCl3": (("La", "Cl"), (1, 3)),
    "Li2SO4": (("Li", "SO4"), (2, 1)),
    "LiBr": (("Li", "Br"), (1, 1)),
    "LiCl": (("Li", "Cl"), (1, 1)),
    "LiClO4": (("Li", "ClO4"), (1, 1)),
    "LiI": (("Li", "I"), (1, 1)),
    "LiNO3": (("Li", "NO3"), (1, 1)),
    "LiOH": (("Li", "OH"), (1, 1)),
    "MgCl2": (("Mg", "Cl"), (1, 2)),
    "Mg(ClO4)2": (("Mg", "ClO4"), (1, 2)),
    "Mg(NO3)2": (("Mg", "NO3"), (1, 2)),
    "MgSO4": (("Mg", "SO4"), (1, 1)),
    "Na2S2O3": (("Na", "S2O3"), (2, 1)),
    "Na2SO4": (("Na", "SO4"), (2, 1)),
    "NaB(OH)4": (("Na", "BOH4"), (1, 1)),
    "NaBr": (("Na", "Br"), (1, 1)),
    "NaCl": (("Na", "Cl"), (1, 1)),
    "NaClO4": (("Na", "ClO4"), (1, 1)),
    "NaF": (("Na", "F"), (1, 1)),
    "NaI": (("Na", "I"), (1, 1)),
    "NaOH": (("Na", "OH"), (1, 1)),
    "NaNO3": (("Na", "NO3"), (1, 1)),
    "RbCl": (("Rb", "Cl"), (1, 1)),
    "SrCl2": (("Sr", "Cl"), (1, 2)),
    "Sr(NO3)2": (("Sr", "NO3"), (1, 2)),
    "sucrose": (("sucrose",), (1,)),
    "tris": (("tris",), (1,)),
    "(trisH)2SO4": (("trisH", "SO4"), (2, 1)),
    "trisHCl": (("trisH", "Cl"), (1, 1)),
    "UO2(NO3)2": (("UO2", "NO3"), (1, 2)),
    "urea": (("urea",), (1,)),
    "Zn(ClO4)2": (("Znjj", "ClO4"), (1, 2)),
    "Zn(NO3)2": (("Znjj", "NO3"), (1, 2)),
}

# Define electrolyte to ions conversion dict for equilibria
_eq2ions = {
    "t_H2CO3": ("CO2", "HCO3", "CO3"),
    "t_HSO4": ("HSO4", "SO4"),
    "t_Mg": ("Mg", "MgOH"),
    "t_trisH": ("trisH", "tris"),
    "t_BOH3": ("BOH3", "BOH4"),
}

# Relative ionic masses in g/mol
_ion2mass = {
    "H": 1.00794,
    "Li": 6.941,
    "Na": 22.98977,
    "K": 39.0983,
    "Rb": 85.4678,
    "Cs": 132.90545,
    "NH4": 18.03846,
    "Mg": 24.3050,
    "Ca": 40.078,
    "Sr": 87.62,
    "MgF": 43.3034,
    "CaF": 59.0764,
    "SrF": 106.6184,
    "Ba": 137.327,
    "TrisH": 122.14298,
    "MgOH": 41.31234,
    "Fejj": 55.845,
    "Fejjj": 55.845,
    "Cdjj": 112.411,
    "Ni": 58.6934,
    "Cuj": 63.546,
    "Cujj": 63.546,
    "Zn": 65.409,
    "F": 18.99840,
    "Cl": 35.453,
    "Br": 79.904,
    "I": 126.90447,
    "NO3": 62.00490,
    "OH": 17.00734,
    "HSO4": 97.07054,
    "SO4": 96.06260,
    "HCO3": 61.01684,
    "CO3": 60.00890,
    "BOH4": 78.84036,
    "NH3": 17.03052,
    "CO2": 44.00950,
    "BOH3": 61.83302,
    "Tris": 121.13504,
    "HF": 20.00634,
    "MgCO3": 83.3139,
    "CaCO3": 100.0869,
    "SrCO3": 147.6289,
}

# Select which ionic mass to use for molinity to molality conversion of eles
_ele2ionmass = {
    "t_HSO4": "SO4",
    "t_trisH": "tris",
    "t_Mg": "Mg",
    "t_BOH3": "BOH3",
    "t_H2CO3": "HCO3",
}


def charges(ions):
    """Find the charges on each of a list of ions."""
    if len(ions) == 0:
        zs = float_([])
        cations = []
        anions = []
        neutrals = []
    else:
        zs = vstack([float_(convert.solute_to_charge[ion]) for ion in ions])
        cations = [ion for ion in ions if convert.solute_to_charge[ion] > 0]
        anions = [ion for ion in ions if convert.solute_to_charge[ion] < 0]
        neutrals = [ion for ion in ions if convert.solute_to_charge[ion] == 0]
    return zs, cations, anions, neutrals


def getallions(eles, fixions):
    """Get all ions given list of electrolytes for equilibria."""
    if len(eles) == 0:
        allions = concatenate([fixions, ["H", "OH"]])
    else:
        allions = concatenate(
            [
                fixions,
                concatenate([_eq2ions[ele] for ele in eles]),
                ["H", "OH"],
            ]
        )
    if len(unique(allions)) < len(allions):
        allions = list(allions)
        allions.reverse()
        seen = set()
        seen_add = seen.add
        allions = [ion for ion in allions if not (ion in seen or seen_add(ion))]
        allions.reverse()
    return allions
