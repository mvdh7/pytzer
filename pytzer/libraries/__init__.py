# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2024  M.P. Humphreys  (GNU GPLv3)
"""Assemble parameter libraries."""
from .library_class import Library
from .ParameterLibrary import ParameterLibrary
from .Clegg22 import Clegg22
from .Clegg23 import Clegg23
from .Clegg94 import Clegg94
from .Greenberg89 import Greenberg89
from .Harvie84 import Harvie84
from .HeMorse93 import HeMorse93
from .Humphreys22 import Humphreys22
from .MarChemSpec import MarChemSpec
from .MarChemSpec25 import MarChemSpec25
from .Millero98 import Millero98
from .Moller88 import Moller88
from .MyMarChemSpecCO2 import MyMarChemSpecCO2
from .Seawater import Seawater
from .Waters13 import Waters13
from .Waters13_Humphreys22 import Waters13_Humphreys22
from .Waters13_Clegg22 import Waters13_Clegg22
from .Waters13_MarChemSpec25 import Waters13_MarChemSpec25
from . import lib_CWTD23

# Aliases for convenience
CHW22 = Clegg22
CWTD23 = Clegg23
CRP94 = Clegg94
GM89 = Greenberg89
HMW84 = Harvie84
HM93 = HeMorse93
HWT22 = Humphreys22
myMCS = MyMarChemSpecCO2
MCS = MarChemSpec
MCS25 = MarChemSpec25
MP98 = MIAMI = Millero98
M88 = Moller88
WM13 = Waters13
WM13_C22 = Waters13_Clegg22
WM13_H22 = Waters13_Humphreys22
WM13_MCS25 = Waters13_MarChemSpec25

# And a dict for even more convenience
libraries = {
    k.lower(): v
    for k, v in {
        "Clegg22": Clegg22,
        "CHW22": Clegg22,
        "Clegg23": Clegg23,
        "CWTD23": Clegg23,
        "Clegg94": Clegg94,
        "CRP94": Clegg94,
        "Greenberg89": Greenberg89,
        "GM89": Greenberg89,
        "Harvie84": Harvie84,
        "HMW84": Harvie84,
        "HeMorse93": HeMorse93,
        "HM93": HeMorse93,
        "Humphreys22": Humphreys22,
        "HWT22": Humphreys22,
        "MyMarChemSpecCO2": MyMarChemSpecCO2,
        "myMCS": MyMarChemSpecCO2,
        "MarChemSpec": MarChemSpec,
        "MCS": MarChemSpec,
        "MarChemSpec25": MarChemSpec25,
        "MCS25": MarChemSpec25,
        "Millero98": Millero98,
        "MP98": Millero98,
        "Moller88": Moller88,
        "M88": Moller88,
        "Seawater": Seawater,
        "Waters13": Waters13,
        "WM13": Waters13,
        "Waters13_Clegg22": Waters13_Clegg22,
        "WM13_C22": Waters13_Clegg22,
        "Waters13_Humphreys22": Waters13_Humphreys22,
        "WM13_H22": Waters13_Humphreys22,
        "Waters13_MarChemSpec25": Waters13_MarChemSpec25,
        "WM13_MCS25": Waters13_MarChemSpec25,
    }.items()
}
