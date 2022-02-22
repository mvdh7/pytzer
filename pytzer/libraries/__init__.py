# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
"""Assemble parameter libraries."""
from .ParameterLibrary import ParameterLibrary
from .Clegg94 import Clegg94
from .Greenberg89 import Greenberg89
from .Harvie84 import Harvie84
from .HeMorse93 import HeMorse93
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

# Aliases for convenience
CRP94 = Clegg94
GM89 = Greenberg89
HMW84 = Harvie84
HM93 = HeMorse93
myMCS = MyMarChemSpecCO2
MCS = MarChemSpec
MCS25 = MarChemSpec25
MP98 = MIAMI = Millero98
M88 = Moller88
WM13 = Waters13
WM13_C22 = Waters13_Clegg22
WM13_H22 = Waters13_Humphreys22
WM13_MCS25 = Waters13_MarChemSpec25

# solutes_MarChemSpec = np.array(
#     ["H", "Na", "Mg", "Ca", "K", "MgOH", "trisH", "Cl", "SO4", "HSO4", "OH", "tris"]
# )
