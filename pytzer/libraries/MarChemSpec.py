# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
import copy
from .. import debyehueckel, dissociation as k
from .MarChemSpec25 import MarChemSpec25

MarChemSpec = copy.deepcopy(MarChemSpec25)
MarChemSpec.update_Aphi(debyehueckel.Aosm_MarChemSpec)
# Add equilibrium constants
MarChemSpec.update_equilibrium("H2O", k.H2O_MF)
MarChemSpec.update_equilibrium("HSO4", k.HSO4_CRP94)
MarChemSpec.update_equilibrium("MgOH", k.MgOH_CW91)
MarChemSpec.update_equilibrium("trisH", k.trisH_BH64)
