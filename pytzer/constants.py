# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
"""Define universal constants."""
from autograd.numpy import float_

# Set constant values
Mw = float_(0.018015) # Molar mass of water in kg/mol [PubChem]
R = float_(8.3144598) # Universal gas constant in J/(mol*K) [CODATA]
F = float_(96485.33289) # Faraday constant in C/mol [CODATA]
b = float_(1.2) # Pitzer model coeff. in sqrt(kg/mol) [Pitzer 1991]
Patm_bar = float_(1.01325) # Atmospheric pressure in bar/atm
Tzero = float_(273.15) # Zero degrees Celsius in K
NA = float_(6.0221367e+23) # Avogadro's constant in 1/mol

# Unit conversion multipliers
cal2J   = float_(4.184) # ENERGY: calorie to Joule
atm2Pa  = float_(101325) # PRESSURE: atmosphere to Pascal
Torr2Pa = atm2Pa / 760 # PRESSURE: Torr to Pascal
mmHg2Pa = float_(133.322387415) # PRESSURE: mmHg to Pascal
dbar2Pa = 1e4 # PRESSURE: dbar to Pascal
dbar2MPa = 1e-2 # PRESSURE: dbar to mega-Pascal
