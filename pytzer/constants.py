# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
"""Define universal constants."""

# Set constant values
Mw = 0.018015  # Molar mass of water in kg/mol [PubChem]
R = 8.3144598  # Universal gas constant in J/(mol*K) [CODATA]
F = 96485.33289  # Faraday constant in C/mol [CODATA]
b = 1.2  # Pitzer model coeff. in sqrt(kg/mol) [Pitzer 1991]
Patm_bar = 1.01325  # Atmospheric pressure in bar/atm
Tzero = 273.15  # Zero degrees Celsius in K
NA = 6.0221367e23  # Avogadro's constant in 1/mol

# Unit conversion multipliers
cal2J = 4.184  # ENERGY: calorie to Joule
atm2Pa = 101325  # PRESSURE: atmosphere to Pascal
Torr2Pa = atm2Pa / 760  # PRESSURE: Torr to Pascal
mmHg2Pa = 133.322387415  # PRESSURE: mmHg to Pascal
dbar2Pa = 1e4  # PRESSURE: dbar to Pascal
dbar2MPa = 1e-2  # PRESSURE: dbar to mega-Pascal
