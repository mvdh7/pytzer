# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2024  M.P. Humphreys  (GNU GPLv3)
"""Define universal constants."""

# Set constant values
mass_water = 0.018015  # Molar mass of water in kg/mol [PubChem]
R = 8.3144598  # Universal gas constant in J/(mol*K) [CODATA]
F = 96485.33289  # Faraday constant in C/mol [CODATA]
b_pitzer = 1.2  # Pitzer model coeff. in sqrt(kg/mol) [Pitzer 1991]
Patm_bar = 1.01325  # Atmospheric pressure in bar/atm
temperatureC_zero = 273.15  # Zero degrees Celsius in K
n_avogadro = 6.0221367e23  # Avogadro's constant in 1/mol

# Unit conversion multipliers
cal_to_J = 4.184  # ENERGY: calorie to Joule
atm_to_Pa = 101325  # PRESSURE: atmosphere to Pascal
torr_to_Pa = atm_to_Pa / 760  # PRESSURE: Torr to Pascal
mmHg_to_Pa = 133.322387415  # PRESSURE: mmHg to Pascal
dbar_to_Pa = 1e4  # PRESSURE: dbar to Pascal
dbar_to_MPa = 1e-2  # PRESSURE: dbar to mega-Pascal
