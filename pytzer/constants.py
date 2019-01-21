from autograd.numpy import float_

# Set constant values
Mw = float_(0.018015)     # Molar mass of water  / kg/mol             [PubChem]
R  = float_(8.3144598)    # Universal gas const. / J /mol /K           [CODATA]
F  = float_(96485.33289)    # Faraday const.     / C /mol              [CODATA]
b  = float_(1.2)          # Pitzer model coeff.  / sqrt(kg/mol)   [Pitzer 1991]
Patm_bar = float_(1.01325) # Atmosphere pressure / bar

# Unit conversion factors
cal2J   = float_(4.184)         #  ENERGY  calorie    to Joule  / cal/J
atm2Pa  = float_(101325)        # PRESSURE atmosphere to Pascal / atm/Pa
Torr2Pa = atm2Pa / 760          # PRESSURE Torr       to Pascal / Torr/Pa
mmHg2Pa = float_(133.322387415) # PRESSURE mmHg       to Pascal / mmHg/Pa
