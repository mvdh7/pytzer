import autograd.numpy as np

# Set constant values
Mw = np.float_(0.018015)  # Molar mass of water  / kg/mol             [PubChem]
R  = np.float_(8.3144598) # Universal gas const. / J /mol /K           [CODATA]
F  = np.float_(96485.33289) # Faraday const.     / C /mol              [CODATA]
b  = np.float_(1.2)       # Pitzer model coeff.  / sqrt(kg/mol)   [Pitzer 1991]
Patm_bar = np.float_(1.01325) # Atmospheric pressure / bar

# Unit conversion factors
cal2j   = np.float_(4.184)         #  ENERGY  calorie    to Joule  / cal/J
atm2pa  = np.float_(101325)        # PRESSURE atmosphere to Pascal / atm/Pa
torr2pa = atm2pa / 760             # PRESSURE Torr       to Pascal / Torr/Pa
mmHg2pa = np.float_(133.322387415) # PRESSURE mmHg       to Pascal / mmHg/Pa
