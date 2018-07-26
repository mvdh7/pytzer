import autograd.numpy as np

# Set constant values
Mw = np.float_(0.018015)  # Molar mass of water  / kg/mol             [PubChem]
R  = np.float_(8.3144598) # Universal gas const. / J /mol /K           [CODATA]
F  = np.float_(96485.33289) # Faraday const.     / C /mol              [CODATA]
b  = np.float_(1.2)       # Pitzer model coeff.  / sqrt(kg/mol)   [Pitzer 1991]

cal2j = np.float_(4.184)  # Conversion factor    / cal/J

Patm_bar = np.float_(1.01325) # Atmospheric pressure / bar
