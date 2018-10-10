from autograd import numpy as np
from . import fitting, model

##### ISOPIESTIC EQUILIBRIUM ##################################################

# Reference osmotic coefficient known
def osm(mols,molsR,osmR):
    
    return np.sum(molsR,axis=1) * osmR / np.sum(mols,axis=1)

# Reference osmotic coeff calculated from pz.model
def osm_cf(mols,molsR,ionsR,T,cf):
    
    osmR = model.osm(molsR,ionsR,T,cf)
    
    return osm(mols,molsR,osmR)

# Reference osmotic coeff calculated from input bC coeffs
def osm_bC(mols,molsR,zCR,zAR,T,b0R,b1R,b2R,C0R,C1R,alph1R,alph2R,omegaR):
    
    osmR = fitting.osm(molsR,zCR,zAR,T,
                       b0R,b1R,b2R,C0R,C1R,alph1R,alph2R,omegaR)

    return osm(mols,molsR,osmR)
