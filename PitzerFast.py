import pytzer as pz
import numpy as np
from copy import deepcopy

mols = np.array([[0.1,0.1]])
ions = np.array(['H','Cl'])
T    = np.array([[278.15]])
cf   = deepcopy(pz.cfdicts.MarChemSpec)

# tweak b0 for H-Cl
def bC_H_Cl_tweak(T):
    
    b0,b1,b2,C0,C1, alph1,alph2,omega, valid = pz.coeffs.bC_H_Cl_CMR93(T)
    
    b0 = b0 * 1.08297976513196
    
    return b0,b1,b2,C0,C1, alph1,alph2,omega, valid
    
cf.bC['H-Cl'] = bC_H_Cl_tweak

osm  = pz.model.osm (mols,ions,T,cf)
acfs = pz.model.acfs(mols,ions,T,cf)
