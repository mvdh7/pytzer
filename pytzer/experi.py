from autograd import numpy as np
from autograd import elementwise_grad as egrad
from . import data, fitting, model

##### ISOPIESTIC EQUILIBRIUM ##################################################

# Reference osmotic coefficient known
def osm(mols,molsR,osmR):
    
    return np.sum(molsR,axis=1) * osmR / np.sum(mols,axis=1)

# Reference osmotic coeff calculated from pz.model
def osm_cf(mols,molsR,ionsR,T,cf):
    
    osmR = model.osm(molsR,ionsR,T,cf)
    
    return osm(mols,molsR,osmR)

# Reference osmotic coeff calculated from input bC coeffs
def osm_bC(tot,totR,isopair,T,bCR):
    
    _,zC ,zA ,nC ,nA  = data.znu([isopair[0]])
    _,zCR,zAR,nCR,nAR = data.znu([isopair[1]])
    
    mols  = np.concatenate((nC *tot ,nA *tot ),axis=1)
    molsR = np.concatenate((nCR*totR,nAR*totR),axis=1)
    
    osmR = fitting.osm(molsR,zCR,zAR,T,*bCR)

    return osm(mols,molsR,osmR)

# Derivatives
dosm_dtot = egrad(osm_bC)
