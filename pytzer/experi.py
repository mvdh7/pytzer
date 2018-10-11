from autograd import numpy as np
from autograd import elementwise_grad as egrad
from scipy import optimize
from . import data, fitting, model

##### ISOPIESTIC EQUILIBRIUM ##################################################

# Reference osmotic coefficient known
def osm(mols,molsR,osmR):
    
    return np.vstack(np.sum(molsR,axis=1) * osmR.ravel() / np.sum(mols,axis=1))

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
dosm_dtot  = egrad(osm_bC)
dosm_dtotR = egrad(osm_bC, argnum=1)

# Simulate a perfect isopiestic dataset
def get_osm(bC,T,molsR,osmR):
    # bC is for the 'test' electrolyte to be simulated
    # molsR and osmR are for the reference electrolyte
    
    def osm_tot(tot,zC,zA,T,bC):
        mols = np.array([[tot,tot],[tot,tot]])
        # ^cheap fix because pz.fitting.osm can't handle single-row mols
        return fitting.osm(mols,zC,zA,T,*bC)

    tot = np.full_like(osmR,np.nan)

    for M in range(len(osmR)):
        tot[M] = optimize.least_squares(lambda tot:
            osm_tot(tot,1.,-1.,np.vstack(T[M]),bC)[0] * (tot + tot) \
                           - osmR[M] * np.sum(molsR[M]), 1.)['x']
            
    return tot

# Define functions to fit isopiestic residuals
def isofit_sys(isoerr,tot):
    return isoerr * (1 + 1/(tot + 0.03))

def isofit_rdm(isoerr,tot):
    return isoerr[0] + isoerr[1] / (tot + 0.03)
