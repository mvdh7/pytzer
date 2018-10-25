from autograd import numpy as np
from autograd import elementwise_grad as egrad
from scipy import optimize
from . import data, fitting, model

##### ISOPIESTIC EQUILIBRIUM ##################################################

# Reference osmotic coefficient known
def osm(mols,molsR,osmR):
    
    return np.vstack(np.sum(molsR,axis=1) * osmR.ravel() / np.sum(mols,axis=1))

# Get test molality from simulated osmotic coefficients
def osm2tot(osm,nu,molsR,osmR):
    
    return np.vstack(np.sum(molsR,axis=1)) * osmR / (osm * nu)

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

# Simulate a perfect isopiestic dataset - FAULTY!!! SEE ALTERNATIVE IN SIMLOOP
def get_isotot(bCT,nuT,zCT,zAT,nCT,nAT,T,molsR,osmR):
    # bC & nu are for the 'test' electrolyte to be simulated
    # molsR and osmR are for the reference electrolyte
    
    def osm_tot(tot,zC,zA,nC,nA,T,bC,M):
        mols = np.array([[tot*nC,tot*nA],[tot*nC,tot*nA]])
        # ^cheap fix because pz.fitting.osm can't handle single-row mols
        return fitting.osm(mols,zC,zA,T,*[bC[X][M] for X in range(5)],
                           *[bC[X] for X in range(5,8)])

    totT = np.full_like(osmR,np.nan)

    for M in range(len(osmR)):
        totT[M] = optimize.least_squares(lambda totT:
            (osm_tot(totT,zCT[0],zAT[0],nCT[0],nAT[0],
                     np.vstack(T[M]),bCT,M)[0] * nuT * totT \
            - osmR[M] * np.sum(molsR[M])).ravel(), 1.)['x']
            
    osmT = fitting.osm(np.concatenate((totT*nCT,totT*nAT),axis=1),
                       zCT,zAT,T,*bCT)
            
    return totT, osmT

# Define functions to fit isopiestic equilibrium residuals
def isofit_sys(isoerr,tot):
    return isoerr * (1 + 1/(tot + 0.03))

def isofit_rdm(isoerr,tot):
    return isoerr[0] + isoerr[1] / (tot + 0.03)

# Define functions to fit vapour pressure lowering residuals
def vplfit_sys(vplerr,tot):
    return vplerr / tot

def vplfit_rdm(vplerr,tot):
    return vplerr[0] + vplerr[1] * np.exp(-tot)
