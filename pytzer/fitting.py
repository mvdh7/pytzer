from autograd import numpy as np
from autograd import elementwise_grad as egrad
from scipy import optimize
from .constants import b, R
from . import coeffs, jfuncs, model

##### PITZER MODEL FUNCTIONS ##################################################

# Define f function
def fG(T,I,Aosm):
    
    return -4 * Aosm * I * np.log(1 + b*np.sqrt(I)) / b

# Excess Gibbs energy - single electrolyte
def Gex_MX(mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega):
    
    mC = mCmA[:,0]
    mA = mCmA[:,1]
    
    I = (mC*zC**2 + mA*zA**2) / 2
    Z = mC*np.abs(zC) + mA*np.abs(zA)

    Aosm = coeffs.Aosm_CRP94(T)[0]
    
    B  = b0 + b1 * model.g(alph1*np.sqrt(I)) + b2 * model.g(alph2*np.sqrt(I))
    CT = C0 + C1 * model.h(omega*np.sqrt(I)) * 4
    
    Gex_nRT = fG(T,I,Aosm) + mC*mA * (2*B + Z*CT)
    
    return Gex_nRT

# LN of activity coefficients - single electrolyte
ln_acf = egrad(Gex_MX)

# Activity coefficients - single electrolyte
def acf(mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega):
    
    return np.exp(ln_acf(mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega))

# LN of mean activity coefficient - single electrolyte
def ln_acfMX(mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega,nC,nA):
    
    ln_acfs = ln_acf(mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega)
    
    return (nC * ln_acfs[:,0] + nA * ln_acfs[:,1]) / (nC + nA)

# Mean activity coefficient - single electrolyte
def acfMX(mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega,nC,nA):
    
    return np.exp(
        ln_acfMX(mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega,nC,nA))

# Osmotic coefficient derivative function - single electrolyte
def osmfunc(ww,mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega):
    
    return ww * R * T \
        * Gex_MX(np.array([mCmA[:,0]/ww,mCmA[:,1]/ww]).transpose(),
                 zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega)

# Osmotic coefficient derivative - single electrolyte
osmD = egrad(osmfunc)

# Osmotic coefficient - single electrolyte
def osm(mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega,nC=None,nA=None):
    
    ww = np.full_like(T,1, dtype='float64')
    
    return 1 - osmD(ww,mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega) \
        / (R * T * (np.sum(mCmA,axis=1)))

##### THREE-COMPONENT SYSTEM ##################################################
        
# Unsymmetric mixing functions
def xij(T,I,z0,z1):
    
    return 6 * z0*z1 * coeffs.Aosm_CRP94(T)[0] * np.sqrt(I)

def etheta(T,I,z0,z1):
    
    x00 = xij(T,I,z0,z0)
    x01 = xij(T,I,z0,z1)
    x11 = xij(T,I,z1,z1)
    
    jfunc = jfuncs.P75_eq47
    
    etheta = z0*z1 * (jfunc(x01)[0] \
                      - 0.5 * (jfunc(x00)[0] + jfunc(x11)[0])) / (4 * I)
    
    return etheta

# Excess Gibbs energy - mixed electrolyte (e.g. H2SO4)
def Gex_MXY(mols,zM,zX,zY,T,
            b0_MX,b1_MX,b2_MX,C0_MX,C1_MX,alph1_MX,alph2_MX,omega_MX,
            b0_MY,b1_MY,b2_MY,C0_MY,C1_MY,alph1_MY,alph2_MY,omega_MY):
    
    mM = mols[:,0]
    mX = mols[:,1]
    mY = mols[:,2]
    
    I = (mM*zM**2 + mX*zX**2 + mY*zY**2) / 2
    Z = mM*np.abs(zM) + mX*np.abs(zX) + mY*np.abs(zY)

    Aosm = coeffs.Aosm_CRP94(T)[0]
    
    B_MX  = b0_MX + b1_MX * model.g(alph1_MX*np.sqrt(I)) \
                  + b2_MX * model.g(alph2_MX*np.sqrt(I))
    CT_MX = C0_MX + C1_MX * model.h(omega_MX*np.sqrt(I)) * 4
    
    B_MY  = b0_MY + b1_MY * model.g(alph1_MY*np.sqrt(I)) \
                  + b2_MY * model.g(alph2_MY*np.sqrt(I))
    CT_MY = C0_MY + C1_MY * model.h(omega_MY*np.sqrt(I)) * 4
        
    Gex_nRT = fG(T,I,Aosm) \
        + mM*mX * (2*B_MX + Z*CT_MX)  \
        + mM*mY * (2*B_MY + Z*CT_MY)  \
        + mX*mY * 2*etheta(T,I,zX,zY)
    
    return Gex_nRT

# Corresponding activity coefficients
ln_acfs_MXY = egrad(Gex_MXY)

##### FITTING FUNCTIONS #######################################################

# Optimisation settings
ojac    = '3-point'
oloss   = 'linear'
omethod = 'trf'

# Input: mean activity coefficient only
def bC_acfMX(mCmA,zC,zA,T,alph1,alph2,omega,nC,nA,acfMX,which_bCs):
        
    b2 = 0
    C1 = 0
    
    # Do optimisation
    if which_bCs == 'b0b1C0':
        
        topt = optimize.least_squares(lambda bC: 
            acfMX(mCmA,zC,zA,T,bC[0],bC[1],0,bC[2],0,
                  alph1,alph2,omega,nC,nA) - acfMX,
            np.float_([0,0,0]), jac=ojac, loss=oloss, method=omethod)
        
        b0 = topt['x'][0]
        b1 = topt['x'][1]
        C0 = topt['x'][2]
        
    elif which_bCs == 'b0b1C0C1':
        
        topt = optimize.least_squares(lambda bC: 
            acfMX(mCmA,zC,zA,T,bC[0],bC[1],0,bC[2],bC[3],
                  alph1,alph2,omega,nC,nA) - acfMX,
            np.float_([0,0,0,0]), jac=ojac, loss=oloss, method=omethod)
        
        b0 = topt['x'][0]
        b1 = topt['x'][1]
        C0 = topt['x'][2]
        C1 = topt['x'][3]
        
    else:
        
        topt = optimize.least_squares(lambda bC: 
            acfMX(mCmA,zC,zA,T,bC[0],bC[1],bC[2],bC[3],bC[4],
                  alph1,alph2,omega,nC,nA) - acfMX,
            np.float_([0,0,0,0,0]), jac=ojac, loss=oloss, method=omethod)
        
        b0 = topt['x'][0]
        b1 = topt['x'][1]
        b2 = topt['x'][2]
        C0 = topt['x'][3]
        C1 = topt['x'][4]
    
    # Get covariance matrix
    mse  = topt.cost * 2 / np.size(T)
    hess = topt.jac.transpose() @ topt.jac
    bCmx = np.linalg.inv(hess) * mse
    
    return b0,b1,b2,C0,C1,bCmx,mse

# Input: mean activity OR osmotic coefficient
def bC(mCmA,zC,zA,T,alph1,alph2,omega,nC,nA,mtarg,which_bCs,mtype):
    
    b2 = 0
    C1 = 0
    
    # Select optimisation function
    if mtype == 'acf':
        ofunc = acfMX
    elif mtype == 'osm':
        ofunc = osm
    
    # Do optimisation
    if which_bCs == 'b0b1C0':
        
        topt = optimize.least_squares(lambda bC: 
            ofunc(mCmA,zC,zA,T,bC[0],bC[1],0,bC[2],0,
                   alph1,alph2,omega,nC,nA) - mtarg,
            np.float_([0,0,0]), jac=ojac, loss=oloss, method=omethod)
        
        b0 = topt['x'][0]
        b1 = topt['x'][1]
        C0 = topt['x'][2]
        
    elif which_bCs == 'b0b1C0C1':
        
        topt = optimize.least_squares(lambda bC: 
            ofunc(mCmA,zC,zA,T,bC[0],bC[1],0,bC[2],bC[3],
                   alph1,alph2,omega,nC,nA) - mtarg,
            np.float_([0,0,0,0]), jac=ojac, loss=oloss, method=omethod)
        
        b0 = topt['x'][0]
        b1 = topt['x'][1]
        C0 = topt['x'][2]
        C1 = topt['x'][3]
        
    else:
        
        topt = optimize.least_squares(lambda bC: 
            ofunc(mCmA,zC,zA,T,bC[0],bC[1],bC[2],bC[3],bC[4],
                   alph1,alph2,omega,nC,nA) - mtarg,
            np.float_([0,0,0,0,0]), jac=ojac, loss=oloss, method=omethod)
        
        b0 = topt['x'][0]
        b1 = topt['x'][1]
        b2 = topt['x'][2]
        C0 = topt['x'][3]
        C1 = topt['x'][4]
    
    # Get covariance matrix
    mse  = topt.cost * 2 / np.size(T)
    hess = topt.jac.transpose() @ topt.jac
    bCmx = np.linalg.inv(hess) * mse
    
    return b0,b1,b2,C0,C1,bCmx,mse

# Three-component H2SO4 system
def bC_MNX(TSO4,alpha,T,b0_MX,b1_MX,C0_MX,C1_MX,b0_MY,b1_MY,C0_MY,C1_MY):
    
    mSO4  = TSO4 * alpha
    mH    = TSO4 + mSO4
    mHSO4 = 2*TSO4 - mH
    
    mols = np.concatenate((mH,mHSO4,mSO4), axis=1)
    
    zM = np.float_(+1)
    zX = np.float_(-1)
    zY = np.float_(-2)
    
    alph1_MX = np.float_(2)
    alph1_MY = np.float_(2)
    
    b2_MX = 0
    b2_MY = 0
    
    alph2_MX = -9
    alph2_MY = -9
    
    omega_MX = np.float_(2.5)
    omega_MY = np.float_(2.5)
    
    ln_acfs = ln_acfs_MXY(mols,zM,zX,zY,T,
        b0_MX,b1_MX,b2_MX,C0_MX,C1_MX,alph1_MX,alph2_MX,omega_MX,
        b0_MY,b1_MY,b2_MY,C0_MY,C1_MY,alph1_MY,alph2_MY,omega_MY)
    
    gH    = np.exp(ln_acfs[:,0])
    gHSO4 = np.exp(ln_acfs[:,1])
    gSO4  = np.exp(ln_acfs[:,2])
    
    KHSO4 = coeffs.KHSO4_CRP94(T)[0]
    
    return mH.ravel() * TSO4.ravel() / (KHSO4 * gHSO4 / (gH * gSO4) \
                    + mH.ravel()) - mHSO4.ravel()
    
# This is fitting given alpha speciation values

def bC_MNX_from_alpha(TSO4,alpha,T):

    return optimize.least_squares(lambda bCs: bC_MNX(TSO4,alpha,T,
        bCs[0],bCs[1],bCs[2],bCs[3],bCs[4],bCs[5],bCs[6],bCs[7]),
        np.zeros(8), jac=ojac, loss=oloss, method=omethod)
    