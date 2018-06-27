from autograd import numpy as np
from autograd import elementwise_grad as egrad
from scipy import optimize
from .constants import b, R
from . import coeffs, model

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

##### FITTING FUNCTIONS #######################################################

# Input: mean activity coefficient only
def bC_acfMX(mCmA,zC,zA,T,alph1,alph2,omega,nC,nA,acfMX,which_bCs):
    
    # Optimisation settings
    ojac    = '3-point'
    oloss   = 'linear'
    omethod = 'trf'
    
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
    
    # Optimisation settings
    ojac    = '3-point'
    oloss   = 'linear'
    omethod = 'trf'
    
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