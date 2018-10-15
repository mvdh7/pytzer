from autograd import numpy as np
from autograd import elementwise_grad as egrad
from scipy import optimize
from . import model
from .constants import Mw, R

##### FREEZING POINT DEPRESSION ###############################################

# Convert freezing point depression to water activity
def fpd2aw(fpd):

    # Equation coefficients from S.L. Clegg (pers. comm., 2018)
    lg10aw = \
        - np.float_(4.209099e-03) * fpd    \
        - np.float_(2.151997e-06) * fpd**2 \
        + np.float_(3.233395e-08) * fpd**3 \
        + np.float_(3.445628e-10) * fpd**4 \
        + np.float_(1.758286e-12) * fpd**5 \
        + np.float_(7.649700e-15) * fpd**6 \
        + np.float_(3.117651e-17) * fpd**7 \
        + np.float_(1.228438e-19) * fpd**8 \
        + np.float_(4.745221e-22) * fpd**9
    
    return np.exp(lg10aw * np.log(10))

# Convert freezing point depression to osmotic coefficient
def fpd2osm(mols,fpd):
    return model.aw2osm(mols,fpd2aw(fpd))

# Get expected FPD at a given molality - assuming that the Pitzer model works
#  down to the FPD...
def tot2fpd(tot,ions,nC,nA,cf):
    
    # One electrolyte at a time - nC and nA are size 1
    
    mols = np.concatenate((tot*nC,tot*nA), axis=1)
    fpd = np.full_like(tot,np.nan)
    
    iT00 = np.vstack([273.15])
    
    for i in range(len(tot)):
        
        imols = np.array([mols[i,:]])
        
        fpd[i] = optimize.least_squares(lambda fpd: \
           (fpd2osm(imols,fpd) - model.osm(imols,ions,iT00-fpd,cf)).ravel(),
                                        5., method='lm')['x'][0]
    
    return fpd

#xmols = np.array([mols[0,:]])
#xfpd = optimize.least_squares(lambda fpd: \
#    (  pz.tconv.fpd2osm(xmols,fpd) \
#     - pz.model.osm(xmols,ions,273.15-fpd,cf)).ravel(),
#                              0.)

# Get expected FPD at a given molality with T conversion
def tot2fpd25(tot,ions,nC,nA,cf):
        
    # One electrolyte at a time - nC and nA are size 1
    
    mols = np.concatenate((tot*nC,tot*nA), axis=1)
    fpd = np.full_like(tot,np.nan)
    T25 = np.full_like(tot,298.15, dtype='float64')
    
    osm25 = model.osm(mols,ions,T25,cf)
    
    iT25 = np.vstack([298.15])
    iT00 = np.vstack([273.15])
    
    for i in range(len(tot)):
        
        if i/10. == np.round(i/10.):
            print('Getting FPD %d of %d...' % (i+1,len(tot)))
        
        imols = np.array([mols[i,:]])
        
        fpd[i] = optimize.least_squares(lambda fpd: \
           (osm2osm(tot[i],nC,nA,ions,iT00-fpd,iT25,iT25,cf,
                    fpd2osm(imols,fpd)) - osm25[i]).ravel(),
                                        0., method='trf')['x'][0]
    
    return fpd

##### TEMPERATURE CONVERSION ##################################################

# --- Temperature subfunctions ------------------------------------------------
    
def y(T0,T1):
    return (T1 - T0) / (R * T0 * T1)

def z(T0,T1):
    return T1 * y(T0,T1) - np.log(T1 / T0) / R

def O(T0,T1):
    return T1 * (z(T0,T1) + (T0 - T1)*y(T0,T1) / 2)

# --- Heat capacity derivatives -----------------------------------------------

# wrt. molality
dCpapp_dm = egrad(model.Cpapp)

def J1(tot,n1,n2,ions,T,cf): # HO58 Ch. 8 Eq. (8-4-9)
    return -Mw * tot**2 * dCpapp_dm(tot,n1,n2,ions,T,cf)

def J2(tot,n1,n2,ions,T,cf): # HO58 Ch. 8 Eq. (8-4-7)
    return tot * dCpapp_dm(tot,n1,n2,ions,T,cf)

# wrt. temperature
G1 = egrad(J1,argnum=4)
G2 = egrad(J2,argnum=4)

# --- Enthalpy derivatives ----------------------------------------------------
    
# wrt. molality
dLapp_dm = egrad(model.Lapp)

def L1(tot,n1,n2,ions,T,cf): # HO58 Ch. 8 Eq. (8-4-9)
    return -Mw * tot**2 * dLapp_dm(tot,n1,n2,ions,T,cf)

def L2(tot,n1,n2,ions,T,cf): # HO58 Ch. 8 Eq. (8-4-7)
    return    model.Lapp(tot,n1,n2,ions,T,cf) \
        + tot * dLapp_dm(tot,n1,n2,ions,T,cf)

# --- Execute temperature conversion ------------------------------------------

# Osmotic coefficient
def osm2osm(tot,n1,n2,ions,T0,T1,TR,cf,osm_T0):
    
    tot = np.vstack(tot)
    T0  = np.vstack(T0)
    T1  = np.vstack(T1)
    TR  = np.vstack(TR)
    
    lnAW_T0 = -osm_T0 * tot * (n1 + n2) * Mw
    
    lnAW_T1 = lnAW_T0 - y(T0,T1) * L1(tot,n1,n2,ions,TR,cf) \
                      + z(T0,T1) * J1(tot,n1,n2,ions,TR,cf) \
                      - O(T0,T1) * G1(tot,n1,n2,ions,TR,cf)

    return -lnAW_T1 / (tot * (n1 + n2) * Mw)
    
# Solute mean activity coefficient
def acf2acf(tot,n1,n2,ions,T0,T1,TR,cf,acf_T0):
    
    tot = np.vstack(tot)
    T0  = np.vstack(T0)
    T1  = np.vstack(T1)
    TR  = np.vstack(TR)
    
    ln_acf_T0 = np.log(acf_T0)
    
    ln_acf_T1 = ln_acf_T0 + (- y(T0,T1) * L2(tot,n1,n2,ions,TR,cf)  \
                             + z(T0,T1) * J2(tot,n1,n2,ions,TR,cf)  \
                             - O(T0,T1) * G2(tot,n1,n2,ions,TR,cf)) \
                          / (n1 + n2)
    
    return np.exp(ln_acf_T1)
