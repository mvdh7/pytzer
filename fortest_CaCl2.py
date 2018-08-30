import numpy as np
import pandas as pd
import pytzer as pz
pd2vs = pz.misc.pd2vs
from PitzH2SO4 import pitzh2so4
import pickle
from time import time
from scipy import optimize

# Import CRP94 Tables 8-10
crp94 = pd.read_excel('datasets/CRP94 Tables 8-10.xlsx')
T = pd2vs(crp94.temp)

# Get Pitzer model coefficients
crp94['b0_H_HSO4'],crp94['b1_H_HSO4'],crp94['b2_H_HSO4'],\
    crp94['C0_H_HSO4'],crp94['C1_H_HSO4'], \
    crp94['alph1_H_HSO4'],crp94['alph2_H_HSO4'],crp94['omega_H_HSO4'],_ \
    = pz.coeffs.bC_H_HSO4_CRP94(T)

crp94['b0_H_SO4' ],crp94['b1_H_SO4' ],crp94['b2_H_SO4' ],\
    crp94['C0_H_SO4' ],crp94['C1_H_SO4' ], \
    crp94['alph1_H_SO4' ],crp94['alph2_H_SO4' ],crp94['omega_H_SO4' ],_ \
    = pz.coeffs.bC_H_SO4_CRP94 (T)

# Get HSO4 dissociation constant
crp94['dissoc_HSO4'] = pz.coeffs.dissoc_HSO4_CRP94(T)[0]

# Format arrays for fitting
tot       = pd2vs(crp94.tot)
b0_H_HSO4 = pd2vs(crp94.b0_H_HSO4)
b1_H_HSO4 = pd2vs(crp94.b1_H_HSO4)
b2_H_HSO4 = pd2vs(crp94.b2_H_HSO4)
C0_H_HSO4 = pd2vs(crp94.C0_H_HSO4)
C1_H_HSO4 = pd2vs(crp94.C1_H_HSO4)
alph1_H_HSO4 = pd2vs(crp94.alph1_H_HSO4)
alph2_H_HSO4 = pd2vs(crp94.alph2_H_HSO4)
omega_H_HSO4 = pd2vs(crp94.omega_H_HSO4)
b0_H_SO4  = pd2vs(crp94.b0_H_SO4)
b1_H_SO4  = pd2vs(crp94.b1_H_SO4)
b2_H_SO4  = pd2vs(crp94.b2_H_SO4)
C0_H_SO4  = pd2vs(crp94.C0_H_SO4)
C1_H_SO4  = pd2vs(crp94.C1_H_SO4)
alph1_H_SO4 = pd2vs(crp94.alph1_H_SO4)
alph2_H_SO4 = pd2vs(crp94.alph2_H_SO4)
omega_H_SO4 = pd2vs(crp94.omega_H_SO4)
dissoc_HSO4 = pd2vs(crp94.dissoc_HSO4)

# Set up for Python solving approach
zH    = np.float_(+1)
zHSO4 = np.float_(-1)
zSO4  = np.float_(-2)

# Set up to solve for mH
TSO4 = pd2vs(crp94.tot)

def minifun(mH,TSO4,zM,zX,zY,T,
            b0_MX,b1_MX,b2_MX,C0_MX,C1_MX,alph1_MX,alph2_MX,omega_MX,
            b0_MY,b1_MY,b2_MY,C0_MY,C1_MY,alph1_MY,alph2_MY,omega_MY,
            dissoc_MX):

    # Calculate ionic speciation
    mH = np.vstack(mH)
    mHSO4 = 2*TSO4 - mH
    mSO4  = mH - TSO4

    # Create molality & ions arrays
    mols = np.concatenate((mH,mHSO4,mSO4), axis=1)

    # Calculate activity coefficients
    ln_acfs = pz.fitting.ln_acfs_MXY(mols,zM,zX,zY,T,
            b0_MX,b1_MX,b2_MX,C0_MX,C1_MX,alph1_MX,alph2_MX,omega_MX,
            b0_MY,b1_MY,b2_MY,C0_MY,C1_MY,alph1_MY,alph2_MY,omega_MY)
    gH    = np.exp(ln_acfs[:,0])
    gHSO4 = np.exp(ln_acfs[:,1])
    gSO4  = np.exp(ln_acfs[:,2])

    DG = np.log(gH*mH * gSO4*mSO4 / (gHSO4*mHSO4)) \
       - np.log(dissoc_MX)

    return DG

# Solve speciation - Fortran
F = {}
F['mH']       = np.full_like(T,np.nan)
F['mHSO4']    = np.full_like(T,np.nan)
F['mSO4']     = np.full_like(T,np.nan)
F['osmST']    = np.full_like(T,np.nan)
F['ln_acfPM'] = np.full_like(T,np.nan)
Fgo = time()
for i in range(len(T)):
    F['mH'][i],F['mHSO4'][i],F['mSO4'][i],F['osmST'][i],F['ln_acfPM'][i] \
        = pitzh2so4(T[i],tot[i],
        b0_H_HSO4[i],b1_H_HSO4[i],C0_H_HSO4[i],C1_H_HSO4[i],
        b0_H_SO4 [i],b1_H_SO4 [i],C0_H_SO4 [i],C1_H_SO4 [i],
        alph1_H_SO4[i],1/dissoc_HSO4[i])
print('Fortran: ' + str(time()-Fgo))

# Solve speciation - Python
P = {}
P['mH']       = np.full_like(T,np.nan)
P['osmST']    = np.full_like(T,np.nan)
P['ln_acfPM'] = np.full_like(T,np.nan)
Pgo = time()
for i in range(len(T)):
    P['mH'][i] = optimize.least_squares(lambda mH: minifun(mH,TSO4[i],
            zH,zHSO4,zSO4,T[i],
            b0_H_HSO4[i],b1_H_HSO4[i],b2_H_HSO4[i],C0_H_HSO4[i],C1_H_HSO4[i],
            alph1_H_HSO4[i],alph2_H_HSO4[i],omega_H_HSO4[i],
            b0_H_SO4 [i],b1_H_SO4 [i],b2_H_SO4 [i],C0_H_SO4 [i],C1_H_SO4 [i],
            alph1_H_SO4 [i],alph2_H_SO4 [i],omega_H_SO4 [i],
            dissoc_HSO4[i]).ravel(),
                                   1.5*TSO4[i],
                                   bounds=(TSO4[i],2*TSO4[i]),
                                   method='trf',
                                   xtol=1e-12)['x']

P['mSO4']  = P['mH'] - TSO4
P['mHSO4'] = TSO4 - P['mSO4']
P['alpha'] = P['mSO4'] / TSO4
print('Python : ' + str(time()-Fgo))

# Save results
with open('fortest6.pkl','wb') as f:
    pickle.dump((crp94,F,P),f)
