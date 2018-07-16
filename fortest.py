import numpy as np
import pandas as pd
import pytzer as pz
pd2vs = pz.misc.pd2vs
from PitzH2SO4 import pitzh2so4
import pickle

# Import CRP94 Tables 8-10
crp94 = pd.read_excel('datasets/CRP94 Tables 8-10.xlsx')
T = pd2vs(crp94.temp)

# Get Pitzer model coefficients
crp94['b0_H_HSO4'],crp94['b1_H_HSO4'],_,\
    crp94['C0_H_HSO4'],crp94['C1_H_HSO4'],_,_,_,_ \
    = pz.coeffs.bC_H_HSO4_CRP94(T)

crp94['b0_H_SO4'],crp94['b1_H_SO4'],_,\
    crp94['C0_H_SO4'],crp94['C1_H_SO4'],crp94['alph1_H_SO4'],_,_,_ \
    = pz.coeffs.bC_H_SO4_CRP94(T)

# Get HSO4 dissociation constant
crp94['dissoc_HSO4'] = pz.coeffs.dissoc_HSO4_CRP94(T)[0]

# Format arrays for Fortran
fT         = T.ravel()
ftot       = pd2vs(crp94.tot).ravel()
fb0_H_HSO4 = pd2vs(crp94.b0_H_HSO4).ravel()
fb1_H_HSO4 = pd2vs(crp94.b1_H_HSO4).ravel()
fC0_H_HSO4 = pd2vs(crp94.C0_H_HSO4).ravel()
fC1_H_HSO4 = pd2vs(crp94.C1_H_HSO4).ravel()
fb0_H_SO4  = pd2vs(crp94.b0_H_SO4).ravel()
fb1_H_SO4  = pd2vs(crp94.b1_H_SO4).ravel()
fC0_H_SO4  = pd2vs(crp94.C0_H_SO4).ravel()
fC1_H_SO4  = pd2vs(crp94.C1_H_SO4).ravel()
falph1     = pd2vs(crp94.alph1_H_SO4).ravel()
fdissoc    = pd2vs(crp94.dissoc_HSO4).ravel()

# Solve speciation
mH       = np.full_like(fT,np.nan)
mHSO4    = np.full_like(fT,np.nan)
mSO4     = np.full_like(fT,np.nan)
osmST    = np.full_like(fT,np.nan)
ln_acfPM = np.full_like(fT,np.nan)
for i in range(len(fT)):
    mH[i],mHSO4[i],mSO4[i],osmST[i],ln_acfPM[i] \
        = pitzh2so4(fT[i],ftot[i],
        fb0_H_HSO4[i],fb1_H_HSO4[i],fC0_H_HSO4[i],fC1_H_HSO4[i],
        fb0_H_SO4 [i],fb1_H_SO4 [i],fC0_H_SO4 [i],fC1_H_SO4 [i],
        falph1[i],fdissoc[i])

# Save results
with open('fortest4.pkl','wb') as f:
    pickle.dump((crp94,mH,mHSO4,mSO4,osmST,ln_acfPM),f)
