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
ftot       = crp94.tot.values
fb0_H_HSO4 = crp94.b0_H_HSO4.values
fb1_H_HSO4 = crp94.b1_H_HSO4.values
fC0_H_HSO4 = crp94.C0_H_HSO4.values
fC1_H_HSO4 = crp94.C1_H_HSO4.values
fb0_H_SO4  = crp94.b0_H_SO4.values
fb1_H_SO4  = crp94.b1_H_SO4.values
fC0_H_SO4  = crp94.C0_H_SO4.values
fC1_H_SO4  = crp94.C1_H_SO4.values
falph1     = crp94.alph1_H_SO4.values
fdissoc    = crp94.dissoc_HSO4.values

# Solve speciation
test = pitzh2so4(fT,ftot,fb0_H_HSO4,fb1_H_HSO4,fC0_H_HSO4,fC1_H_HSO4,
                 fb0_H_SO4,fb1_H_SO4,fC0_H_SO4,fC1_H_SO4,falph1,fdissoc)

# Save results
with open('fortest1.pkl','wb') as f:
    pickle.dump((crp94,test),f)
