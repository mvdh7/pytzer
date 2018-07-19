import numpy as np
import pandas as pd
import pytzer as pz
pd2vs = pz.misc.pd2vs
from PitzH2SO4 import pitzh2so4
import pickle
from time import time

# Import CRP94 Tables 8-10
crp94 = pd.read_excel('datasets/CRP94 Tables 8-10.xlsx')
T    = pd2vs(crp94.temp)
TSO4 = pd2vs(crp94.tot)
dissoc_HSO4 = pz.coeffs.dissoc_HSO4_CRP94(T)[0]

# Import q coefficients
q = np.loadtxt('datasets/H2SO4 fits/H2SO4 coeffs.txt', skiprows=9)
q = q[:27,1]

# Import their covariance matrix
qmx = np.loadtxt('datasets/H2SO4 fits/H2SO4 covmx.txt', skiprows=17)
qmx = qmx[:27,:27]

# Set up function to evaluate new fit (ie. imported coefficients)
def CRP94new(T,q):
    
    # H-HSO4
    b0_H_HSO4 = pz.coeffs.CRP94_eq24(T,q[:4])
    b1_H_HSO4 = pz.coeffs.CRP94_eq24(T,np.array([q[4], 1.5, q[5], q[6]]))
    C0_H_HSO4 = pz.coeffs.CRP94_eq24(T,q[ 7:11]) / 2
    C1_H_HSO4 = pz.coeffs.CRP94_eq24(T,np.array([-0.025,
                                                 q[23],
                                                 q[24],
                                                 0.0025])) / 2
    alph1_H_HSO4 = np.float_(2)
    omega_H_HSO4 = np.float_(2.5)
    
    # H-SO4
    b0_H_SO4  = pz.coeffs.CRP94_eq24(T,q[11:15])
    b1_H_SO4  = pz.coeffs.CRP94_eq24(T,q[15:19])
    C0_H_SO4  = pz.coeffs.CRP94_eq24(T,q[19:23]) / (2 * np.sqrt(2))
    C1_H_SO4  = pz.coeffs.CRP94_eq24(T,np.array([0,
                                                 -0.176776695,
                                                 q[25],
                                                 0])) / (2 * np.sqrt(2))
    alph1_H_SO4 = 2 + 100 * q[26] * (1/T - 1/298.15)
    omega_H_SO4  = np.float_(2.5)
    
    return b0_H_HSO4,b1_H_HSO4,C0_H_HSO4,C1_H_HSO4,alph1_H_HSO4,omega_H_HSO4, \
           b0_H_SO4 ,b1_H_SO4 ,C0_H_SO4 ,C1_H_SO4 ,alph1_H_SO4 ,omega_H_SO4

# Solve speciation - Fortran
Ureps = 10
F = {var:np.full((np.size(T),Ureps),np.nan) \
     for var in ['mH','mHSO4','mSO4','osmST','ln_acfPM']}
Fgo = time()
for u in range(Ureps):
    
    # Randomly perturb the q coefficients
    qu = np.random.multivariate_normal(q,qmx)
    
    b0_H_HSO4,b1_H_HSO4,C0_H_HSO4,C1_H_HSO4,_,_, \
        b0_H_SO4,b1_H_SO4,C0_H_SO4,C1_H_SO4,alph1_H_SO4,_ = CRP94new(T,qu)
    
    # Solve speciation (Fortran)
    for i in range(len(T)):
        F['mH'][i,u],F['mHSO4'][i,u],F['mSO4'][i,u],F['osmST'][i,u], \
        F['ln_acfPM'][i,u] = pitzh2so4(T[i],TSO4[i],
            b0_H_HSO4[i],b1_H_HSO4[i],C0_H_HSO4[i],C1_H_HSO4[i],
            b0_H_SO4 [i],b1_H_SO4 [i],C0_H_SO4 [i],C1_H_SO4 [i],
            alph1_H_SO4[i],1/dissoc_HSO4[i])
        
print('Runtime: ' + str(time()-Fgo))

# Save results
with open('fortest_sim.pkl','wb') as f:
    pickle.dump((crp94,F),f)
