import numpy as np
import pandas as pd
import pytzer as pz
pd2vs = pz.misc.pd2vs
from PitzCaCl2 import pitzcacl2
import pickle
from time import time
from scipy.io import savemat
from sys import argv
#from scipy import optimize

# Determine which set of coefficients to use - 9 or 10
whichtab = argv[1]

# Import RC97 Table 11
rc97 = pd.read_excel('datasets/RC97 Table 11.xlsx')
T   = pd2vs(rc97.temp)
tot = pd2vs(rc97.tot)

# Define Pitzer model coefficients (RC97 Table 9)
if whichtab == '9':
    b0_1 = np.float_(-0.298649948)
    b1_1 = np.float_( 2.17867147 )
    C0_1 = np.float_( 0.022524497)
    C1_1 = np.float_( 0.540824906)
    b0_2 = np.float_(-1.01330402 )
    b1_2 = np.float_( 7.27456368 )
    C0_2 = np.float_( 0.064161134)
    C1_2 = np.float_(-3.69567654 )

    D_MX3  = 0
    D_MX4  = 0
    D_MX5  = 0
    D_NX3  = 0
    D_NX4  = 0
    D_NX5  = 0
    D_MNX2 = 0
    D_MNX3 = 0

# Define Pitzer model coefficients (RC97 Table 10)
elif whichtab == '10':
    b0_1 = np.float_(- 0.951245617)
    b1_1 = np.float_(  3.06698827 )
    C0_1 = np.float_(- 0.062276316)
    C1_1 = np.float_(  1.68840214 )
    b0_2 = np.float_(  0          )
    b1_2 = np.float_(-14.7306408  )
    C0_2 = np.float_(  0.059275582)
    C1_2 = np.float_( 21.2226853  )

    D_MX3  = np.float_( 0.0303640612 )
    D_MX4  = np.float_(-0.0019593304 )
    D_MX5  = np.float_( 0.00004710087)
    D_NX3  = np.float_( 0            )
    D_NX4  = np.float_(-0.0001348980 )
    D_NX5  = np.float_( 0            )
    D_MNX2 = np.float_(-0.0105349140 )
    D_MNX3 = np.float_( 0            )

# Solve speciation - Fortran
F = {}
F['tot']      = tot
F['mCaCl']    = np.full_like(T,np.nan)
F['mCa']      = np.full_like(T,np.nan)
F['mCl']      = np.full_like(T,np.nan)
F['alpha']    = np.full_like(T,np.nan)
F['osm']      = np.full_like(T,np.nan)
F['ln_acfPM'] = np.full_like(T,np.nan)

Fgo = time()

for i in range(len(T)):
    F['mCaCl'][i],F['mCa'][i],F['mCl'][i],F['alpha'][i],F['osm'][i], \
        F['ln_acfPM'][i] \
        = pitzcacl2(T[i],tot[i], b0_1,b1_1,C0_1,C1_1,b0_2,b1_2,C0_2,C1_2,
                    D_MX3,D_MX4,D_MX5,D_NX3,D_NX4,D_NX5,D_MNX2,D_MNX3)

print('Fortran: ' + str(time()-Fgo))

## Set up for Python solving approach
#zH    = np.float_(+1)
#zHSO4 = np.float_(-1)
#zSO4  = np.float_(-2)
#
## Set up to solve for mH
#TSO4 = pd2vs(rc97.tot)
#
#def minifun(mH,TSO4,zM,zX,zY,T,
#            b0_MX,b1_MX,b2_MX,C0_MX,C1_MX,alph1_MX,alph2_MX,omega_MX,
#            b0_MY,b1_MY,b2_MY,C0_MY,C1_MY,alph1_MY,alph2_MY,omega_MY,
#            dissoc_MX):
#
#    # Calculate ionic speciation
#    mH = np.vstack(mH)
#    mHSO4 = 2*TSO4 - mH
#    mSO4  = mH - TSO4
#
#    # Create molality & ions arrays
#    mols = np.concatenate((mH,mHSO4,mSO4), axis=1)
#
#    # Calculate activity coefficients
#    ln_acfs = pz.fitting.ln_acfs_MXY(mols,zM,zX,zY,T,
#            b0_MX,b1_MX,b2_MX,C0_MX,C1_MX,alph1_MX,alph2_MX,omega_MX,
#            b0_MY,b1_MY,b2_MY,C0_MY,C1_MY,alph1_MY,alph2_MY,omega_MY)
#    gH    = np.exp(ln_acfs[:,0])
#    gHSO4 = np.exp(ln_acfs[:,1])
#    gSO4  = np.exp(ln_acfs[:,2])
#
#    DG = np.log(gH*mH * gSO4*mSO4 / (gHSO4*mHSO4)) \
#       - np.log(dissoc_MX)
#
#    return DG
#
## Solve speciation - Python
#P = {}
#P['mH']       = np.full_like(T,np.nan)
#P['osmST']    = np.full_like(T,np.nan)
#P['ln_acfPM'] = np.full_like(T,np.nan)
#Pgo = time()
#for i in range(len(T)):
#    P['mH'][i] = optimize.least_squares(lambda mH: minifun(mH,TSO4[i],
#            zH,zHSO4,zSO4,T[i],
#            b0_H_HSO4[i],b1_H_HSO4[i],b2_H_HSO4[i],C0_H_HSO4[i],C1_H_HSO4[i],
#            alph1_H_HSO4[i],alph2_H_HSO4[i],omega_H_HSO4[i],
#            b0_H_SO4 [i],b1_H_SO4 [i],b2_H_SO4 [i],C0_H_SO4 [i],C1_H_SO4 [i],
#            alph1_H_SO4 [i],alph2_H_SO4 [i],omega_H_SO4 [i],
#            dissoc_HSO4[i]).ravel(),
#                                   1.5*TSO4[i],
#                                   bounds=(TSO4[i],2*TSO4[i]),
#                                   method='trf',
#                                   xtol=1e-12)['x']
#
#P['mSO4']  = P['mH'] - TSO4
#P['mHSO4'] = TSO4 - P['mSO4']
#P['alpha'] = P['mSO4'] / TSO4
#print('Python : ' + str(time()-Fgo))

# Save results
with open('pickles/fortest_CaCl2_' + whichtab + '.pkl','wb') as f:
    pickle.dump((rc97,F),f)

savemat('pickles/fortest_CaCl2_' + whichtab + '.mat',F)
