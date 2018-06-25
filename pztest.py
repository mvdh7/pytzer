import autograd.numpy as np
from autograd import elementwise_grad as egrad
#from scipy.misc import derivative
from scipy import optimize
from scipy.optimize import minimize
import pytzer as pz
#from pytzer.constants import R, Mw
import time

## Set dict of coefficient functions
#cf = pz.cdicts.GM89
#
#for ca in ['Ca-OH','H-Cl','H-OH','H-SO4','Na-OH']:
#    cf.bC[ca]    = pz.coeffs.zero_bC
#
#for ii in ['H-Na','Ca-H','Cl-OH','OH-SO4']:
#    cf.theta[ii] = pz.coeffs.zero_theta
#
#for iij in ['Ca-Na-OH','H-Na-Cl','H-Na-SO4','H-Na-OH','Ca-H-Cl','Ca-H-SO4',
#            'Ca-H-OH','H-Cl-SO4','Na-Cl-OH','Ca-Cl-OH','H-Cl-OH','Na-OH-SO4',
#            'Ca-OH-SO4','H-OH-SO4']:
#    cf.psi[iij]  = pz.coeffs.zero_psi
#
## Import test dataset
#T,tots,ions,idf = pz.miami.getIons('M88 Table 4.csv')
#mols = np.copy(tots)

cf = pz.cdicts.CRP94
cf.bC['H-OH'] = pz.coeffs.zero_bC
    
T,tots,ions,idf = pz.miami.getIons('CRP94 solver.csv')
#mols = np.copy(tots)
#
## Calculate excess Gibbs energy and activity coeffs (no dissociation)
#Gexs = pz.miami.Gex_nRT(mols,ions,T,cf)
#acfs = np.exp(pz.miami.ln_acfs(mols,ions,T,cf))
#
## Test osmotic coefficients - NaCl compares well with Archer (1992)
## M88 Table 4 also works almost perfectly, without yet including unsymm. terms!
#osm = pz.miami.osm(mols,ions,T,cf)
#aw  = pz.miami.osm2aw(mols,osm)
#
#osmST = osm * np.sum(mols,axis=1) / (3 * np.sum(mols[:,1:],axis=1))
#
## Differentiate fG wrt I
#dfG_dI = egrad(pz.miami.fG, argnum=1)
#dfG = dfG_dI(np.array([298.15]),np.array([6.]),cf)
#
## Get mean activity coefficient (CRP94)
#acf_mean = np.cbrt(acfs[:,0]**2 * acfs[:,2] * mols[:,0]**2 * mols[:,2] \
#    / (4 * np.sum(mols[:,1:],axis=1)**3))

## Solve for pH - M88 system
#def minifun(pH,mols,ions,T,cf):
#    
#    # Calculate [H+] and [OH-]
#    mH  = np.vstack(np.full_like(mols[:,0],-np.log10(pH)))
#    mOH = np.copy(mH)
#    
#    # Add them to main arrays
#    mols = np.concatenate((mols,mH,mOH), axis=1)
#    ions = np.append(ions,['H','OH'])
#    
#    # Calculate activity coefficients
#    ln_acfs = pz.miami.ln_acfs(mols,ions,T,cf)
#    gH  = np.exp(ln_acfs[:,4])
#    gOH = np.exp(ln_acfs[:,5])
#    
#    # Set up DG equation
#    DG = np.log(gH*mH.ravel() * gOH*mOH.ravel()) \
#        - np.log(cf.K['H2O'](T)[0])
#    
#    return DG

# Solve for pH - CRP94 system
def minifun(mH,tots,ions,T,cf):
    
    # Calculate [H+] and ionic speciation
#    mH = np.vstack(-np.log10(pH))
    mH = np.vstack(mH)
    mHSO4 = 2*tots - mH
    mSO4  = mH - tots
    
    # Create molality & ions arrays
    mols = np.concatenate((mH,mHSO4,mSO4), axis=1)
    ions = np.array(['H','HSO4','SO4'])
    
    # Calculate activity coefficients
    ln_acfs = pz.miami.ln_acfs(mols,ions,T,cf)
    gH    = np.exp(ln_acfs[:,0])
    gHSO4 = np.exp(ln_acfs[:,1])
    gSO4  = np.exp(ln_acfs[:,2])

    # Set up DG equation
#    DG = np.log(gH*mH.ravel() * gSO4*mSO4.ravel() / (gHSO4*mHSO4.ravel())) \
#        - np.log(cf.K['HSO4'](T)[0])
    
    DG = cf.getKeq(T, mH=mH,gH=gH, mHSO4=mHSO4,gHSO4=gHSO4, 
                   mSO4=mSO4,gSO4=gSO4)
    
    return DG

#go = time.time()

EQ = np.full_like(T,np.nan)
for i in range(len(EQ)):
    
    iT = np.array([T[i]])
    itots = np.array([tots[i,:]])
    
#    EQ[i] = minimize(lambda pH:minifun(pH,itots,ions,iT,cf),1., \
#                     method='Nelder-Mead')['x'][0]
    
    EQ[i] = optimize.least_squares(lambda mH: minifun(mH,itots,ions,iT,cf),
                                   1.5*itots[0],
                                   bounds=(itots[0],2*itots[0]),
                                   method='trf',
                                   xtol=1e-12)['x']

#print(time.time()-go)

#EQ = np.full_like(T,np.nan)
#for i in range(len(EQ)):
#    
#    iT = np.array([T[i]])
#    imols = np.array([mols[i,:]])
#    
#    EQ[i] = minimize(lambda pH:minifun(pH,imols,ions,iT,cf)**2,7.)['x'][0]
