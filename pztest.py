import autograd.numpy as np
from autograd import elementwise_grad as egrad
#from scipy.misc import derivative
from scipy.optimize import minimize
import pytzer as pz
#from pytzer.constants import R, Mw

# Set dict of coefficient functions
cf = pz.cdicts.GM89

for ca in ['Ca-OH','H-Cl','H-OH','H-SO4','Na-OH']:
    cf.bC[ca]    = pz.coeffs.zero_bC

for ii in ['H-Na','Ca-H','Cl-OH','OH-SO4']:
    cf.theta[ii] = pz.coeffs.zero_theta

for iij in ['Ca-Na-OH','H-Na-Cl','H-Na-SO4','H-Na-OH','Ca-H-Cl','Ca-H-SO4',
            'Ca-H-OH','H-Cl-SO4','Na-Cl-OH','Ca-Cl-OH','H-Cl-OH','Na-OH-SO4',
            'Ca-OH-SO4','H-OH-SO4']:
    cf.psi[iij]  = pz.coeffs.zero_psi

# Import test dataset
T,tots,ions,idf = pz.miami.getIons('M88 Table 4.csv')
mols = np.copy(tots)

# Calculate excess Gibbs energy and activity coeffs (no dissociation)
Gexs = pz.miami.Gex_nRT(mols,ions,T,cf)
acfs = np.exp(pz.miami.ln_acfs(mols,ions,T,cf))

# Test osmotic coefficients - NaCl compares well with Archer (1992)
# M88 Table 4 also works almost perfectly, without yet including unsymm. terms!
osm = pz.miami.osm(mols,ions,T,cf)
aw  = pz.miami.osm2aw(mols,osm)

# Differentiate fG wrt I
dfG_dI = egrad(pz.miami.fG, argnum=1)
dfG = dfG_dI(np.array([298.15]),np.array([6.]),cf)

# Solve for pH
def minifun(pH,mols,ions,T,cf):
    
    # Calculate [H+] and [OH-]
    mH  = np.vstack(np.full_like(mols[:,0],10**-pH))
    mOH = np.copy(mH)
    
    # Add them to main arrays
    mols = np.concatenate((mols,mH,mOH), axis=1)
    ions = np.append(ions,['H','OH'])
    
    # Calculate activity coefficients
    ln_acfs = pz.miami.ln_acfs(mols,ions,T,cf)
    gH  = np.exp(ln_acfs[:,4])
    gOH = np.exp(ln_acfs[:,5])
    
    # Set up DG equation
    DG = np.log(gH*mH.ravel() * gOH*mOH.ravel()) \
        - np.log(cf.K['H2O'](T)[0])
    
    return DG

EQ = np.full_like(T,np.nan)
for i in range(len(EQ)):
    
    iT = np.array([T[i]])
    imols = np.array([mols[i,:]])
    
    EQ[i] = minimize(lambda pH:minifun(pH,imols,ions,iT,cf)**2,7.)['x'][0]

