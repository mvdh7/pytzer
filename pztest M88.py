import autograd.numpy as np
from scipy import optimize
from time import time
import pytzer as pz
pd2vs = pz.misc.pd2vs

# Set dict of coefficient functions
cf = pz.cdicts.M88

for ca in ['Ca-OH','H-Cl','H-SO4','Na-OH']:
    cf.bC[ca]    = pz.coeffs.bC_zero

for ii in ['H-Na','Ca-H','Cl-OH','OH-SO4']:
    cf.theta[ii] = pz.coeffs.theta_zero

for iij in ['Ca-Na-OH','H-Na-Cl','H-Na-SO4','H-Na-OH','Ca-H-Cl','Ca-H-SO4',
            'Ca-H-OH','H-Cl-SO4','Na-Cl-OH','Ca-Cl-OH','H-Cl-OH','Na-OH-SO4',
            'Ca-OH-SO4','H-OH-SO4']:
    cf.psi[iij]  = pz.coeffs.psi_zero

# Import test dataset
T,mols,ions,idf = pz.io.getIons('datasets/M88 Table 4.csv')

# Calculate excess Gibbs energy and activity coeffs (no dissociation)
idf['Gex'] = pz.model.Gex_nRT(mols,ions,T,cf)
acfs = np.exp(pz.model.ln_acfs(mols,ions,T,cf))

# Test osmotic coefficients vs M88 Table 4 - works almost perfectly!
idf['osm'] = pz.model.osm(mols,ions,T,cf)
idf['aw' ] = pz.model.osm2aw(mols,pd2vs(idf.osm))

# Set up function to solve for pH
def minifunc_M88(pH,mols,ions,T,cf):
    
    # Calculate [OH-]
    mH  = np.vstack(10**-pH)
    mOH = np.copy(mH)
    
    # Add them to main arrays
    mols = np.concatenate((mols,mH,mOH), axis=1)
    ions = np.append(ions,['H','OH'])
    
    # Calculate activity coefficients
    ln_acfs = pz.model.ln_acfs(mols,ions,T,cf)
    gH  = np.vstack(np.exp(ln_acfs[:,4]))
    gOH = np.vstack(np.exp(ln_acfs[:,5]))
    
    # Set up DG equation
    DG = cf.getKeq(T, mH=mH,gH=gH, mOH=mOH,gOH=gOH)
    
    return DG

# Solve for pH with minimize
go = time()    

pH_min = np.full_like(T,np.nan)
for i in range(len(pH_min)):
    
    iT    = np.vstack(T[i])
    imols = np.vstack([mols[i,:]])
    
    pH_min[i] = optimize.minimize(lambda pH: \
        minifunc_M88(pH,imols,ions,iT,cf)**2,5.,
        method='L-BFGS-B')['x'][0]

print('min: ' + str(time()-go))

idf['pH_min'] = pH_min

# Evaluate DG at solution (should be zero)
idf['DG_min'] = minifunc_M88(pH_min,mols,ions,T,cf)

# Solve for pH with least_squares
go = time()

pH_lsq = np.full_like(T,np.nan)
for i in range(len(pH_lsq)):
    
    iT    = np.vstack(T[i])
    imols = np.vstack([mols[i,:]])
    
    pH_lsq[i] = optimize.least_squares(lambda pH: \
        minifunc_M88(pH,imols,ions,iT,cf).ravel(),5.,
        method='trf')['x'][0]

print('lsq: ' + str(time()-go))

idf['pH_lsq'] = pH_lsq

# Evaluate DG at solution (should be zero)
idf['DG_lsq'] = minifunc_M88(pH_lsq,mols,ions,T,cf)