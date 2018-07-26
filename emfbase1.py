#%%
from autograd import numpy as np
import pytzer as pz
from pytzer.constants import R,F
pd2vs = pz.misc.pd2vs
#
emfbase = pz.data.emf('datasets/')

cf = pz.cdicts.cdict()
cf.bC['Na-Cl'] = pz.coeffs.bC_Na_Cl_A92ii
cf.dh['Aosm'] = pz.coeffs.Aosm_MPH

tot  = pd2vs(emfbase.m)
T    = pd2vs(emfbase.t) 
mols = np.concatenate((tot,tot), axis=1)
ions = np.array(['Na','Cl'])

ln_acfs = pz.model.ln_acfs(mols,ions,T,cf)

rmols = np.full_like(mols,0.1)
rln_acfs = pz.model.ln_acfs(rmols,ions,T,cf)

# Calculate A92ii activities
emfbase['acfMX_calc'] = np.exp(np.vstack(
    pz.model.ln_acf2ln_acf_MX(ln_acfs[:,0],ln_acfs[:,1],
                              emfbase.nC.values,emfbase.nA.values)))

emfbase['racfMX_calc'] = np.exp(np.vstack(
    pz.model.ln_acf2ln_acf_MX(rln_acfs[:,0],rln_acfs[:,1],
                              emfbase.nC.values,emfbase.nA.values)))

emfbase['ln_acf_racf_calc'] = np.log(emfbase.acfMX_calc / emfbase.racfMX_calc)

# Calculate activity ratio from measurements
emfbase['ln_acf_racf'] = F * pd2vs(emfbase.emf) / (2 * R * T) \
    - np.log(tot / 0.1)

# Save for MATLAB
emfbase.to_csv('datasets/emfbase1.csv')
