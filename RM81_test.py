import pytzer as pz
from autograd import numpy as np
from matplotlib import pyplot as plt

cf = pz.cdicts.MIAMI
cf.bC['Mg-SO4'] = pz.coeffs.bC_Mg_SO4_PP86ii

tot  = np.vstack([0.1,1,2,3,3.6176])
mols = np.concatenate((tot,tot),axis=1)
ions = np.array(['Mg','SO4'])
T    = np.full_like(tot,298.15)

osm = pz.model.osm(mols,ions,T,cf)
ln_acfs = pz.model.ln_acfs(mols,ions,T,cf)
ln_acfMX = pz.model.ln_acf2ln_acf_MX(ln_acfs[:,0],ln_acfs[:,1],2.,2.)
acfMX = np.exp(ln_acfMX)
