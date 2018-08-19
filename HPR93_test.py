import pytzer as pz
from autograd import numpy as np
from matplotlib import pyplot as plt

cf = pz.cdicts.MIAMI

sqtot = np.vstack(np.linspace(0.01,2,100))
tot = sqtot**2

mols = np.concatenate((tot,tot),axis=1)
ions = np.array(['Na','SO4'])

Ts = np.float_([273.15,298.15,323.15,348.15,373.15])

_,ax = plt.subplots(1,2, figsize=(10,4))

# Figure for comparison with HPR93 Fig. 3
# DOES NOT WORK!
for Ti in Ts:

    T = np.full_like(tot,Ti)
    
    osm = pz.model.osm(mols,ions,T,cf)
    
    ax[0].plot(sqtot,osm, label=str(Ti))
    
    ax[0].set_xlim([0,2])
    ax[0].set_ylim([0.5,1])

    ax[0].legend()

    ln_acfs = pz.model.ln_acfs(mols,ions,T,cf)
    acfMX = np.exp(np.vstack(
            pz.model.ln_acf2ln_acf_MX(ln_acfs[:,0],ln_acfs[:,1],2.,1.)))

    ax[1].plot(sqtot,acfMX, label=str(Ti))
    
    ax[1].set_xlim([0.5,2])
    ax[1].set_ylim([0.09,0.2])

    ax[1].legend()
