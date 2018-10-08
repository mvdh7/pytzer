from autograd import numpy as np
from autograd import elementwise_grad as egrad
import pytzer as pz
from scipy.io import savemat

# Define test conditions
tot  = np.vstack(np.linspace(0.01,6.25,500))
T    = np.full_like(tot,298.15)
vp0  = pz.misc.vp_H2O(T) # in kPa
vpX  = vp0 * 0.5

# Set up osmotic coefficient function and its derivatives
def fx_osm(tot,T,vpX):
    
    mols = np.concatenate((tot,tot),axis=1)
    vp0  = pz.misc.vp_H2O(T)
    aw  = vpX / vp0
    
    osm = pz.model.aw2osm(mols,aw)
    
    return osm

fx_dosm_dtot = egrad(fx_osm,argnum=0)
fx_dosm_dT   = egrad(fx_osm,argnum=1)
fx_dosm_dvpX = egrad(fx_osm,argnum=2)

# Define typical uncertainties
Utot = np.float_(0.01) # mol/kg
UT   = np.float_(0.1)  # K
UvpX = np.float_(1e-3) # kPa

# Evaluate osmotic coefficient and its derivatives
osm = fx_osm(tot,T,vpX)

dosm_dtot = fx_dosm_dtot(tot,T,vpX) * Utot
dosm_dT   = fx_dosm_dT  (tot,T,vpX) * UT
dosm_dvpX = fx_dosm_dvpX(tot,T,vpX) * UvpX

Uosm = np.sqrt(dosm_dtot**2 + dosm_dT**2 + dosm_dvpX**2)

pct_dosm_dtot = 100 * np.abs(dosm_dtot) / Uosm
pct_dosm_dT   = 100 * np.abs(dosm_dT  ) / Uosm
pct_dosm_dvpX = 100 * np.abs(dosm_dvpX) / Uosm

savemat('pickles/vpl_prop.mat',{'tot'      : tot,
                                'dosm_dT'  : dosm_dT,
                                'dosm_dtot': dosm_dtot,
                                'dosm_dvpX': dosm_dvpX})
