import pytzer as pz
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize

# Prepare model cdict
cf = pz.cdicts.cdict()
cf.bC['Na-Cl'] = pz.coeffs.bC_Na_Cl_A92ii
cf.bC['K-Cl' ] = pz.coeffs.bC_K_Cl_GM89
cf.theta['K-Na'] = pz.coeffs.theta_zero
cf.psi['K-Na-Cl'] = pz.coeffs.psi_zero
cf.dh['Aosm']  = pz.coeffs.Aosm_MPH
cf.dh['AH']    = pz.coeffs.AH_MPH

# Calculate expected FPD
mols = np.full((100,2),5.05368, dtype='float64')
ions = np.array(['Na','Cl'])

T = np.vstack(np.linspace(245,275, num=100, dtype='float64'))
fpd = 273.15 - T

osmM = pz.model.osm(mols,ions,T,cf)
osmF = pz.tconv.fpd2osm(mols,fpd)

# Get ideal FPD
xmols = np.array([mols[0,:]])
xfpd = optimize.least_squares(lambda fpd: \
    (  pz.tconv.fpd2osm(xmols,fpd) \
     - pz.model.osm(xmols,ions,273.15-fpd,cf)).ravel(),
                              5.)['x'][0]

xfpd2 = pz.tconv.tot2fpd_X(np.vstack(mols[:,0]),ions,1.,1.,cf)

# Plot results
fig,ax = plt.subplots(2,1)

ax[0].plot(fpd,osmM, c='r')
ax[0].plot(fpd,osmF, c='b')
ax[0].grid(alpha=0.5)

ax[1].plot(fpd,osmF-osmM, c='m')
ax[1].grid(alpha=0.5)

#ax[2].plot(fpd,pz.coeffs.Aosm_CRP94(T)[0])
#ax[2].plot(fpd,pz.coeffs.Aosm_MPH(T)[0])
