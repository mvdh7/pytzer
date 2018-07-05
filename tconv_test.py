from autograd import numpy as np
from autograd import elementwise_grad as egrad
import pytzer as pz

T = np.float_([298.15])

m = np.float_(6)

I = m
nu = np.float_(2)

cf = pz.cdicts.M88

fL_from_fG_autograd = -pz.model.dfG_T_dT(T,I,cf) * T**2 / m

cf.dh['AH'] = pz.coeffs.AH_MPH
fL = pz.model.fL(T,I,cf,nu)

fL_from_fG_with_AH  = -pz.model.dfG_T_dT(T,I,cf) * T**2 / m

dAosm = egrad(lambda T: pz.coeffs.Aosm_M88(T)[0])

dAosm_dT_autograd = dAosm(T)
dAosm_dT_from_AH  = cf.dh['AH'](T) / (4 * pz.constants.R * T**2)

# is I still differentiated correctly? - yes!
dfG_dI = egrad(pz.model.fG,argnum=1)

print(dfG_dI(T,I,cf))
print(pz.model.fG(T,I+1e-6,cf) - pz.model.fG(T,I,cf))

mols = np.array([[6.,6.]])
ions = np.array(['Na','Cl'])
tot  = np.vstack(mols[:,0])

dGex_T_dT = egrad(pz.model.Gex_nRT, argnum=2)

Lphi = -T**2 * dGex_T_dT(mols,ions,T,cf) * pz.constants.R / m

Lphi_pz = pz.model.Lapp(tot,1.,1.,ions,T,cf)
