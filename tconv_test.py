from autograd import numpy as np
from autograd import elementwise_grad as egrad
import pytzer as pz

T = np.float_(298.15)

m = np.float_(0.5)

I = m
nu = np.float_(2)

cf = pz.cdicts.M88

fL_from_fG_autograd = -pz.model.dfG_T_dT(T,I,cf) * T**2 / m

cf.dh['AH'] = pz.coeffs.AH_MPH
fL = pz.model.fL(T,I,cf,nu)

fL_from_fG_with_AH  = -pz.model.dfG_T_dT(T,I,cf) * T**2 / m

dAosm = egrad(lambda T: pz.coeffs.Aosm_M88(T)[0])

dAosm_dT_autograd = dAosm(T)
dAosm_dT_from_AH  = cf.dh['AH'](T)[0] / (4 * pz.constants.R * T**2)

# is I still differentiated correctly?
