from autograd import numpy as np
from autograd import elementwise_grad as egrad
import pytzer as pz

T = np.float_(298.15)
I = np.float_(0.5)

cf = pz.cdicts.M88
cf.dh['AH'] = pz.coeffs.AH_MPH

fG = pz.model.fG(T,I,cf)

dfG_T_dT = pz.model.dfG_T_dT(T,I,cf)
fL_deriv = dfG_T_dT * -T**2 / 0.5

fL = pz.model.fL(T,I,cf)

#

dAosm = egrad(lambda T: pz.coeffs.Aosm_M88(T)[0])

test = dAosm(T)
AH = cf.dh['AH'](T)[0] / (pz.constants.R * T)
