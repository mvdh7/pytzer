import pytzer as pz
from autograd import elementwise_grad as egrad
from autograd import numpy as np

# Choose function to evaluate
bCfunc = pz.coeffs.bC_Ca_Cl_GM89

# Define individual coefficient functions
def fx_b0(T):
    return bCfunc(T)[0]
fx_db0_dT = egrad(fx_b0)

def fx_b1(T):
    return bCfunc(T)[1]
fx_db1_dT = egrad(fx_b1)

def fx_C0(T):
    return bCfunc(T)[3]
fx_dC0_dT = egrad(fx_C0)

def fx_C1(T):
    return bCfunc(T)[4]
fx_dC1_dT = egrad(fx_C1)

# Calculate values at specific temperature
T = np.vstack([298.15,298.15])

b0  = fx_b0(T)
db0 = fx_db0_dT(T)
b1  = fx_b1(T)
db1 = fx_db1_dT(T)
C0  = fx_C0(T)
dC0 = fx_dC0_dT(T)
C1  = fx_C1(T)
dC1 = fx_dC1_dT(T)

Cphi  = C0  * 2 * np.sqrt(2)
dCphi = dC0 * 2 * np.sqrt(2)
