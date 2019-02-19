import pytzer as pz
import numpy as np
from scipy.io import savemat
from scipy.integrate import quad
from scipy.optimize import least_squares

sqI = np.vstack(np.arange(0.001,4,0.001))
I = sqI**2
T = np.full_like(I,298.15)

cf = pz.cfdicts.CoefficientDictionary()
cf.dh['Aosm'] = pz.coeffs.Aosm_MarChemSpec
cf.jfunc = pz.jfuncs.Harvie

x12 = pz.model.xij(T,I,1,2,cf)
x13 = pz.model.xij(T,I,1,3,cf)
x23 = pz.model.xij(T,I,2,3,cf)

J_Harvie   = pz.jfuncs.Harvie  (x23)
J_P75_eq46 = pz.jfuncs.P75_eq46(x23)
J_P75_eq47 = pz.jfuncs.P75_eq47(x23)

#%% Fit functions like P75
L = np.logical_and(0 <= np.sqrt(x23), np.sqrt(x23) <= 100)
x = x23[L]
J = J_Harvie[L]

# Using form of P75 Eq. 47
mfitfx = lambda C,x: x/(C[0] + C[1]*x**-C[2] * np.exp(-C[3]*x**C[4]))
mfit = least_squares(lambda C: mfitfx(C,x) - J, [4,4.581,0.7237,0.0120,0.528])
J_fit = mfitfx(mfit['x'],x23)

# Using form of P75 Eq. 47 but turbo-charged
mfitfx2 = lambda C,x: x \
    / (C[0] + C[1]*x**-C[2] * np.exp(-C[3]*x**C[4]) \
       + C[5]*x + C[6]*np.log(x))
mfit2 = least_squares(lambda C: mfitfx2(C,x) - J,
                      [4,4.581,0.7237,0.0120,0.528,0,0])
J_fit2 = mfitfx2(mfit2['x'],x23)

#%% Do numerical integral
def fJquad(x):
    
    # P91 Chapter 3 Eq. (B-12) [p123]
    q = lambda x, y: -(x/y) * np.exp(-y)
    
    J = np.full_like(x,np.nan)
    
    for i,xi in enumerate(x):
    
        # P91 Chapter 3 Eq. (B-13) [p123]
        J[i] = quad(lambda y: \
            (1 + q(xi,y) + q(xi,y)**2 / 2 - np.exp(q(xi,y))) * y**2,
            0,np.inf)[0] / xi
    
    return J

J_quad = fJquad(x23)

#%% Save results
savemat('testfiles/jfunk.mat',
        {'T': T,
         'I': I,
         'x12': x12,
         'x13': x13,
         'x23': x23,
         'J_Harvie'  : J_Harvie  ,
         'J_P75_eq46': J_P75_eq46,
         'J_P75_eq47': J_P75_eq47,
         'J_quad'    : J_quad    ,
         'J_fit'     : J_fit     ,
         'J_fit2'    : J_fit2    })
