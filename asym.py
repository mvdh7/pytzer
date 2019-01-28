import numpy as np
import pytzer as pz
from autograd import elementwise_grad as egrad

I = np.vstack(np.genfromtxt('print_coeffs/Asymmetric comparison.csv',
                            delimiter=',',
                            skip_header=3,
                            usecols=0))

T = np.full_like(I,298.15)

Etheta = np.full((np.size(I),2),np.nan)

EthetaGrad = egrad(pz.model.etheta,argnum=1)

cf = pz.cfdicts.CoefficientDictionary()

# Fix Aosm at 0.3915 for testing
cf.dh['Aosm'] = lambda T: [np.full_like(T,0.3915), None]

# Start with P75 Eq. (47)
cf.jfunc = pz.jfuncs.P75_eq47

Etheta[  : 9,0] = pz.model.etheta(T[  : 9],I[  : 9],1,2,cf).ravel()
Etheta[  : 9,1] =      EthetaGrad(T[  : 9],I[  : 9],1,2,cf).ravel()

Etheta[12:21,0] = pz.model.etheta(T[12:21],I[12:21],1,3,cf).ravel()
Etheta[12:21,1] =      EthetaGrad(T[12:21],I[12:21],1,3,cf).ravel()

Etheta[24:33,0] = pz.model.etheta(T[24:33],I[24:33],2,3,cf).ravel()
Etheta[24:33,1] =      EthetaGrad(T[24:33],I[24:33],2,3,cf).ravel()

# Switch to Harvie
cf.jfunc = pz.jfuncs.Harvie

Etheta[36:45,0] = pz.model.etheta(T[36:45],I[36:45],1,2,cf).ravel()
Etheta[36:45,1] =      EthetaGrad(T[36:45],I[36:45],1,2,cf).ravel()

Etheta[48:57,0] = pz.model.etheta(T[48:57],I[48:57],1,3,cf).ravel()
Etheta[48:57,1] =      EthetaGrad(T[48:57],I[48:57],1,3,cf).ravel()

Etheta[60:  ,0] = pz.model.etheta(T[60:  ],I[60:  ],2,3,cf).ravel()
Etheta[60:  ,1] =      EthetaGrad(T[60:  ],I[60:  ],2,3,cf).ravel()

np.savetxt('print_coeffs/Asymm MPH.csv',
           Etheta,
           delimiter=',')
