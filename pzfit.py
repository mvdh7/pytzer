from autograd import numpy as np
from scipy import optimize
import pytzer as pz

m = np.array([.1,.5,1.,2.,3.,4.,5.,6.], dtype='float64')
T = np.full_like(m,298.15, dtype='float64')

nC = np.float_(1)
nA = np.float_(1)

mCmA = np.array([m*nC,m*nA]).transpose()

zC = np.float_(+1)
zA = np.float_(-1)

b0 = np.float_(0.0779)
b1 = np.float_(0.2689)
b2 = 0
C0 = np.float_(0.000932) / 2
C1 = 0

alph1 = np.float_(2.0)
alph2 = -9
omega = -9

nC = np.float_(1)
nA = np.float_(1)

ww = np.full_like(T,1.)

tgex = pz.fitting.Gex_MX(mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega)
tlaf = pz.fitting.ln_acf(mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega)
tacf = pz.fitting.acf   (mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega)
tosm = pz.fitting.osm   (mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega)
tacw = pz.miami.osm2aw  (mCmA,tosm)
tmaf = pz.fitting.acf_MX(mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega,nC,nA)

# Do fitting!
topt = optimize.least_squares(lambda bC: 
    pz.fitting.acf_MX(mCmA,zC,zA,T,bC[0],bC[1],b2,bC[2],C1,
                      alph1,alph2,omega,nC,nA) - tmaf,
    np.float_([0,0,0]), jac='3-point')

b0_fit = topt['x'][0]
b1_fit = topt['x'][1]
C0_fit = topt['x'][2]

# Get covariance matrix
mse = np.mean((pz.fitting.acf_MX(mCmA,zC,zA,T,b0_fit,b1_fit,b2,C0_fit,C1,
                                 alph1,alph2,omega,nC,nA) - tmaf)**2)
hess = topt.jac.transpose() @ topt.jac
bCmx = np.linalg.inv(hess) * mse
