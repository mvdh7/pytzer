from autograd import numpy as np
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
omega = np.float_(2.5)

nC = np.float_(1)
nA = np.float_(1)

ww = np.full_like(T,1.)

tgex = pz.fitting.Gex_MX(mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega)
tlaf = pz.fitting.ln_acf(mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega)
tacf = pz.fitting.acf   (mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega)
tosm = pz.fitting.osm   (mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega)
tacw = pz.model.osm2aw  (mCmA,tosm)
tmaf = pz.fitting.acfMX (mCmA,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega,nC,nA)

#b0f,b1f,b2f,C0f,C1f,bCmx,mse \
#    = pz.fitting.bC_acfMX(mCmA,zC,zA,T,alph1,alph2,omega,nC,nA,tmaf,'b0b1C0')
    
b0f,b1f,b2f,C0f,C1f,bCmx,mse \
    = pz.fitting.bC(mCmA,zC,zA,T,alph1,alph2,omega,nC,nA,tosm,
                    'b0b1C0','osm')
    
