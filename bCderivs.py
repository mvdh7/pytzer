from autograd import numpy            as np
from autograd import elementwise_grad as egrad
from scipy.io import savemat
import pytzer as pz

sqtot = np.vstack(np.linspace(0.01,np.sqrt(6.25),100))
tot   = sqtot**2
mols  = np.concatenate((tot,tot),axis=1)

zC = np.float_(+1)
zA = np.float_(-1)

T = np.full_like(tot,298.15)

b0,b1,b2,C0,C1,alph1,alph2,omega,_ = pz.coeffs.bC_Na_Cl_A92ii(T)

osm = pz.fitting.osm(mols,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega)

dosm_db0 = egrad(pz.fitting.osm, argnum=4)
dosm_db1 = egrad(pz.fitting.osm, argnum=5)
dosm_dC0 = egrad(pz.fitting.osm, argnum=7)
dosm_dC1 = egrad(pz.fitting.osm, argnum=8)

db0 = dosm_db0(mols,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega)
db1 = dosm_db1(mols,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega)
dC0 = dosm_dC0(mols,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega)
dC1 = dosm_dC1(mols,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega)

savemat('pickles/dosm_dbC.mat',{'tot': tot,
                                'osm': osm,
                                'db0': db0,
                                'db1': db1,
                                'dC0': dC0,
                                'dC1': dC1})
