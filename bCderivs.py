from autograd import numpy            as np
from autograd import elementwise_grad as egrad
from scipy.io import savemat
import pytzer as pz

# Select electrolyte
ele = 'NaCl'

if ele == 'NaCl':
    tmax = 6.25
    bCfunc = pz.coeffs.bC_Na_Cl_A92ii
    
elif ele == 'KCl':
    tmax = 3.5
    bCfunc = pz.coeffs.bC_K_Cl_ZD17

# Set up inputs
sqtot = np.vstack(np.linspace(0.01,np.sqrt(tmax),100))
tot   = sqtot**2
mols  = np.concatenate((tot,tot),axis=1)

zC = np.float_(+1)
zA = np.float_(-1)

T = np.full_like(tot,298.15)

b0,b1,b2,C0,C1,alph1,alph2,omega,_ = bCfunc(T)

osm = pz.fitting.osm(mols,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega)

osm0 = pz.fitting.osm(mols,zC,zA,T,0,0,0,0,0,-9,-9,-9)

# Get derivatives of osm wrt. bC
dosm_db0 = egrad(pz.fitting.osm, argnum=4)
dosm_db1 = egrad(pz.fitting.osm, argnum=5)
dosm_dC0 = egrad(pz.fitting.osm, argnum=7)
dosm_dC1 = egrad(pz.fitting.osm, argnum=8)

db0 = dosm_db0(mols,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega)
db1 = dosm_db1(mols,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega)
dC0 = dosm_dC0(mols,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega)
dC1 = dosm_dC1(mols,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega)

# Find influence of alph1 and omega
db1_an = dosm_db1(mols,zC,zA,T,b0,b1,b2,C0,C1,alph1 * 1.1,alph2,omega)
db1_au = dosm_db1(mols,zC,zA,T,b0,b1,b2,C0,C1,alph1 * 0.9,alph2,omega)
dC1_on = dosm_dC1(mols,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega * 1.1)
dC1_ou = dosm_dC1(mols,zC,zA,T,b0,b1,b2,C0,C1,alph1,alph2,omega * 0.9)

savemat('pickles/bCderivs_' + ele + '.mat',
        {'tot'   : tot,
         'osm'   : osm,
         'osm0'  : osm0,
         'db0'   : db0,
         'db1'   : db1,
         'dC0'   : dC0,
         'dC1'   : dC1,
         'db1_an': db1_an,
         'db1_au': db1_au,
         'dC1_on': dC1_on,
         'dC1_ou': dC1_ou,
         'bCs'   : (b0[0],b1[0],C0[0],C1[0])})
