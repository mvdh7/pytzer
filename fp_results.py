import numpy as np
import pytzer as pz
from scipy.io import savemat

cf = pz.cdicts.MPH

ele = 'NaCl'
method = 'vpl'

fstem = ele + '_' + method

sqtot = np.vstack(np.linspace(0.001,np.sqrt(pz.prop.solubility25[ele]),100))
tot   = sqtot**2

_,zC,zA,nC,nA = pz.data.znu([ele])
mols  = np.concatenate((tot*nC,tot*nA),axis=1)
T     = np.full_like(tot,298.15)

bC = np.array([bC[0] for bC in \
               cf.bC['-'.join(pz.data.ele2ions([ele])[0])](T[0])[0:5]])

alph1 = np.float_(2)
alph2 = -9
omega = np.float_(2.5)

bC_cv = np.genfromtxt('E:/Dropbox/_UEA_MPH/fort-pitzer/results/' \
                      + fstem + '_summary.res', skip_header=18, max_rows=4)

bC_cv = np.insert(bC_cv,2,0, axis=0)
bC_cv = np.insert(bC_cv,2,0, axis=1)

osm_sim, Uosm_sim = pz.fitting.ppg_osm(mols,zC,zA,T,bC,bC_cv,alph1,alph2,omega)

savemat('splines/fp_' + fstem + '.mat',{'tot'     : tot     ,
                                        'osm_sim' : osm_sim ,
                                        'Uosm_sim': Uosm_sim})
