from autograd import jacobian as jac
from autograd import numpy as np
import pickle
import pytzer as pz
from scipy.io import savemat

ele = 'NaCl'

with open('pickles/simloop_fpd_osm25_bC_' + ele + '_100.pkl','rb') as f:
    bCfpd,bCfpd_cv,_,_,_,_ = pickle.load(f)
    
with open('pickles/simloop_vpl_bC_' + ele + '_1000.pkl','rb') as f:
    bCvpl,bCvpl_cv,_,_,_,_ = pickle.load(f)
    
def bCbest(bCs,bCvrs):
#    return (bCs[0:5]*bCvrs[5:10] + bCs[5:10]*bCvrs[0:5]) \
#        / (bCvrs[0:5] + bCvrs[5:10])
    b0 = np.array([(bCs[0]*bCvrs[5] + bCs[5]*bCvrs[0]) \
                   / (bCvrs[0] + bCvrs[5])])
    b1 = np.array([(bCs[1]*bCvrs[6] + bCs[6]*bCvrs[1]) \
                   / (bCvrs[1] + bCvrs[6])])
    b2 = np.array([0.])
#    b2 = np.array([(bCs[2]*bCvrs[7] + bCs[7]*bCvrs[2]) \
#                   / (bCvrs[2] + bCvrs[7])])
    C0 = np.array([(bCs[3]*bCvrs[8] + bCs[8]*bCvrs[3]) \
                   / (bCvrs[3] + bCvrs[8])])
    C1 = np.array([(bCs[4]*bCvrs[9] + bCs[9]*bCvrs[4]) \
                   / (bCvrs[4] + bCvrs[9])])
    
#    VM = np.float_(3.3)
#    FM = np.float_(8.5)
#    return (bCs[0:5]*FM + bCs[5:10]*VM) / (VM + FM)
    
    return np.concatenate((b0,b1,b2,C0,C1))
    
cf   = pz.cdicts.MPH
ions = pz.data.ele2ions([ele])[0]
#bCs  = np.array([cf.bC['-'.join(ions)](298.15)[X] for X in range(5)])
#bCs  = np.concatenate((bCs,bCs))

bCs     = np.concatenate((bCfpd,bCvpl))

bC0_cv  = np.zeros_like(bCfpd_cv)
xbCs_cv = np.concatenate((bCfpd_cv,bC0_cv),axis=1)
ybCs_cv = np.concatenate((bC0_cv,bCvpl_cv),axis=1)
bCs_cv  = np.concatenate((xbCs_cv,ybCs_cv),axis=0)
bCvrs   = np.diag(bCs_cv)

bC = bCbest(bCs,bCvrs)

bCbest_jac = jac(bCbest)(bCs,bCvrs)
bCbest_jac[np.isnan(bCbest_jac)] = 0

bC_cv = bCbest_jac @ bCs_cv @ bCbest_jac.transpose()

with open('pickles/combicov_' + ele + '.pkl','wb') as f:
    pickle.dump((bC_cv),f)

# Calculate osmotic coefficient and propagate error with sim. results
_,zC,zA,nC,nA = pz.data.znu([ele])
    
sqtot = np.vstack(np.linspace(0.001,
                              np.sqrt(pz.prop.solubility25[ele]),
                              100))
tot  = sqtot**2
mols = np.concatenate((tot*nC,tot*nA),axis=1)
T    = np.full_like(tot,298.15)

_,_,_,_,_,alph1,alph2,omega,_ = cf.bC['-'.join(ions)](T)

##%% Brute force attempt - no good
#bCsim_fpd = np.random.multivariate_normal(bCfpd,bCfpd_cv,int(1e5))
#bCsim_vpl = np.random.multivariate_normal(bCvpl,bCvpl_cv,int(1e5))
#
#bCsim_mean = (bCsim_fpd + bCsim_vpl) / 2
#bCsim_cv = np.cov(bCsim_mean, rowvar=False)

osm_best,Uosm_best = pz.fitting.ppg_osm(mols,zC,zA,T,bC,bC_cv,
                                        alph1,alph2,omega)

savemat('pickles/combicov_' + ele + '.mat',{ 'osm_best': osm_best ,
                                            'Uosm_best': Uosm_best})
