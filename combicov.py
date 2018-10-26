from autograd import jacobian as jac
from autograd import numpy as np
import pickle
import pytzer as pz
from scipy.io import savemat
from scipy import optimize

ele = 'NaCl'
ions = pz.data.ele2ions([ele])[0]

with open('pickles/simloop_fpd_osm25_bC_' + ele + '_1000.pkl','rb') as f:
    bCfpd,bCfpd_cv,_,_,_,_ = pickle.load(f)
    
with open('pickles/simloop_vpl_bC_' + ele + '_1000.pkl','rb') as f:
    bCvpl,bCvpl_cv,_,_,_,_ = pickle.load(f)
    
# ===== Fit the matrix ========================================================

# Evaluate combined uncertainty
sqtot = np.vstack(np.linspace(0.001,
                              np.sqrt(pz.prop.solubility25[ele]),
                              100))
tot   = sqtot**2

_,zC,zA,nC,nA = pz.data.znu([ele])
mols  = np.concatenate((tot*nC,tot*nA),axis=1)
T     = np.full_like(tot,298.15)

alph1 = np.float_(2)
alph2 = -9
omega = np.float_(2.5)

# Get reference model coefficients
cf = pz.cdicts.MPH
b0,b1,b2,C0,C1,_,_,_,_ = cf.bC['-'.join(ions)](T[0])
bC = np.concatenate((b0,b1,b2,C0,C1))

# Get propagation splines
Ufpd = pz.fitting.ppg_osm(mols,zC,zA,T,bC,bCfpd_cv,
                          alph1,alph2,omega)[1]

Uvpl = pz.fitting.ppg_osm(mols,zC,zA,T,bC,bCvpl_cv,
                          alph1,alph2,omega)[1]

Ucombi = Uvpl * Ufpd / (Uvpl + Ufpd)

#%% Construct matrix propagation function for optimisation

# Convert between vector and matrix format for VCM
def vec2mx(vec):
    mx = np.full((5,5),np.nan)
    for i in range(5):
        for j in range(i,5):
            vj = int(j + 4.5*i - 0.5*i**2)
            mx[i,j] = vec[vj]
            mx[j,i] = vec[vj]
    return mx

def mx2vec(mx):
    vec = np.full(15,np.nan)
    for i in range(5):
        for j in range(i,5):
            vj = int(j + 4.5*i - 0.5*i**2)
            vec[vj] = mx[i,j]
    return vec

# Convert between 4- and 5-coefficient vector
def v42v5(v4):
    return np.insert(v4,(2,5,7,7,7),0)

def v52v4(v5):
    return v5[[0,1,3,4,5,7,8,12,13,14]]

def v4cv2v5(v4cv,vdiag):
    return v42v5(np.insert(v4cv,(0,3,5,6),vdiag))

# Define propagation functions for fitting
def ppg_mx(mols,zC,zA,T,bC,bC_cv_vec,alph1,alph2,omega):
    # fit all vars and covars
    bC_cv = vec2mx(bC_cv_vec)
    return pz.fitting.ppg_osm(mols,zC,zA,T,bC,bC_cv,alph1,alph2,omega)[1]

def ppg_mx_bC4(mols,zC,zA,T,bC,bC_cv_vec,alph1,alph2,omega):
    # fit all vars and covars except b2, which is set to zero
    bC_cv = vec2mx(v42v5(bC_cv_vec))
    return pz.fitting.ppg_osm(mols,zC,zA,T,bC,bC_cv,alph1,alph2,omega)[1]

def ppg_mx_bC4_diag(mols,zC,zA,T,bC,bC_cv_vec,vdiag,alph1,alph2,omega):
    # fit covars only, except b2, which is set to zero
    bC_cv = vec2mx(v4cv2v5(bC_cv_vec,vdiag))
    return pz.fitting.ppg_osm(mols,zC,zA,T,bC,bC_cv,alph1,alph2,omega)[1]

# Calculate values for VCM diagonal
vdiag = (np.diag(bCvpl_cv) + np.diag(bCfpd_cv))[[0,1,3,4]]

#test = optimize.least_squares(lambda bC_cv_vec: \
#    (ppg_mx_bC4_diag(mols,zC,zA,T,bC,bC_cv_vec,vdiag,alph1,alph2,omega) \
#     - Ucombi).ravel(), np.zeros(6), method='lm')

test = optimize.least_squares(lambda bC_cv_vec: \
    (ppg_mx_bC4(mols,zC,zA,T,bC,bC_cv_vec,alph1,alph2,omega) \
     - Ucombi).ravel(), np.zeros(10), method='trf')#, 
#     bounds=([0,-np.inf,-np.inf,-np.inf,0,-np.inf,-np.inf,0,-np.inf,0],
#             [np.inf,np.inf,np.inf,np.inf,np.inf,
#              np.inf,np.inf,np.inf,np.inf,np.inf]))

#%% Save for plotting
#testmx = vec2mx(v4cv2v5(test['x'],vdiag))
testmx = vec2mx(v42v5(test['x']))

#testmx = np.identity(5) * vdiag

Utest = pz.fitting.ppg_osm(mols,zC,zA,T,bC,testmx,
                           alph1,alph2,omega)[1]

UtestACF = pz.fitting.ppg_acfMX(mols,zC,zA,T,bC,testmx,
                                alph1,alph2,omega,nC,nA)[1]

savemat('pickles/combicov2_' + ele + '.mat',{'tot'   : tot ,
                                             'Ufpd'  : Ufpd,
                                             'Uvpl'  : Uvpl,
                                             'Ucombi': Ucombi,
                                             'Utest' : Utest ,
                                             'UtestACF':UtestACF})

#%% === Original approach =====================================================
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
