from autograd import jacobian as jac
from autograd import numpy    as np
from scipy.io  import savemat
import pickle
import pytzer as pz

# Open simulation results
with open('pickles/simloop_pytzer_bC_NaCl_8.pkl','rb') as f:
    bCsim,bCsim_cv,bCdir,bCdir_cv,Uele,Ureps = pickle.load(f)

#%% Calculate activity coefficient
sqtot = np.vstack(np.linspace(0.001,1.81,100))
tot   = sqtot**2
mols  = np.concatenate((tot,tot),axis=1)
T     = np.full_like(tot,298.15)

# Define propagation equation
def ppg_acfMX(mCmA,zC,zA,T,bC,alph1,alph2,omega,nC,nA):
    
    b0 = bC[0]
    b1 = bC[1]
    b2 = bC[2]
    C0 = bC[3]
    C1 = bC[4]
    
    return pz.fitting.acfMX(mCmA,zC,zA,T,b0,b1,b2,C0,C1,
                            alph1,alph2,omega,nC,nA)

fx_JacfMX = jac(ppg_acfMX,argnum=4)

zC = np.float_(+1)
zA = np.float_(-1)
nC = np.float_(1)
nA = np.float_(1)
alph1 = np.float_(2.5)
alph2 = -9
omega = -9

acfMX_sim   = ppg_acfMX(mols,zC,zA,T,bCsim,alph1,alph2,omega,nC,nA)
JacfMX_sim  = fx_JacfMX(mols,zC,zA,T,bCsim,alph1,alph2,omega,nC,nA).squeeze()
UacfMX_sim  = np.vstack(np.diag(JacfMX_sim @ bCsim_cv @ JacfMX_sim.transpose()))

acfMX_dir  = ppg_acfMX(mols,zC,zA,T,bCdir,alph1,alph2,omega,nC,nA)
JacfMX_dir = fx_JacfMX(mols,zC,zA,T,bCdir,alph1,alph2,omega,nC,nA).squeeze()
UacfMX_dir = np.vstack(np.diag(JacfMX_dir @ bCdir_cv @ JacfMX_dir.transpose()))

savemat('pickles/simloop_res.mat',{'tot'       : tot                 ,
                                   'acfMX_sim' : np.vstack(acfMX_sim),
                                   'acfMX_dir' : np.vstack(acfMX_dir),
                                   'UacfMX_sim': UacfMX_sim          ,
                                   'UacfMX_dir': UacfMX_dir          })
