from autograd import jacobian as jac
from autograd import numpy    as np
from scipy.io  import savemat
import pickle
import pytzer as pz

# Open simulation results
with open('pickles/simloop_pytzer_bC_KCl_9600.pkl','rb') as f:
    bC_mean,bC_cv,bCo,bCo_cv,Uele,Ureps = pickle.load(f)

# Calculate activity coefficient
sqtot = np.vstack(np.linspace(0.001,1.81,100))
tot   = sqtot**2
mols  = np.concatenate((tot,tot),axis=1)
#ions  = np.array(['K','Cl'])
T     = np.full_like(tot,298.15)

bCo_cv = np.insert(bCo_cv,(2,3),0, axis=1)
bCo_cv = np.insert(bCo_cv,(2,3),0, axis=0)

#cf = pz.cdicts.cdict()
#cf.bC['K-Cl'] = pz.coeffs.bC_K_Cl_A99
#cf.dh['Aosm'] = pz.coeffs.Aosm_MPH
#
#osm25 = pz.model.osm(mols,ions,T,cf)

# Define propagation equation
def ppg_acf(mCmA,zC,zA,T,bC,alph1,alph2,omega,nC,nA):
    
    b0 = bC[0]
    b1 = bC[1]
    b2 = bC[2]
    C0 = bC[3]
    C1 = bC[4]
    
    return pz.fitting.acfMX(mCmA,zC,zA,T,b0,b1,b2,C0,C1,
                            alph1,alph2,omega,nC,nA)

fx_Jacf = jac(ppg_acf,argnum=4)

zC = np.float_(+1)
zA = np.float_(-1)
nC = np.float_(1)
nA = np.float_(1)
alph1 = np.float_(2.5)
alph2 = -9
omega = -9

acf   = ppg_acf(mols,zC,zA,T,bC_mean,alph1,alph2,omega,nC,nA)
Jacf  = fx_Jacf(mols,zC,zA,T,bC_mean,alph1,alph2,omega,nC,nA).squeeze()
Uacf  = np.vstack(np.diag(Jacf @ bC_cv @ Jacf.transpose()))

acfo  = ppg_acf(mols,zC,zA,T,bCo,alph1,alph2,omega,nC,nA)
Jacfo = fx_Jacf(mols,zC,zA,T,bCo,alph1,alph2,omega,nC,nA).squeeze()
Uacfo = np.vstack(np.diag(Jacfo @ bCo_cv @ Jacfo.transpose()))

savemat('pickles/simloop_res.mat',{'tot'  : tot,
                                   'acf'  : np.vstack(acf),
                                   'acfo' : np.vstack(acfo),
                                   'Uacf' : Uacf,
                                   'Uacfo': Uacfo})
