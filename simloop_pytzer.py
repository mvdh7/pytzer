from autograd import numpy as np
#from sys import path#, argv
#if 'E:\Dropbox\_UEA_MPH\pitzer-spritzer\python' not in path:
#    path.append('E:\\Dropbox\\_UEA_MPH\\pitzer-spritzer\\python')
import pickle, time
from multiprocessing import Pool
import pytzer as pz

# Get electrolyte to analyse and number of repeats from user input
#ele   =     argv[1]
#Ureps = int(argv[2])
ele = 'NaCl'
Ureps = 100

# Get electrolyte-specific information
fcs = {'NaCl' : 'b0b1C0C1',
        'KCl'  : 'b0b1C0'  }

aos = {'NaCl' : np.float_([-9, 2,-9,-9, 2.5]),
        'KCl'  : np.float_([-9, 2,-9,-9,-9  ])}

fc = fcs[ele]
ao = aos[ele]

# Load FPD dataset
with open('pickles/simpytz_fpd.pkl','rb') as f:
    fpdbase,fpdp,err_cfs_both,fpd_sys_std = pickle.load(f)

# Preprocess dataset
L = fpdbase.ele == ele
Eargs = [np.vstack(fpdbase.loc[L,var].values) \
         for var in ['m', 'zC', 'zA', 'nC', 'nA', 'src', 'ele', 'fpd', 't25']]
tot,zC,zA,nC,nA,srcs,eles,fpd,t25 = Eargs
fpdbase = fpdbase[L]

# Get pytzer fitting function inputs
mCmA = np.concatenate([tot*nC,tot*nA], axis=1)
T = np.full_like(tot,298.15, dtype='float64')
alph1 = ao[1]
alph2 = ao[2]
omega = ao[4]

# Prepare model cdict
cf = pz.cdicts.cdict()
cf.bC['Na-Cl'] = pz.coeffs.bC_Na_Cl_A92ii
cf.bC['K-Cl' ] = pz.coeffs.bC_K_Cl_GM89
cf.theta['K-Na'] = pz.coeffs.theta_zero
cf.psi['K-Na-Cl'] = pz.coeffs.psi_zero
cf.dh['Aosm']  = pz.coeffs.Aosm_M88
cf.dh['AH']    = pz.coeffs.AH_MPH

#%% Define optimisation function
def Eopt(rseed=None):

    # Seed random numbers
    np.random.seed(rseed)

    # Simulate uncertainties
    Uosm = np.full_like(T,np.nan)
    
    for src in fpdp.loc[ele].index:
        SL = srcs == src
        
        Uosm[SL] = pz.sim.fpd(fpdbase[SL],ele,src,cf,
                              err_cfs_both,fpd_sys_std).ravel()

    b0,b1,b2,C0,C1,bCmx,mse \
        = pz.fitting.bC(mCmA,zC,zA,T,alph1,alph2,omega,nC,nA,Uosm,fc,'osm')

    return b0,b1,b2,C0,C1

b0,b1,b2,C0,C1 = Eopt()

#%% Multiprocessing loop
if __name__ == '__main__':

    # Set initial random seed (for reproducibility)
    np.random.seed(295)

    # Generate seeds for random number generator
    rseeds = np.random.randint(0,2**32,size=Ureps,dtype='int64')

    Xtstart = time.time() # begin timer - multiprocessing

    with Pool() as pool:
        bCpool = pool.map(Eopt,rseeds)
        pool.close()
        pool.join()

    # Format pool output
    bCpool = np.array([bCpool[X] for X in range(Ureps)])
    bCpool_cv = np.cov(bCpool,rowvar=False)

    Xtend = time.time() # end timer - multiprocessing

    # Calculate and print processing time
    print('multiprocessing %s: %d reps in %.2f seconds' \
        % (ele,Ureps,(Xtend - Xtstart)))

    # Pickle results
    with open('E:\Dropbox\_UEA_MPH\pytzer\pickles' \
              + '\simloop_pytzer_bC_' + ele + '.pkl','wb') as f:
        pickle.dump((bCpool_cv,ele,Ureps),f)
