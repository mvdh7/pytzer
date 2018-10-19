# Run as
#>> python simloop_fpd_osm25.py <Uele> <Ureps>
# where <Uele>  is the electrolyte to analyse
#       <Ureps> is the number of Monte-Carlo simulations to execute
# Requires: pickles/simpar_fpd_osm25.pkl

import numpy  as np
import pickle
import pytzer as pz
pd2vs = pz.misc.pd2vs
from multiprocessing import Pool
from scipy.io        import savemat
from sys             import argv
from time            import time

#argv = ['','KCl','10']

# Get input args
Uele  =     argv[1]
Ureps = int(argv[2])

# Load raw datasets
with open('pickles/simpar_fpd_osm25.pkl','rb') as f:
    fpdbase,mols,ions,T,fpderr_rdm,fpderr_sys = pickle.load(f)

# Select electrolyte for analysis
fpdbase,mols,ions,T = pz.data.subset_ele(fpdbase,mols,ions,T,
                                         np.array([Uele]))

#if Uele == 'CaCl2':
#    L = np.logical_or(fpdbase.m <= 3.5,fpdbase.src != 'OBS90')
#    fpdbase = fpdbase[L]
#    mols    = mols   [L]
#    T       = T      [L]

# Prepare model cdict
cf = pz.cdicts.MPH
eles = fpdbase.ele
cf.add_zeros(fpdbase.ele)

# Extract metadata from fpdbase
tot  = pd2vs(fpdbase.m  )
srcs = pd2vs(fpdbase.src)
zC   = pd2vs(fpdbase.zC )
zA   = pd2vs(fpdbase.zA )
nC   = pd2vs(fpdbase.nC )
nA   = pd2vs(fpdbase.nA )

# Identify which coefficients to fit
wbC = {'NaCl' : 'b0b1C0C1',
       'KCl'  : 'b0b1C0C1',
       'CaCl2': 'b0b1C0C1'}
which_bCs = wbC[Uele]

T25   = pd2vs(fpdbase.t25)

nCvec = pd2vs(fpdbase.nC)
nAvec = pd2vs(fpdbase.nA)

# Prepare for simulation
Eions = pz.data.ele2ions(np.array([Uele]))[0]
    
# Set up for fitting
alph1 = np.float_(2)
alph2 = -9
omega = np.float_(2.5)
osm25_calc = pd2vs(fpdbase.osm25_calc)

# Define weights for fitting
weights = np.ones(np.size(T25)) # uniform
#weights = np.sqrt(tot) # sqrt of molality
# ... based on random errors in each dataset:
#weights = np.full_like(tot,1, dtype='float64')
#for src in np.unique(srcs):
#    SL = srcs == src
#    weights[SL] = 1 / np.sqrt(np.sum(fpderr_rdm[Uele][src]**2))
#weights = weights

# Do the fit to the original dataset
b0dir,b1dir,b2dir,C0dir,C1dir,bCdir_cv,mseo \
    = pz.fitting.bC(mols,zC,zA,T25,alph1,alph2,omega,nC,nA,
                    pd2vs(fpdbase.osm25_meas),weights,which_bCs,'osm')
bCdir = np.hstack((b0dir,b1dir,b2dir,C0dir,C1dir))

## Check understanding of MSE calculation
#mseo_dir = np.mean(((pd2vs(fpdbase.osm25) - pz.fitting.osm(mols,zC,zA,T1,
#                                            b0dir,b1dir,b2dir,C0dir,C1dir,
#                                            alph1,alph2,omega)) * weights)**2)
#mse_dir  = np.mean((pd2vs(fpdbase.osm25 - fpdbase.osm25_calc) * weights)**2)

##%% Test simulation function
#fpdbase['osm25_sim'] = pz.sim.fpd_osm25(tot,osm25_calc,srcs,Uele,
#                                        fpderr_rdm,fpderr_sys)
#fpdbase['dosm25'] = fpdbase.osm25_sim - fpdbase.osm25_calc
#fpdbase.to_csv('pickles/fpdbase_sim_osm25.csv')

#%% Define fitting function
def Eopt(rseed=None):

    # Seed random numbers
    np.random.seed(rseed)

    # Simulate new osm25 dataset
    Uosm25 = pz.sim.fpd_osm25(tot,osm25_calc,srcs,Uele,fpderr_rdm,fpderr_sys)

    # Solve for Pitzer model coefficients
    b0,b1,b2,C0,C1,_,_ \
        = pz.fitting.bC(mols,zC,zA,T25,alph1,alph2,omega,nC,nA,Uosm25,
                        weights,which_bCs,'osm')

    return b0,b1,b2,C0,C1

b0,b1,b2,C0,C1 = Eopt()

#%% Multiprocessing loop
if __name__ == '__main__':
    
    print('multiprocessing %s...' % Uele, end='\r')
    
    # Set initial random seed (for reproducibility)
    np.random.seed(295)

    # Generate seeds for random number generator
    rseeds = np.random.randint(0,2**32,size=Ureps,dtype='int64')

    Xtstart = time() # begin timer - multiprocessing

    with Pool() as pool:
        bCpool = pool.map(Eopt,rseeds)
        pool.close()
        pool.join()

    # Reformat pool output
    b0 = np.float_([bCpool[X][0] for X in range(Ureps)])
    b1 = np.float_([bCpool[X][1] for X in range(Ureps)])
    b2 = np.float_([bCpool[X][2] for X in range(Ureps)])
    C0 = np.float_([bCpool[X][3] for X in range(Ureps)])
    C1 = np.float_([bCpool[X][4] for X in range(Ureps)])
    bC = np.vstack((b0,b1,b2,C0,C1))
    bCsim = np.mean(bC, axis=1)
    bCsim_cv   = np.cov(bC)

    # Calculate and print processing time
    Xtend = time() # end timer - multiprocessing
    print('multiprocessing %s: %d reps in %.1f seconds' \
        % (Uele,Ureps,(Xtend - Xtstart)))

    # Calculate activity coefficient and propagate error with sim. results
    sqtot = np.vstack(np.linspace(0.001,
                                  np.sqrt(pz.prop.solubility25[Uele]),
                                  100))
    tot   = sqtot**2
    mols  = np.concatenate((tot,tot),axis=1)
    T     = np.full_like(tot,298.15)
    _,zC,zA,nC,nA = pz.data.znu([Uele])
    
    # Get example propagation splines
    acfMX_sim, UacfMX_sim = pz.fitting.ppg_acfMX(mols,zC,zA,T,bCsim,bCsim_cv,
                                                 alph1,alph2,omega,nC,nA)
    
    acfMX_dir, UacfMX_dir = pz.fitting.ppg_acfMX(mols,zC,zA,T,bCdir,bCdir_cv,
                                                 alph1,alph2,omega,nC,nA)
    
    osm_sim, Uosm_sim = pz.fitting.ppg_osm(mols,zC,zA,T,bCsim,bCsim_cv,
                                           alph1,alph2,omega)
    
    osm_dir, Uosm_dir = pz.fitting.ppg_osm(mols,zC,zA,T,bCdir,bCdir_cv,
                                           alph1,alph2,omega)

    # Pickle/save results...
    fstem = 'pickles/simloop_fpd_osm25_bC_' + Uele + '_' + str(Ureps)
    # ...for Python:
    with open(fstem + '.pkl','wb') as f:
        pickle.dump((bCsim,bCsim_cv,bCdir,bCdir_cv,Uele,Ureps),f)
    # ...for MATLAB:
    savemat(fstem + '.mat', {'bCsim'     : bCsim     ,
                             'bCsim_cv'  : bCsim_cv  ,
                             'bCdir'     : bCdir     ,
                             'bCdir_cv'  : bCdir_cv  ,
                             'Uele'      : Uele      ,
                             'Ureps'     : Ureps     ,
                             'tot'       : tot       ,
                             'acfMX_sim' : acfMX_sim ,
                             'acfMX_dir' : acfMX_dir ,
                             'UacfMX_sim': UacfMX_sim,
                             'UacfMX_dir': UacfMX_dir,
                             'osm_sim'   : osm_sim   ,
                             'Uosm_sim'  : Uosm_sim  ,
                             'osm_dir'   : osm_dir   ,
                             'Uosm_dir'  : Uosm_dir  })
