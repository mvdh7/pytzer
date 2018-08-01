# Run as
#>> python simloop_vpl.py <Uele> <Ureps>
# where <Uele>  is the electrolyte to analyse
#       <Ureps> is the number of Monte-Carlo simulations to execute
# Requires: datasets/vpl.xlsx
#           pickles/simpar_vpl.pkl

import numpy  as np
import pandas as pd
import pickle
import pytzer as pz
pd2vs = pz.misc.pd2vs
from autograd        import jacobian as jac
from multiprocessing import Pool
from scipy.io        import savemat
from sys             import argv
from time            import time

#argv = ['','NaCl','10']

# Get input args
Uele  =     argv[1]
Ureps = int(argv[2])

# Load raw datasets
datapath = 'datasets/'
vplbase,mols,ions,T = pz.data.vpl(datapath)

# Select electrolytes for analysis
vplbase,mols,ions,T = pz.data.subset_ele(vplbase,mols,ions,T,
                                         np.array([Uele]))

# Exclude datasets with T > 373.15 (i.e. my D-H functions are out of range)
Tx = vplbase.t <= 373.15
# Also, take only data at 298.15 K
Tx = np.logical_and(Tx,vplbase.t == 298.15)
vplbase = vplbase[Tx]
mols    = mols   [Tx]
T       = T      [Tx]

# Create initial electrolytes pivot table
vple = pd.pivot_table(vplbase,
                      values  = ['m'  ],
                      index   = ['ele'],
                      aggfunc = [np.min,np.max,len])

# Load outputs from simpytz_vpl.py
with open('pickles/simpar_vpl.pkl','rb') as f:
    _,vplerr_rdm,vplerr_sys = pickle.load(f)

# Prepare model cdict
cf = pz.cdicts.MPH
eles = vplbase.ele
cf.add_zeros(vplbase.ele)

# Extract metadata from vplbase
tot  = pd2vs(vplbase.m  )
srcs = pd2vs(vplbase.src)
_,zC,zA,nC,nA = pz.data.znu(vple.index)

# Identify which coefficients to fit
wbC = {'NaCl' : 'b0b1C0C1',
       'KCl'  : 'b0b1C0C1'}
which_bCs = wbC[Uele]

vplbase['t25'] = 298.15
T1    = pd2vs(vplbase.t25)

nCvec = pd2vs(vplbase.nC)
nAvec = pd2vs(vplbase.nA)

# Prepare for simulation
Eions = pz.data.ele2ions(np.array([Uele]))[0]

# Calculate osmotic coefficient etc.
vplbase['osm_calc'] = pz.model.osm(mols,ions,T,cf)
vplbase['osm_meas'] = pz.model.aw2osm(mols,pd2vs(vplbase.aw))
vplbase['osm25_meas'] = pz.tconv.osm2osm(tot,nCvec,nAvec,Eions,
                                         T,T1,T1,cf,pd2vs(vplbase.osm_meas))
vplbase['osm25_calc'] = pz.model.osm(mols,ions,T1,cf)
    
#%% Simulate new datasets

# Set up for fitting
alph1 = np.float_(2)
alph2 = -9
omega = np.float_(2.5)
osm_calc = pd2vs(vplbase.osm_calc)

# Define weights for fitting
#weights = np.ones(np.size(T1)) # uniform
#weights = np.sqrt(tot) # sqrt of molality
# ... based on random errors in each dataset:
weights = np.full_like(tot,1, dtype='float64')
for src in np.unique(srcs):
    SL = srcs == src
#    weights[SL] = 1 / np.sqrt(np.sum(vplerr_rdm[Uele][src]**2))
    Smax = np.max(tot[SL])
    Smin = np.min(tot[SL])
    weights[SL] = (vplerr_rdm[Uele][src][0] * (Smax - Smin) \
        - vplerr_rdm[Uele][src][1] * (np.exp(-Smax) - np.exp(-Smin))) \
           / (Smax - Smin)
weights = 1 / weights

# Do the fit to the original dataset
b0dir,b1dir,b2dir,C0dir,C1dir,bCdir_cv,mseo \
    = pz.fitting.bC(mols,zC,zA,T1,alph1,alph2,omega,nC,nA,
                    pd2vs(vplbase.osm_meas),weights,which_bCs,'osm')
bCdir = np.hstack((b0dir,b1dir,b2dir,C0dir,C1dir))

## Check understanding of MSE calculation
#mseo_dir = np.mean(((pd2vs(vplbase.osm25) - pz.fitting.osm(mols,zC,zA,T1,
#                                            b0dir,b1dir,b2dir,C0dir,C1dir,
#                                            alph1,alph2,omega)) * weights)**2)
#mse_dir  = np.mean((pd2vs(vplbase.osm25 - vplbase.osm25_calc) * weights)**2)

#%% Define fitting function
def Eopt(rseed=None):

    # Seed random numbers
    np.random.seed(rseed)

    # Simulate new VPL dataset
    Uosm = pz.sim.vpl(tot,pd2vs(vplbase.osm_calc),
                      srcs,Uele,vplerr_rdm,vplerr_sys)
    
    # Solve for Pitzer model coefficients
    b0,b1,b2,C0,C1,_,_ \
        = pz.fitting.bC(mols,zC,zA,T,alph1,alph2,omega,nC,nA,Uosm,
                        weights,which_bCs,'osm')

    return b0,b1,b2,C0,C1

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
    sqtot = np.vstack(np.linspace(0.001,np.sqrt(np.max(tot)),100))
    tot   = sqtot**2
    mols  = np.concatenate((tot,tot),axis=1)
    T     = np.full_like(tot,298.15)
    
    # Get example propagation splines
    acfMX_sim, UacfMX_sim = pz.fitting.ppg_acfMX(mols,zC,zA,T,bCsim,bCsim_cv,
                                                 alph1,alph2,omega,nC,nA)
    
    acfMX_dir, UacfMX_dir = pz.fitting.ppg_acfMX(mols,zC,zA,T,bCdir,bCdir_cv,
                                                 alph1,alph2,omega,nC,nA)
    
    osm_sim, Uosm_sim = pz.fitting.ppg_osm(mols,zC,zA,T,bCsim,bCsim_cv,
                                           alph1,alph2,omega)

    # Pickle/save results...
    fstem = 'pickles/simloop_vpl_bC_' + Uele + '_' + str(Ureps)
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
                             'Uosm_sim'  : Uosm_sim  })
