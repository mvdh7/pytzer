# Run as
#>> python simloop_vpl.py <Uele> <Ureps>
# where <Uele>  is the electrolyte to analyse
#       <Ureps> is the number of Monte-Carlo simulations to execute
# Requires: pickles/simpar_vpl.pkl

import numpy  as np
import pickle
import pytzer as pz
pd2vs = pz.misc.pd2vs
from multiprocessing import Pool
from scipy.io        import savemat
from sys             import argv
from time            import time

#argv = ['','CaCl2','10']

# Get input args
Uele  =     argv[1]
Ureps = int(argv[2])

# Load raw datasets
with open('pickles/simpar_vpl.pkl','rb') as f:
    vplbase,mols,ions,T,vplerr_sys,vplerr_rdm = pickle.load(f)

# Select electrolyte for analysis
vplbase,mols,Eions,T = pz.data.subset_ele(vplbase,mols,ions,T,
                                         np.array([Uele]))

# Prepare model cdict
cf = pz.cdicts.MPH
eles = vplbase.ele
cf.add_zeros(vplbase.ele)

# Extract metadata from vplbase
tot  = pd2vs(vplbase.m  )
srcs = pd2vs(vplbase.src)
_,zC,zA,nC,nA = pz.data.znu([Uele])

# Identify which coefficients to fit
wbC = {'NaCl' : 'b0b1C0C1',
       'KCl'  : 'b0b1C0C1',
       'CaCl2': 'b0b1C0C1'}
which_bCs = wbC[Uele]

# Set up for fitting
alph1 = np.float_(2)
alph2 = -9
omega = np.float_(2.5)
osm_calc = pd2vs(vplbase.osm_calc)

# Define weights for fitting
weights = np.ones(np.size(T)) # uniform
#weights = np.sqrt(tot) # sqrt of molality
# ... based on random errors in each dataset:
#weights_dir = np.full_like(tot,1, dtype='float64')
#for src in np.unique(srcs):
#    SL = srcs == src
#    weights_dir[SL] = pz.misc.rms((pd2vs(vplbase.osm_meas) \
#                                 - pd2vs(vplbase.osm_calc))[SL])
#weights_dir = weights_dir * np.sqrt(tot)
#weights_dir = 1 / weights_dir
weights_dir = weights

# Do the fit to the original dataset
b0dir,b1dir,b2dir,C0dir,C1dir,bCdir_cv,mseo \
    = pz.fitting.bC(mols,zC,zA,T,alph1,alph2,omega,nC,nA,
                    pd2vs(vplbase.osm_meas),weights_dir,which_bCs,'osm')
bCdir = np.hstack((b0dir,b1dir,b2dir,C0dir,C1dir))

## Check understanding of MSE calculation
#mseo_dir = np.mean(((pd2vs(vplbase.osm25) - pz.fitting.osm(mols,zC,zA,T1,
#                                            b0dir,b1dir,b2dir,C0dir,C1dir,
#                                            alph1,alph2,omega)) * weights)**2)
#mse_dir  = np.mean((pd2vs(vplbase.osm25 - vplbase.osm25_calc) * weights)**2)

# Define fitting function
def Eopt(rseed=None):

    # Seed random numbers
    np.random.seed(rseed)

    # Simulate new VPL dataset
    Uosm = pz.sim.vpl(tot,pd2vs(vplbase.osm_calc),
                      srcs,Uele,vplerr_sys,vplerr_rdm)

    # Solve for Pitzer model coefficients
    b0,b1,b2,C0,C1,_,_ \
        = pz.fitting.bC(mols,zC,zA,T,alph1,alph2,omega,nC,nA,Uosm,
                        weights,which_bCs,'osm')

    return b0,b1,b2,C0,C1

## Test dataset simulation
#Ureps_sim = 20
#Uosm_sim = np.full((np.size(T),Ureps_sim),np.nan)
#
#for U in range(Ureps_sim):
#    Uosm_sim[:,U] = pz.sim.vpl(tot,pd2vs(vplbase.osm_calc),
#                               srcs,Uele,vplerr_sys,vplerr_rdm).ravel()
#
#savemat('pickles/Uosm_sim_vpl_' + Uele + '.mat',{'Uosm_sim' : Uosm_sim})
#vplbase.to_csv('pickles/Uosm_sim_vpl_' + Uele + '.csv')

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
    
    _,zC,zA,nC,nA = pz.data.znu([Uele])
    mols  = np.concatenate((tot*nC,tot*nA),axis=1)
    T     = np.full_like(tot,298.15)

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
                             'Uosm_sim'  : Uosm_sim  ,
                             'osm_dir'   : osm_dir   ,
                             'Uosm_dir'  : Uosm_dir  })
