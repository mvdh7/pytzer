# Run as
#>> python simloop_fpd.py <Uele> <Ureps>
# where <Uele>  is the electrolyte to analyse
#       <Ureps> is the number of Monte-Carlo simulations to execute
# Requires: datasets/fpd.xlsx
#           pickles/simpar_fpd.pkl


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

#argv = ['','KCl','10']

# Get input args
Uele  =     argv[1]
Ureps = int(argv[2])

# Load raw datasets
datapath = 'datasets/'
fpdbase,mols,ions,T = pz.data.fpd(datapath)

# Select electrolytes for analysis
fpdbase,mols,ions,T = pz.data.subset_ele(fpdbase,mols,ions,T,
                                         np.array([Uele]))

# Exclude smoothed datasets
S = fpdbase.smooth == 0
fpdbase = fpdbase[S]
mols    = mols   [S]

# Create initial electrolytes pivot table
fpde = pd.pivot_table(fpdbase,
                      values  = ['m'  ],
                      index   = ['ele'],
                      aggfunc = [np.min,np.max,len])

# Load outputs from simpytz_fpd.py
with open('pickles/simpar_fpd.pkl','rb') as f:
    _,fpderr_rdm,fpderr_sys = pickle.load(f)

# Prepare model cdict
cf = pz.cdicts.MPH
eles = fpdbase.ele
cf.add_zeros(fpdbase.ele)

# Extract metadata from fpdbase
tot  = pd2vs(fpdbase.m  )
srcs = pd2vs(fpdbase.src)
_,zC,zA,nC,nA = pz.data.znu(fpde.index)

# Identify which coefficients to fit
wbC = {'NaCl' : 'b0b1C0C1',
       'KCl'  : 'b0b1C0'  }
which_bCs = wbC[Uele]

fpdbase['t25'] = 298.15
T1    = pd2vs(fpdbase.t25)

nCvec = pd2vs(fpdbase.nC)
nAvec = pd2vs(fpdbase.nA)

# Prepare for simulation
Eions = pz.data.ele2ions(np.array([Uele]))[0]

fpdbase['fpd_calc'] = pz.tconv.tot2fpd(tot,Eions,nC,nA,cf)

# Calculate osmotic coefficient etc.
fpdbase['osm'] = pz.tconv.fpd2osm(mols,pd2vs(fpdbase.fpd))
fpdbase['osm25'] = pz.tconv.osm2osm(tot,nCvec,nAvec,ions,
                                    273.15-pd2vs(fpdbase.fpd),T1,T1,
                                    cf,pd2vs(fpdbase.osm))
fpdbase['osm25_calc'] = pz.model.osm(mols,ions,T1,cf)
    
#%% Simulate new datasets

# Set up for fitting
alph1 = np.float_(2)
alph2 = -9
omega = np.float_(2.5)
fpd_calc = pd2vs(fpdbase.fpd_calc)

# Define weights for fitting
#weights = np.ones(np.size(T1)) # uniform
#weights = np.sqrt(tot) # sqrt of molality
# ... based on random errors in each dataset:
weights = np.full_like(tot,1, dtype='float64')
for src in np.unique(srcs):
    SL = srcs == src
    weights[SL] = 1 / np.sqrt(np.sum(fpderr_rdm[Uele][src]**2))
weights = weights

# Do the fit to the original dataset
b0dir,b1dir,b2dir,C0dir,C1dir,bCdir_cv,mseo \
    = pz.fitting.bC(mols,zC,zA,T1,alph1,alph2,omega,nC,nA,pd2vs(fpdbase.osm25),
                    weights,which_bCs,'osm')
bCdir = np.hstack((b0dir,b1dir,b2dir,C0dir,C1dir))

## Check understanding of MSE calculation
#mseo_dir = np.mean(((pd2vs(fpdbase.osm25) - pz.fitting.osm(mols,zC,zA,T1,
#                                            b0dir,b1dir,b2dir,C0dir,C1dir,
#                                            alph1,alph2,omega)) * weights)**2)
#mse_dir  = np.mean((pd2vs(fpdbase.osm25 - fpdbase.osm25_calc) * weights)**2)

# Define fitting function
def Eopt(rseed=None):

    # Seed random numbers
    np.random.seed(rseed)

    # Simulate new FPD dataset
    Ufpd = pz.sim.fpd(tot,pd2vs(fpdbase.fpd_calc),
                      srcs,Uele,fpderr_rdm,fpderr_sys)
    
    # Convert FPD to osmotic coefficient
    Uosm = pz.tconv.fpd2osm(mols,Ufpd)
    
    # Convert osmotic coefficient to 298.15 K
    Uosm25 = pz.tconv.osm2osm(tot,nCvec,nAvec,ions,
                              273.15 - Ufpd,T1,T1,
                              cf,Uosm)

    # Solve for Pitzer model coefficients
    b0,b1,b2,C0,C1,_,_ \
        = pz.fitting.bC(mols,zC,zA,T1,alph1,alph2,omega,nC,nA,Uosm25,
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
    
    acfMX_sim   = np.vstack(
            ppg_acfMX(mols,zC,zA,T,bCsim,alph1,alph2,omega,nC,nA)[0])
    JacfMX_sim  = fx_JacfMX(mols,zC,zA,T,bCsim,
                            alph1,alph2,omega,nC,nA).squeeze()
    UacfMX_sim  = np.vstack(np.diag(
            JacfMX_sim @ bCsim_cv @ JacfMX_sim.transpose()))
    
    acfMX_dir  = np.vstack(
            ppg_acfMX(mols,zC,zA,T,bCdir,alph1,alph2,omega,nC,nA)[0])
    JacfMX_dir = fx_JacfMX(mols,zC,zA,T,bCdir,
                           alph1,alph2,omega,nC,nA).squeeze()
    UacfMX_dir = np.vstack(np.diag(
            JacfMX_dir @ bCdir_cv @ JacfMX_dir.transpose()))

    # Pickle/save results...
    fstem = 'pickles/simloop_fpd_bC_' + Uele + '_' + str(Ureps)
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
                             'UacfMX_dir': UacfMX_dir})
