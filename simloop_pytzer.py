# Run as
# > python simloop_pytzer.py <Uele> <Ureps>
# where <Uele>  is the electrolyte to analyse
#       <Ureps> is the number of Monte-Carlo simulations to execute

import numpy  as np
import pandas as pd
import pickle
import pytzer as pz
pd2vs = pz.misc.pd2vs
from multiprocessing import Pool
from scipy.io        import savemat
from sys             import argv
from time            import time

argv = ['','NaCl','10']

# Get input args
Uele  =     argv[1]
Ureps = int(argv[2])

# Load raw datasets
datapath = 'datasets/'
fpdbase,mols,ions = pz.data.fpd(datapath)

# Select electrolytes for analysis
fpdbase,mols,ions = pz.data.subset_ele(fpdbase,mols,ions,
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

# Prepare model cdict
cf = pz.cdicts.cdict()
cf.add_zeros(fpdbase.ele)
cf.dh['Aosm'] = pz.coeffs.Aosm_MPH
cf.dh['AH'  ] = pz.coeffs.AH_MPH
cf.bC['Na-Cl'] = pz.coeffs.bC_Na_Cl_A92ii
cf.bC['K-Cl' ] = pz.coeffs.bC_K_Cl_A99

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
for E,ele in enumerate(fpde.index):

    EL = fpdbase.ele == ele
    Eions = pz.data.ele2ions(np.array([ele]))[0]
    
    fpdbase['fpd_calc'] = pz.tconv.tot2fpd(tot[EL],Eions,nC[E],nA[E],cf)

# Calculate osmotic coefficient etc.
fpdbase['osm'] = pz.tconv.fpd2osm(mols,pd2vs(fpdbase.fpd))
fpdbase['osm25'] = pz.tconv.osm2osm(tot,nCvec,nAvec,ions,
                                    273.15-pd2vs(fpdbase.fpd),T1,T1,
                                    cf,pd2vs(fpdbase.osm))
fpdbase['osm25_calc'] = pz.model.osm(mols,ions,T1,cf)

# Load outputs from simpytz_fpd.py
with open('pickles/simpytz_fpd.pkl','rb') as f:
    _,fpderr_rdm,fpderr_sys = pickle.load(f)
    
#%% Simulate new datasets

# Set up for fitting
alph1 = np.float_(2.5)
alph2 = -9
omega = np.float_(2)
fpd_calc = pd2vs(fpdbase.fpd_calc)

# Define weights for fitting
#weights = np.ones(np.size(T1)) # uniform
#weights = np.sqrt(tot) # sqrt of molality
# ... based on random errors in each dataset:
weights = np.full_like(tot,1, dtype='float64')
for src in np.unique(srcs):
    SL = srcs == src
    weights[SL] = 1 / np.sqrt(np.sum(fpderr_rdm[ele][src]**2))
weights = weights

# Do the fit to the original dataset
b0o,b1o,b2o,C0o,C1o,bCo_cv,mseo \
    = pz.fitting.bC(mols,zC,zA,T1,alph1,alph2,omega,nC,nA,pd2vs(fpdbase.osm25),
                    weights,which_bCs,'osm')
bCo = np.hstack((b0o,b1o,b2o,C0o,C1o))

# Check understanding of MSE calculation
mseo_dir = np.mean(((pd2vs(fpdbase.osm25) - pz.fitting.osm(mols,zC,zA,T1,
                                            b0o,b1o,b2o,C0o,C1o,
                                            alph1,alph2,omega)) * weights)**2)
mse_dir  = np.mean((pd2vs(fpdbase.osm25 - fpdbase.osm25_calc) * weights)**2)

# Define fitting function
def Eopt(rseed=None):

    # Seed random numbers
    np.random.seed(rseed)

    # Simulate new FPD dataset
    Ufpd = pz.sim.fpd(tot,pd2vs(fpdbase.fpd_calc),
                      srcs,ele,fpderr_rdm,fpderr_sys)
    
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
    bC_mean = np.mean(bC, axis=1)
    bC_cv   = np.cov(bC)

    # Calculate and print processing time
    Xtend = time() # end timer - multiprocessing
    print('multiprocessing %s: %d reps in %.2f seconds' \
        % (Uele,Ureps,(Xtend - Xtstart)))

    # Pickle/save results
    fstem = 'pickles/simloop_pytzer_bC_' + Uele + '_' + str(Ureps)
    with open(fstem + '.pkl','wb') as f:
        pickle.dump((bC_mean,bC_cv,bCo,bCo_cv,Uele,Ureps),f)
    savemat(fstem + '.mat', {'bC_mean' : bC_mean,
                             'bC_cv'   : bC_cv  ,
                             'bCo'     : bCo    ,
                             'bCo_cv'  : bCo_cv ,
                             'Uele'    : Uele   ,
                             'Ureps'   : Ureps  })
