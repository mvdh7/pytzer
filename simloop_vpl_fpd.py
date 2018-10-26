# Run as
#>> python simloop_vpl_fpd.py <Uele> <Ureps>
# where <Uele>  is the electrolyte to analyse
#       <Ureps> is the number of Monte-Carlo simulations to execute
# Requires: pickles/simpar_vpl.pkl       from simpar_vpl.py
#           pickles/simpar_fpd_osm25.pkl from simpar_fpd_osm25.py

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

# ===== VAPOUR PRESSURE LOWERING ==============================================

# Load raw datasets
with open('pickles/simpar_vpl.pkl','rb') as f:
    vplbase,vplmols,vplions,vplT,vplerr_sys,vplerr_rdm = pickle.load(f)

# Select electrolyte for analysis
vplbase,vplmols,vplions,vplT = pz.data.subset_ele(vplbase,vplmols,vplions,vplT,
                                                  np.array([Uele]))

# Extract metadata from vplbase
vpltot  = pd2vs(vplbase.m  )
vplsrcs = pd2vs(vplbase.src)
vplzC   = pd2vs(vplbase.zC )
vplzA   = pd2vs(vplbase.zA )
vplnC   = pd2vs(vplbase.nC )
vplnA   = pd2vs(vplbase.nA )

# Set up for fitting
vplosm_calc = pd2vs(vplbase.osm_calc)
vplosm_meas = pd2vs(vplbase.osm_meas)
vplweights  = np.ones(np.size(vplT))

# ===== FREEZING POINT DEPRESSION =============================================

# Load raw datasets
with open('pickles/simpar_fpd_osm25.pkl','rb') as f:
    fpdbase,fpdmols,fpdions,fpdT,fpderr_sys,fpderr_rdm = pickle.load(f)

# Select electrolyte for analysis
fpdbase,fpdmols,fpdions,fpdT = pz.data.subset_ele(fpdbase,fpdmols,fpdions,fpdT,
                                                  np.array([Uele]))

# Extract metadata from fpdbase
fpdtot  = pd2vs(fpdbase.m  )
fpdsrcs = pd2vs(fpdbase.src)
fpdzC   = pd2vs(fpdbase.zC )
fpdzA   = pd2vs(fpdbase.zA )
fpdnC   = pd2vs(fpdbase.nC )
fpdnA   = pd2vs(fpdbase.nA )
fpdT25  = pd2vs(fpdbase.t25)

# Set up for fitting
fpdosm25_calc = pd2vs(fpdbase.osm25_calc)
fpdosm25_meas = pd2vs(fpdbase.osm25_meas)
fpdweights    = np.ones(np.size(vplT))

# ===== BOTH TOGETHER =========================================================

# Combine vectors
mols     = np.concatenate((vplmols    ,fpdmols      ))
zC       = np.concatenate((vplzC      ,fpdzC        ))
zA       = np.concatenate((vplzA      ,fpdzA        ))
T        = np.concatenate((vplT       ,fpdT25       ))
nC       = np.concatenate((vplnC      ,fpdnC        ))
nA       = np.concatenate((vplnA      ,fpdnA        ))
osm_meas = np.concatenate((vplosm_meas,fpdosm25_meas))
weights  = np.concatenate((vplweights ,fpdweights   ))

# Get ions
Eions = pz.data.ele2ions(np.array([Uele]))[0]

# Identify which coefficients to fit
wbC = {'NaCl' : 'b0b1C0C1',
       'KCl'  : 'b0b1C0'  ,
       'CaCl2': 'b0b1C0C1'}
which_bCs = wbC[Uele]

# Prepare model cdict
cf = pz.cdicts.MPH
eles = vplbase.ele
cf.add_zeros(vplbase.ele)

# Set alphas and omega
alph1 = np.float_(2)
alph2 = -9
omega = np.float_(2.5)

# Do the fit to the original datasets
b0dir,b1dir,b2dir,C0dir,C1dir,bCdir_cv,mseo \
    = pz.fitting.bC(mols,zC,zA,T,alph1,alph2,omega,nC,nA,
                    osm_meas,weights,which_bCs,'osm')
bCdir = np.hstack((b0dir,b1dir,b2dir,C0dir,C1dir))

# Define fitting function
def Eopt(rseed=None):

    # Seed random numbers
    np.random.seed(rseed)

    # Simulate new datasets
    Uvpl = pz.sim.vpl(vpltot,vplosm_calc,vplsrcs,Uele,vplerr_sys,vplerr_rdm)
    Ufpd = pz.sim.fpd_osm25(fpdtot,fpdosm25_calc,fpdsrcs,Uele,
                            fpderr_sys,fpderr_rdm)
    Uosm = np.concatenate((Uvpl,Ufpd))

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
    fstem = 'pickles/simloop_vpl_fpd_bC_' + Uele + '_' + str(Ureps)
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
