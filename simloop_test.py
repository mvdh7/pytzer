import numpy as np
import pandas as pd
import pickle
from scipy.io import savemat
import pytzer as pz
pd2vs = pz.misc.pd2vs

# Load raw datasets
datapath = 'datasets/'
fpdbase,mols,ions = pz.data.fpd(datapath)

# Select electrolytes for analysis
fpdbase,mols,ions = pz.data.subset_ele(fpdbase,mols,ions,
                                       np.array(['NaCl']))

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
eles = fpdbase.ele
cf.add_zeros(fpdbase.ele)

cf.bC['Na-Cl'] = pz.coeffs.bC_Na_Cl_A92ii
cf.bC['K-Cl' ] = pz.coeffs.bC_K_Cl_ZD17
cf.bC['Ca-Cl'] = pz.coeffs.bC_Ca_Cl_GM89
cf.bC['Mg-Cl'] = pz.coeffs.bC_Mg_Cl_PP87i
cf.dh['Aosm']  = pz.coeffs.Aosm_MPH
cf.dh['AH']    = pz.coeffs.AH_MPH
cf.jfunc = pz.jfuncs.P75_eq47

# Extract metadata from fpdbase
tot  = pd2vs(fpdbase.m  )
srcs = pd2vs(fpdbase.src)
_,zC,zA,nC,nA = pz.data.znu(fpde.index)

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
sele = 'NaCl'
Ureps = int(20)

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
    b0,b1,_,C0,C1,_,_ \
        = pz.fitting.bC(mols,zC,zA,T1,alph1,alph2,omega,nC,nA,Uosm25,
                        weights,'b0b1C0C1','osm')

    return Ufpd,Uosm25,b0,b1,C0,C1

fpd_sim   = np.full((np.size(tot),Ureps),np.nan)
osm25_sim = np.full((np.size(tot),Ureps),np.nan)
b0 = np.full(Ureps,np.nan)
b1 = np.full(Ureps,np.nan)
C0 = np.full(Ureps,np.nan)
C1 = np.full(Ureps,np.nan)

sqtot_fitted = np.vstack(np.linspace(0.001,2.5,100))
tot_fitted   = sqtot_fitted**2
mols_fitted  = np.concatenate((tot_fitted,tot_fitted), axis=1)
osm25_fitted = np.full((np.size(tot_fitted),Ureps),np.nan)
T1_fitted    = np.full_like(tot_fitted,298.15, dtype='float64')

for U in range(Ureps):
    print(U)
    Ufpd_sim,Uosm25_sim,Ub0,Ub1,UC0,UC1 = Eopt()
    fpd_sim[:,U]   = Ufpd_sim.ravel()
    osm25_sim[:,U] = Uosm25_sim.ravel()
    b0[U] = Ub0
    b1[U] = Ub1
    C0[U] = UC0
    C1[U] = UC1
    osm25_fitted[:,U] = pz.fitting.osm(mols_fitted,zC,zA,
                                       T1_fitted,b0[U],b1[U],0,C0[U],C1[U],
                                       alph1,-9,omega).ravel()
    
osm25_fitted_calc = pz.model.osm(mols_fitted,ions,T1_fitted,cf)
    
# Save results for MATLAB
fpdbase.to_csv('pickles/simloop_test.csv')
savemat('pickles/simloop_test.mat',{'fpd_sim'           : fpd_sim,
                                    'osm25_sim'         : osm25_sim,
                                    'tot_fitted'        : tot_fitted,
                                    'osm25_fitted'      : osm25_fitted,
                                    'osm25_fitted_calc' : osm25_fitted_calc})

## Quick results viz
#from matplotlib import pyplot as plt
#fig,ax = plt.subplots(1,1)
#ax.scatter(fpdbase.m,fpd_sim-fpd_calc)
