from autograd import numpy as np
import pandas as pd
import pickle
import pytzer as pz
from pytzer.misc import pd2vs
from scipy import optimize

# Define test and reference electrolytes
tst = 'KCl'
ref = 'NaCl'

# Define cdict
cf = pz.cdicts.MPH
cf.add_zeros(np.array([tst,ref]))

# Load datasets and simpar results
trtxt = 't' + tst + '_r' + ref

isobase = pd.read_csv('pickles/simpar_iso_isobase_' + trtxt + '.csv')

with open('pickles/simpar_iso_all.pkl','rb') as f:
    isoerr_sys,isoerr_rdm = pickle.load(f)

ttot_exp    = pd2vs(isobase[tst])
rtot_exp    = pd2vs(isobase[ref])
rosm25_calc = pd2vs(isobase['osm_calc_' + ref])
srcs        = pd2vs(isobase.src)
T           = pd2vs(isobase.t)

# Get electrolyte info
_,tzC,tzA,tnC,tnA = pz.data.znu([tst])
_,_  ,_  ,rnC,rnA = pz.data.znu([tst])
tions = pz.data.ele2ions([tst])[0]

# Solve for expected test molality & osmotic coefficient
def tot2osm(tot,nC,nA,ions,T,cf):
    mols = np.concatenate((tot*nC,tot*nA),axis=1)
    return pz.model.osm(mols,ions,T,cf)

ttot_calc = np.full_like(rtot_exp,np.nan)

for M in range(len(ttot_calc)):
    
    ttot_calc[M] = optimize.least_squares(lambda ttot: \
        rtot_exp[M]*rosm25_calc[M] - ttot*tot2osm(np.vstack([ttot]),tnC,tnA,
            tions,np.vstack([T[M]]),cf).ravel(), 1.)['x']

tmols_calc = np.concatenate((ttot_calc*tnC,ttot_calc*tnA),axis=1)
tosm25_calc = pz.model.osm(tmols_calc,tions,T,cf)

tmols_exp = np.concatenate((ttot_exp*tnC,ttot_exp*tnA),axis=1)
rmols_exp = np.concatenate((rtot_exp*rnC,rtot_exp*rnA),axis=1)
tosm25_exp = pz.experi.osm(tmols_exp,rmols_exp,rosm25_calc)

## Test simulation function
#isobase['osm25_sim'] = pz.sim.iso(ttot_calc,tosm25_calc,srcs,tst,ref,
#                                  isoerr_sys,isoerr_rdm)
#isobase['dosm25_sim'] = isobase.osm25_sim - isobase['osm_calc_' + tst]
#isobase.to_csv('pickles/isobase_test.csv')
#
## Get simulated molalities
#ttot_sim = pz.experi.osm2tot(osm,nu,molsR,osmR)

# Do the fit to the original dataset
weights = np.full_like(T,1.)
_,_,_,_,_,talph1,talph2,tomega,_ = cf.bC['-'.join(tions)](T)
which_bCs = 'b0b1C0'

b0dir,b1dir,b2dir,C0dir,C1dir,bCdir_cv,mseo \
    = pz.fitting.bC(tmols_exp,tzC,tzA,T,talph1,talph2,tomega,tnC,tnA,
                    tosm25_exp,weights,which_bCs,'osm')
bCdir = np.hstack((b0dir,b1dir,b2dir,C0dir,C1dir))

#%% Define fitting function
def Eopt(rseed=None):

    # Seed random numbers
    np.random.seed(rseed)

    # Simulate new isopiestic equilibrium dataset
    Uosm = pz.sim.iso(ttot_calc,tosm25_calc,srcs,tst,ref,isoerr_sys,isoerr_rdm)

    # Solve for Pitzer model coefficients
    b0,b1,b2,C0,C1,_,_ \
        = pz.fitting.bC(tmols_calc,tzC,tzA,T,talph1,talph2,tomega,tnC,tnA,Uosm,
                        weights,which_bCs,'osm')

    return b0,b1,b2,C0,C1

