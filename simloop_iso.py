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

rtot        = pd2vs(isobase[ref])
rosm25_calc = pd2vs(isobase['osm_calc_' + ref])
srcs        = pd2vs(isobase.src)
T           = pd2vs(isobase.t)

# Get electrolyte info
_,_,_,tnC,tnA = pz.data.znu([tst])
tions = pz.data.ele2ions([tst])[0]

#%% Solve for expected test molality & osmotic coefficient
def tot2osm(tot,nC,nA,ions,T,cf):
    mols = np.concatenate((tot*nC,tot*nA),axis=1)
    return pz.model.osm(mols,ions,T,cf)

ttot_calc = np.full_like(rtot,np.nan)

for M in range(len(ttot_calc)):
    
    ttot_calc[M] = optimize.least_squares(lambda ttot: \
        rtot[M]*rosm25_calc[M] - ttot*tot2osm(np.vstack([ttot]),tnC,tnA,
            tions,np.vstack([T[M]]),cf).ravel(), 1.)['x']
    
tmols = np.concatenate((ttot_calc*tnC,ttot_calc*tnA),axis=1)
tosm25_calc = pz.model.osm(tmols,tions,T,cf)

#%%
#from scipy import optimize

#test = pz.fitting.osm(rmols,rzC,rzA,T,rbC)

#test = optimize.least_squares(lambda tot:
#    pz.fitting.osm(mols,zC,zA,T,*bC) * nu * tot \
#    - osmR[M] * np.sum(molsR[M]), 1.)['x']

## Test simulation function
#isobase['osm25_sim'] = pz.sim.iso(rtot,tosm25_calc,srcs,tst,ref,
#                                  isoerr_sys,isoerr_rdm)
#isobase['dosm25_sim'] = isobase.osm25_sim - isobase['osm_calc_' + tst]
#isobase.to_csv('pickles/isobase_test.csv')

## Get simulated molalities
#ttot_sim = pz.experi.osm2tot(osm,nu,molsR,osmR)
