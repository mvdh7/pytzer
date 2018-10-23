from autograd import numpy as np
import pandas as pd
import pickle
import pytzer as pz
from pytzer.misc import pd2vs

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
_,_,_,rnC,rnA = pz.data.znu([ref])

tbC = cf.bC['K-Cl'](T)
rmols = np.concatenate((rtot*rnC,rtot*rnA),axis=1)

tosm25_calc = pz.experi.get_osm(tbC,T,rmols,rosm25_calc)

## Test simulation function
#isobase['osm25_sim'] = pz.sim.iso(rtot,tosm25_calc,srcs,tst,ref,
#                                  isoerr_sys,isoerr_rdm)
#isobase['dosm25_sim'] = isobase.osm25_sim - isobase['osm_calc_' + tst]
#isobase.to_csv('pickles/isobase_test.csv')

## Get simulated molalities
#ttot_sim = pz.experi.osm2tot(osm,nu,molsR,osmR)
