from autograd import numpy as np
import pytzer as pz
from scipy.io import savemat
import pandas as pd

# Load raw isopiestic dataset and cut to only 298.15 K data
isobase = pz.data.iso('datasets/')
isobase = isobase.loc[isobase.t == 298.15]

# Select only rows where there is a KCl and NaCl measurement
isopair = ['KCl','NaCl']
isobase,mols0,mols1,ions0,ions1,T = pz.data.get_isopair(isobase,isopair)

# Calculate reference model stuff
cf = pz.cdicts.MPH
cf.bC['K-Cl'] = pz.coeffs.bC_K_Cl_A99

isobase['osm_ref_' + isopair[0]] = pz.model.osm(mols0,ions0,T,cf)
isobase['osm_ref_' + isopair[1]] = pz.model.osm(mols1,ions1,T,cf)

isobase['aw_ref_' + isopair[0]] = pz.model.osm2aw(mols0,pz.misc.pd2vs(
                                  isobase['osm_ref_' + isopair[0]]))
isobase['aw_ref_' + isopair[1]] = pz.model.osm2aw(mols1,pz.misc.pd2vs(
                                  isobase['osm_ref_' + isopair[1]]))

# Calculate osmotic coefficients from the measurements
isobase['osm_meas_' + isopair[0]] = isobase['osm_ref_' + isopair[1]] \
                                  * np.sum(mols1,axis=1) / np.sum(mols0,axis=1)

isobase['osm_meas_' + isopair[1]] = isobase['osm_ref_' + isopair[0]] \
                                  * np.sum(mols0,axis=1) / np.sum(mols1,axis=1)


# Save isobase for MATLAB plotting
isobase.to_csv('pickles/data_iso.csv')
savemat('pickles/data_iso.mat',{'mols0': mols0,
                                'mols1': mols1})

# Create sources pivot table
isoe = pd.pivot_table(isobase,
                      values  = isopair,
                      index   = ['src'],
                      aggfunc = [np.min,np.max,len])
isoe.to_csv('pickles/data_isoe.csv')
