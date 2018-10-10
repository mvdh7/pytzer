from autograd import numpy as np
import pytzer as pz
from scipy.io import savemat
from scipy import optimize
import pandas as pd
pd2vs = pz.misc.pd2vs

# Need to invest a better sense of which electrolyte is the ref throughout

# Define test and ref electrolytes:
tst = 'KCl'
ref = 'NaCl'

# Load raw isopiestic dataset and cut to only 298.15 K
isobase = pz.data.iso('datasets/')
isobase = isobase.loc[isobase.t == 298.15]

# Select only rows where there is a KCl and NaCl measurement
isopair = [tst,ref]
isobase,molsT,molsR,ionsT,ionsR,T = pz.data.get_isopair(isobase,isopair)

# Calculate reference model stuff
cf = pz.cdicts.MPH
cf.bC['K-Cl'] = pz.coeffs.bC_K_Cl_A99 # works much better than ZD17...!

isobase['osm_calc_' + tst] = pz.model.osm(molsT,ionsT,T,cf)
isobase['osm_calc_' + ref] = pz.model.osm(molsR,ionsR,T,cf)

isobase['aw_calc_' + tst] = pz.model.osm2aw(molsT,pz.misc.pd2vs(
                                            isobase['osm_calc_' + tst]))
isobase['aw_calc_' + ref] = pz.model.osm2aw(molsR,pz.misc.pd2vs(
                                            isobase['osm_calc_' + ref]))

# Calculate osmotic coefficients from the measurements
isobase['osm_meas_' + tst] = pz.experi.osm(molsT,molsR,
                                    isobase['osm_calc_' + ref])
isobase['dosm_' + tst] = isobase['osm_meas_' + tst] \
                       - isobase['osm_calc_' + tst]

#isobase['osm_meas_' + ref] = pz.experi.osm(molsR,molsT,
#                                    isobase['osm_calc_' + tst])

# Get charges
_,zCT,zAT,_,_ = pz.data.znu([tst])
_,zCR,zAR,_,_ = pz.data.znu([ref])

# Get reference bC coeffs at 298.15 K
bCT = cf.bC[ionsT[0] + '-' + ionsT[1]](298.15)
bCR = cf.bC[ionsR[0] + '-' + ionsR[1]](298.15)

# Derive expected uncertainty profile shapes
pshape = {'totR': np.vstack(np.linspace(0.001,2.5,100)**2)}
pshape['molsR'] = np.concatenate((pshape['totR'],pshape['totR']),axis=1)
pshape['T']     = np.full_like(pshape['totR'],298.15)
pshape['osmR']  = pz.fitting.osm(pshape['molsR'],zCR,zAR,pshape['T'],*bCR)
pshape['tot']   = pz.experi.get_osm(bCT,
                                    pshape['T'],
                                    pshape['molsR'],
                                    pshape['osmR'])
pshape['mols']  = np.concatenate((pshape['tot'],pshape['tot']),axis=1)

# Get derivatives
osmargs = [pshape['tot'],pshape['totR'],isopair,pshape['T'],bCR]
pshape['dosm_dtot']  = pz.experi.dosm_dtot (*osmargs)
pshape['dosm_dtotR'] = pz.experi.dosm_dtotR(*osmargs)

# Create sources pivot table
isoe = pd.pivot_table(isobase,
                      values  = isopair,
                      index   = ['src'],
                      aggfunc = [np.min,np.max,len])

#%% Cycle through sources and fit residuals
isoerr_sys = {}
isobase['dosm_' + tst + '_sys'] = np.nan

for src in isoe.index:
    
    # Logical to select data by source
    SL = isobase.src == src
    
    # Fit residuals
    isoerr_sys[src] = optimize.least_squares(lambda isoerr:
        pz.experi.isofit(isoerr,isobase[tst][SL]) \
        - isobase['dosm_' + tst][SL],4e-4)['x']
        
    # Subtract systematic errors
    isobase.loc[SL,'dosm_' + tst + '_sys'] = isobase['dosm_' + tst][SL] \
        - pz.experi.isofit(isoerr_sys[src],isobase[tst][SL])

#%% Simulation function
def sim_iso():
    
    # Simulate new molality datasets (both electrolytes)
    

    # Simulate a new set of bC coeffs (reference) - provide as an input?

    
    # Calculate reference osmotic coefficient
    
    
    # Calculate measured osmotic coefficient
    
    
    # Fit new bC coeffs (measured)
    

    return

# Save isobase for MATLAB plotting
isobase.to_csv('pickles/data_iso.csv')
savemat('pickles/data_iso.mat',pshape)
savemat('pickles/isoerr_sys.mat',isoerr_sys)

isoe.to_csv('pickles/data_isoe.csv')
