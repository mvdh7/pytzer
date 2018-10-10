from autograd import numpy as np
import pytzer as pz
from scipy.io import savemat
from scipy import optimize
import pandas as pd
pd2vs = pz.misc.pd2vs

# Load raw isopiestic dataset and cut to only 298.15 K data
isobase = pz.data.iso('datasets/')
isobase = isobase.loc[isobase.t == 298.15]

# Select only rows where there is a KCl and NaCl measurement
isopair = ['KCl','NaCl']
isobase,mols0,mols1,ions0,ions1,T = pz.data.get_isopair(isobase,isopair)

# Calculate reference model stuff
cf = pz.cdicts.MPH
cf.bC['K-Cl'] = pz.coeffs.bC_K_Cl_A99 # works much better than ZD17...!

isobase['osm_ref_' + isopair[0]] = pz.model.osm(mols0,ions0,T,cf)
isobase['osm_ref_' + isopair[1]] = pz.model.osm(mols1,ions1,T,cf)

isobase['aw_ref_' + isopair[0]] = pz.model.osm2aw(mols0,pz.misc.pd2vs(
                                  isobase['osm_ref_' + isopair[0]]))
isobase['aw_ref_' + isopair[1]] = pz.model.osm2aw(mols1,pz.misc.pd2vs(
                                  isobase['osm_ref_' + isopair[1]]))

# Calculate osmotic coefficients from the measurements
isobase['osm_meas_' + isopair[0]] = pz.experi.osm(mols0,mols1,
                                    isobase['osm_ref_' + isopair[1]])

isobase['osm_meas_' + isopair[1]] = pz.experi.osm(mols1,mols0,
                                    isobase['osm_ref_' + isopair[0]])

# Get charges
_,zC0,zA0,_,_ = pz.data.znu([isopair[0]])
_,zC1,zA1,_,_ = pz.data.znu([isopair[1]])

# Get reference bC coeffs at 298.15 K
bC0 = cf.bC[ions0[0] + '-' + ions0[1]](298.15)
bC1 = cf.bC[ions1[0] + '-' + ions1[1]](298.15)

# Derive expected uncertainty profile shapes
pshape = {'totR': np.vstack(np.linspace(0.001,2.5,200)**2)}
pshape['molsR'] = np.concatenate((pshape['totR'],pshape['totR']),axis=1)
pshape['T']     = np.full_like(pshape['totR'],298.15)
pshape['osmR']  = pz.fitting.osm(pshape['molsR'],1.,-1.,pshape['T'],*bC1)
pshape['tot']   = np.full_like(pshape['totR'],np.nan)

for M in range(len(pshape['T'])):
    pshape['tot'][M] = optimize.least_squares(lambda tot:
        pz.fitting.osm(np.array([[tot,tot]]), # WTF???!??!?!1
                       1.,-1.,
                       np.vstack(pshape['T'][M]),
                       *bC0).ravel(),
                                              1.)['x']

#pz.fitting.osm(np.array([[tot,tot]]),1.,-1.,np.vstack(pshape['T'][M]),
#                       *bC0).ravel() * (tot + tot) \
#                       - pshape['osmR'][M] * np.sum(pshape['molsR'][M])
        
# Simulate a perfect isopiestic dataset
#optimize.least_squares()

# Get derivatives
isobase['dosm0_dtot0'] = pz.experi.dosm_dtot(pd2vs(isobase[isopair[0]]),
                                             pd2vs(isobase[isopair[1]]),
                                             isopair,T,bC1)

# Simulation function
def sim_iso():
    
    # Simulate new molality datasets (both electrolytes)
    

    # Simulate a new set of bC coeffs (reference) - provide as an input?

    
    # Calculate reference osmotic coefficient
    
    
    # Calculate measured osmotic coefficient
    
    
    # Fit new bC coeffs (measured)
    

    return






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
