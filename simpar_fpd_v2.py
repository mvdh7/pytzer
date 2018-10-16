# Import libraries
from autograd import numpy as np
from autograd import elementwise_grad as egrad
import pandas as pd
#from scipy import optimize
from scipy.io import savemat
#from scipy.interpolate import pchip
#import pickle
import pytzer as pz
from pytzer.misc import ismember, pd2vs

# Load raw datasets
datapath = 'datasets/'
fpdbase,mols,ions,T = pz.data.fpd(datapath)

# Select electrolytes for analysis
fpdbase,mols,ions,T = pz.data.subset_ele(fpdbase,mols,ions,T,
                                         np.array(['NaCl',
                                                   'KCl',
                                                   'CaCl2']))#,
                                                   #MgCl2']))

#%% Exclude smoothed datasets
S = fpdbase.smooth == 0
fpdbase = fpdbase[S]
mols    = mols   [S]
T       = T      [S]

# Prepare model cdict
cf = pz.cdicts.MPH
eles = fpdbase.ele
cf.add_zeros(fpdbase.ele)
cf.bC['K-Cl'] = pz.coeffs.bC_K_Cl_A99 # works much better than ZD17...!
#cf.bC['Ca-Cl'] = pz.coeffs.bC_Ca_Cl_JESS

# Calculate osmotic coefficient at measurement temperature
fpd = pd2vs(fpdbase.fpd)
fpdbase['osm_meas'] = pz.tconv.fpd2osm(mols,fpd)
fpdbase['osm_calc'] = pz.model.osm(mols,ions,T,cf)

# Convert temperatures to 298.15 K
fpdbase['t25'] = 298.15
T25 = pd2vs(fpdbase.t25)

# Create initial electrolytes pivot table
fpde = pd.pivot_table(fpdbase,
                      values  = ['m'],
                      index   = ['ele'],
                      aggfunc = [np.min,np.max,len])

# Convert measured FPD into osmotic coeff. at 298.15 K
fpdbase['osm25_meas'] = np.nan

for ele in fpde.index:
    
    Efpdbase,_,Eions,_ = pz.data.subset_ele(fpdbase,mols,ions,T,
                                            np.array([ele]))
    EL = ismember(fpdbase.ele,np.array([ele]))
    
    if ele == 'CaCl2':
    
        fpdbase.loc[EL,'osm25_meas'] = pz.isoref.osm2osm25_CaCl2(
                pd2vs(Efpdbase.m),pd2vs(Efpdbase.t),pd2vs(Efpdbase.osm_meas))
        
    else:
        
        fpdbase.loc[EL,'osm25_meas'] = pz.tconv.osm2osm(
            pd2vs(Efpdbase.m),pd2vs(Efpdbase.nC),pd2vs(Efpdbase.nA),
            Eions,pd2vs(Efpdbase.t),pd2vs(Efpdbase.t25),pd2vs(Efpdbase.t25),
            cf,pd2vs(Efpdbase.osm_meas))

# Calculate model osmotic coefficient at 298.15 K
fpdbase['osm25_calc'] = pz.model.osm(mols,ions,T25,cf)

# Overwrite CaCl2 reference values using PCHIP of RC97 look-up table
L = fpdbase.ele == 'CaCl2'
fpdbase.loc[L,'osm25_calc'] = pz.isoref.osm_CaCl2(fpdbase.m[L])

# Calculate osmotic coefficient residuals
fpdbase['dosm'  ] = fpdbase.osm_meas   - fpdbase.osm_calc
fpdbase['dosm25'] = fpdbase.osm25_meas - fpdbase.osm25_calc

# Create electrolytes/sources pivot table
fpdp = pd.pivot_table(fpdbase,
                      values  = ['m','dosm25'],
                      index   = ['ele','src'],
                      aggfunc = [np.mean,np.std,len])

# Calculate expected FPD
tot = pd2vs(fpdbase.m)
_,_,_,nC,nA = pz.data.znu(fpdp.index.levels[0])
fpdbase['fpd_calc'] = np.nan
#fpdbase['dfpd'    ] = np.nan
#fpdbase['dfpd_sys'] = np.nan
#fpderr_sys = {}
#fpderr_rdm = {}

for E,ele in enumerate(fpdp.index.levels[0]): 
    print('Optimising FPD fit for ' + ele + '...')
    
    # Calculate expected FPD
    Eions = pz.data.ele2ions(np.array([ele]))[0]
    EL = fpdbase.ele == ele
    
    if ele == 'CaCl2':
# Replicate pz.tconv.tot2fpd25 function, but using PCHIP interpolation to get
#  CaCl2 osmotic coefficient at 298.15 K following RC97
        fpdbase.loc[EL,'fpd_calc'] = pz.isoref.tot2fpd25_CaCl2(tot[EL])
        
    else:
        fpdbase.loc[EL,'fpd_calc'] = pz.tconv.tot2fpd(tot[EL],
                                                      Eions,
                                                      nC[E],
                                                      nA[E],
                                                      cf)

fpdbase['dfpd'] = fpdbase.fpd - fpdbase.fpd_calc

# Update electrolytes/sources pivot table
fpdp = pd.pivot_table(fpdbase,
                      values  = ['m','dfpd','dosm25'],
                      index   = ['ele','src'],
                      aggfunc = [np.mean,np.std,len])

# Get pshapes
pshape_fpd = {'tot': np.vstack(np.linspace(0.001,2.5,100))**2}
pshape_fpd['t25'] = np.full_like(pshape_fpd['tot'],298.15)

# Solve for exact FPD across molality range
pshape_fpd['fpd_CaCl2'] = pz.isoref.tot2fpd25_CaCl2(pshape_fpd['tot'])

#%% Calculate osmotic coefficient from FPD
mols_CaCl2 = np.concatenate((pshape_fpd['tot'],pshape_fpd['tot']*2),axis=1)
pshape_fpd['osm_fpd_CaCl2'] = pz.tconv.fpd2osm(mols_CaCl2,
                                               pshape_fpd['fpd_CaCl2'])

# Convert osmotic coefficient to 298.15 K
pshape_fpd['osm25_fpd_CaCl2'] = pz.isoref.osm2osm25_CaCl2(
        pshape_fpd['tot'],
        273.15-pshape_fpd['fpd_CaCl2'],
        pshape_fpd['osm_fpd_CaCl2'])

# Calculate osmotic coefficient directly
pshape_fpd['osm25_calc_CaCl2'] = pz.isoref.osm_CaCl2(pshape_fpd['tot'])

## Calculate osmotic coefficient with small linear error in FPD temperature
#pshape_fpd['fpd_err_CaCl2'] = pshape_fpd['fpd_CaCl2'] \
#    + pshape_fpd['tot'] * 0
#    
## Get osmotic coefficient at 298.15 K with FPD temp error
#pshape_fpd['osm_fpd_err_CaCl2'] = pz.tconv.fpd2osm(mols_CaCl2,
#        pshape_fpd['fpd_err_CaCl2'])
#pshape_fpd['osm25_fpd_err_CaCl2'] = pz.isoref.osm2osm25_CaCl2(
#        pshape_fpd['tot'] * 2,
#        273.15-pshape_fpd['fpd_err_CaCl2'],
#        pshape_fpd['osm_fpd_err_CaCl2'])

# Get uncertainty profiles for CaCl2
def fx_pshape_err_CaCl2(dT,mT,dtot,mtot):
    
    fpd_err_CaCl2 = pshape_fpd['fpd_CaCl2'] * mT + dT
    osm_fpd_err_CaCl2 = pz.tconv.fpd2osm(mols_CaCl2 * mtot + dtot,
            fpd_err_CaCl2)
    osm25_fpd_err_CaCl2 = pz.isoref.osm2osm25_CaCl2(
            pshape_fpd['tot'] * mtot + dtot,
            273.15 - fpd_err_CaCl2,
            osm_fpd_err_CaCl2)
    
    return osm25_fpd_err_CaCl2

fxps_dT   = egrad(fx_pshape_err_CaCl2,argnum=0)
fxps_mT   = egrad(fx_pshape_err_CaCl2,argnum=1)
fxps_dtot = egrad(fx_pshape_err_CaCl2,argnum=2)
fxps_mtot = egrad(fx_pshape_err_CaCl2,argnum=3)

f0 = np.full_like(pshape_fpd['tot'],0.)
f1 = np.full_like(pshape_fpd['tot'],1.)

pshape_fpd['dosm25_fpd_ddT_CaCl2']   = fxps_dT  (f0,f1,f0,f1)
pshape_fpd['dosm25_fpd_dmT_CaCl2']   = fxps_mT  (f0,f1,f0,f1)
pshape_fpd['dosm25_fpd_ddtot_CaCl2'] = fxps_dtot(f0,f1,f0,f1)
pshape_fpd['dosm25_fpd_dmtot_CaCl2'] = fxps_mtot(f0,f1,f0,f1)

# Save results for MATLAB figures
fpdbase.to_csv('pickles/simpar_fpd_v2.csv')
savemat('pickles/simpar_fpd_v2.mat',{'pshape_fpd':pshape_fpd})

##%% Repeat for NaCl
#ions_NaCl = np.array(['Na','Cl'])
#_,_,_,nC_NaCl,nA_NaCl = pz.data.znu(['NaCl'])
#pshape_fpd['fpd_NaCl'] = pz.tconv.tot2fpd25(pshape_fpd['tot'],
#                                          ions_NaCl,nC_NaCl,nA_NaCl,cf)
## if i switch to tot2fpd25 here ^^ it makes the osm WORK at 25 but NOT at FP
## (i.e. opposite of tot2fpd) - both are shifted UP equally in MATLAB subplot 2
## This is essentially due to the difference between Archer's model at the two
##  different temperatures vs the theoretical temperature conversion approach
#
#mols_NaCl = np.concatenate((pshape_fpd['tot'],pshape_fpd['tot']),axis=1)
#pshape_fpd['osm_fpd_NaCl'] = pz.tconv.fpd2osm(mols_NaCl,
#                                              pshape_fpd['fpd_NaCl'])
#pshape_fpd['osm25_fpd_NaCl'] = pz.tconv.osm2osm(pshape_fpd['tot'],
#                                                nC_NaCl,nA_NaCl,ions_NaCl,
#                                                273.15-pshape_fpd['fpd_NaCl'],
#                                                pshape_fpd['t25'],
#                                                pshape_fpd['t25'],
#                                                cf,pshape_fpd['osm_fpd_NaCl'])
#
#pshape_fpd['osm_calc_NaCl'] = pz.model.osm(mols_NaCl,
#                                           ions_NaCl,
#                                           273.15-pshape_fpd['fpd_NaCl'],
#                                           cf)
#pshape_fpd['osm25_calc_NaCl'] = pz.model.osm(mols_NaCl,
#                                             ions_NaCl,
#                                             pshape_fpd['t25'],
#                                             cf)

#%% Run uncertainty propagation analysis [FPD]
#tot = pd2vs(fpdbase.m)
#_,_,_,nC,nA = pz.data.znu(fpdp.index.levels[0])
#fpdbase['fpd_calc'] = np.nan
#fpdbase['dfpd'    ] = np.nan
#fpdbase['dfpd_sys'] = np.nan
#fpderr_sys = {}
#fpderr_rdm = {}
#
#for E,ele in enumerate(fpdp.index.levels[0]): 
#    print('Optimising FPD fit for ' + ele + '...')
#    
#    # Calculate expected FPD
#    Eions = pz.data.ele2ions(np.array([ele]))[0]
#    EL = fpdbase.ele == ele
#    
#    if ele == 'CaCl2':
##        fpdbase.loc[EL,'fpd_calc'] = pz.tconv.tot2fpd25(tot[EL],
##                                                        Eions,
##                                                        nC[E],
##                                                        nA[E],
##                                                        cf)
#        
## =============================================================================
## Replicate pz.tconv.tot2fpd25 function, but using PCHIP interpolation to get
##  CaCl2 osmotic coefficient at 298.15 K following RC97
#        
#        Ctot = tot[EL]
#        CnC = nC[E]
#        CnA = nA[E]
#        Cmols = np.concatenate((Ctot*CnC,Ctot*CnA), axis=1)
#        Cfpd = np.full_like(Ctot,np.nan)
#        CT25 = np.full_like(Ctot,298.15, dtype='float64')
#        
##        Cosm25 = pz.model.osm(Cmols,Eions,CT25,cf)
#        
#        # Use PCHIP interpolation to get calculated osm25 for CaCl2
#        with open('pickles/fortest_CaCl2_10.pkl','rb') as f:
#            rc97,F = pickle.load(f)
#        pchip_CaCl2 = pchip(rc97.tot,rc97.osm)
#        
#        Cosm25 = pchip_CaCl2(Ctot)
#        
#        CiT25 = np.vstack([298.15])
#        CiT00 = np.vstack([273.15])
#        
#        for i in range(len(Ctot)):
#            
#            if i/10. == np.round(i/10.):
#                print('Getting FPD %d of %d...' % (i+1,len(Ctot)))
#            
#            imols = np.array([Cmols[i,:]])
#            
## NOT QUITE RIGHT: pz.tconv.osm2osm still uses cf for thermal properties of
##                  CaCl2, which are not properly constrained there...
#            
#            Cfpd[i] = optimize.least_squares(lambda fpd: \
#               (pz.tconv.osm2osm(Ctot[i],CnC,CnA,Eions,CiT00-fpd,CiT25,CiT25,
#                                 cf,
#                        pz.tconv.fpd2osm(imols,fpd)) - Cosm25[i]).ravel(),
#                                            0., method='trf')['x'][0]
#               
#        fpdbase.loc[EL,'fpd_calc'] = Cfpd
#        
## =============================================================================      
#        
#    else:
#        fpdbase.loc[EL,'fpd_calc'] = pz.tconv.tot2fpd(tot[EL],
#                                                      Eions,
#                                                      nC[E],
#                                                      nA[E],
#                                                      cf)
#    fpdbase.loc[EL,'dfpd'] = fpdbase.fpd[EL] - fpdbase.fpd_calc[EL]
#    
#    # Estimate uncertainties for each source
#    fpderr_sys[ele] = {}
#    fpderr_rdm[ele] = {}
#    
#    for src in fpdp.loc[ele].index:
#        
#        SL = np.logical_and(EL,fpdbase.src == src)
##        if ele == 'CaCl2':
##            SL = np.logical_and(SL,fpdbase.m <= 1.5)
#        
#        # Evaluate systematic component of error
#        fpderr_sys[ele][src] = optimize.least_squares(lambda syserr: \
#            syserr[1] * fpdbase[SL].m + USYS * syserr[0] - fpdbase[SL].dfpd,
#                                             [0.,0.])['x']
#        
#        if USYS == 1:
#            if sum(SL) < 6:
#                fpderr_sys[ele][src][1] = 0
#                fpderr_sys[ele][src][0] = optimize.least_squares(
#                    lambda syserr: syserr - fpdbase[SL].dfpd,0.)['x'][0]
#
#        fpdbase.loc[SL,'dfpd_sys'] \
#            =  fpdbase.dfpd[SL] \
#            - (fpdbase.m[SL] * fpderr_sys[ele][src][1] \
#               +               fpderr_sys[ele][src][0])
#                       
#        # Evaluate random component of error
#        fpderr_rdm[ele][src] = optimize.least_squares(lambda rdmerr: \
#            rdmerr[1] * fpdbase[SL].m + rdmerr[0] \
#            - np.abs(fpdbase[SL].dfpd_sys), [0,0])['x']
#        
#        if fpderr_rdm[ele][src][0] < 0:
#            fpderr_rdm[ele][src][0] = 0
#            fpderr_rdm[ele][src][1] = optimize.least_squares(lambda rdmerr: \
#                rdmerr * fpdbase[SL].m \
#                - np.abs(fpdbase[SL].dfpd_sys), 0.)['x']
#        
#        if (sum(SL) < 6) or (fpderr_rdm[ele][src][1] < 0):
#            fpderr_rdm[ele][src][1] = 0
#            fpderr_rdm[ele][src][0] = optimize.least_squares(lambda rdmerr: \
#                rdmerr - np.abs(fpdbase[SL].dfpd_sys), 0.)['x'][0]
#         
## Add 'all' fields for easier plotting in MATLAB
#for ele in fpdp.index.levels[0]:
#    Eksys = list(fpderr_sys[ele].keys())
#    fpderr_sys[ele]['all_int'] = np.array( \
#        [fpderr_sys[ele][src][0] for src in Eksys])
#    fpderr_sys[ele]['all_grad'] = np.array( \
#        [fpderr_sys[ele][src][1] for src in Eksys])
#    Ekrdm = list(fpderr_rdm[ele].keys())
#    fpderr_rdm[ele]['all_int'] = np.array( \
#        [fpderr_rdm[ele][src][0] for src in Ekrdm])
#    fpderr_rdm[ele]['all_grad'] = np.array( \
#        [fpderr_rdm[ele][src][1] for src in Ekrdm])
#
##%% Generate fit splines for MATLAB
#pshape_fpd = {'tot': np.vstack(np.linspace(0.001,3,100))**2}
#pshape_fpd['t25'] = np.full_like(pshape_fpd['tot'],298.15)
#
## Define and differentiate conversion function
#def fpd2osm25(tot,n1,n2,ions,fpd,T1,TR,cf):
#    mols = np.concatenate((tot*n1,tot*n2),axis=1)
#    return pz.tconv.osm2osm(tot,n1,n2,ions,273.15-fpd,T1,TR,cf,
#                            pz.tconv.fpd2osm(mols,fpd))
#
#dosm25_dfpd = egrad(fpd2osm25,argnum=4)
#
#
#for ele in fpdp.index.levels[0]:
#    
#    Eions = pz.data.ele2ions(np.array([ele]))[0]
#    _,_,_,EnC,EnA = pz.data.znu(np.array([ele]))
#    
#    pshape_fpd['fpd_' + ele] = pz.tconv.tot2fpd(pshape_fpd['tot'],
#                                                Eions,EnC,EnA,cf)
#    
#    pshape_fpd['osm25_' + ele] = fpd2osm25(pshape_fpd['tot'],EnC,EnA,Eions,
#                                           pshape_fpd['fpd_' + ele],
#                                           pshape_fpd['t25'],
#                                           pshape_fpd['t25'],
#                                           cf)
#    
#    pshape_fpd['dosm25_' + ele] = dosm25_dfpd(pshape_fpd['tot'],EnC,EnA,Eions,
#                                              pshape_fpd['fpd_' + ele],
#                                              pshape_fpd['t25'],
#                                              pshape_fpd['t25'],
#                                              cf)

## Pickle outputs for simloop
#with open('pickles/simpar_fpd.pkl','wb') as f:
#    pickle.dump((fpdbase,fpderr_rdm,fpderr_sys),f)
#

#savemat('pickles/simpar_fpd.mat',{'fpderr_sys':fpderr_sys,
#                                  'fpderr_rdm':fpderr_rdm,
#                                  'pshape_fpd':pshape_fpd})
#
#print('FPD fit optimisation complete!')
