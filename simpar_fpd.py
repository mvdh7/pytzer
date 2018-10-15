# Import libraries
from autograd import numpy as np
from autograd import elementwise_grad as egrad
import pandas as pd
from scipy import optimize
from scipy.io import savemat
from scipy.interpolate import pchip
import pickle
import pytzer as pz
pd2vs = pz.misc.pd2vs
from mvdh import ismember

# Set whether to allow uniform systematic offset
USYS = np.float_(1) # 0 for no, 1 for yes

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
cf.bC['Ca-Cl'] = pz.coeffs.bC_Ca_Cl_JESS

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
    
    fpdbase.loc[EL,'osm25_meas'] = pz.tconv.osm2osm(
            pd2vs(Efpdbase.m),pd2vs(Efpdbase.nC),pd2vs(Efpdbase.nA),
            Eions,pd2vs(Efpdbase.t),pd2vs(Efpdbase.t25),pd2vs(Efpdbase.t25),
            cf,pd2vs(Efpdbase.osm_meas))

# Calculate model osmotic coefficient at 298.15 K
fpdbase['osm25_calc'] = pz.model.osm(mols,ions,T25,cf)
fpdbase['dosm'  ] = fpdbase.osm_meas   - fpdbase.osm_calc
fpdbase['dosm25'] = fpdbase.osm25_meas - fpdbase.osm25_calc

## Quickly visualise residuals
#from matplotlib import pyplot as plt
#fig,ax = plt.subplots(1,1)
#fpdbase.plot.scatter('m','dosm25', ax=ax)
#ax.set_xlim((0,6))
#ax.grid(alpha=0.5)

# Create electrolytes/sources pivot table
fpdp = pd.pivot_table(fpdbase,
                      values  = ['m','dosm25'],
                      index   = ['ele','src'],
                      aggfunc = [np.mean,np.std,len])

#%% Run uncertainty propagation analysis [FPD]
tot = pd2vs(fpdbase.m)
_,_,_,nC,nA = pz.data.znu(fpdp.index.levels[0])
fpdbase['fpd_calc'] = np.nan
fpdbase['dfpd'    ] = np.nan
fpdbase['dfpd_sys'] = np.nan
fpderr_sys = {}
fpderr_rdm = {}

for E,ele in enumerate(fpdp.index.levels[0]): 
    print('Optimising FPD fit for ' + ele + '...')
    
    # Calculate expected FPD
    Eions = pz.data.ele2ions(np.array([ele]))[0]
    EL = fpdbase.ele == ele
    
    if ele == 'CaCl2':
#        fpdbase.loc[EL,'fpd_calc'] = pz.tconv.tot2fpd25(tot[EL],
#                                                        Eions,
#                                                        nC[E],
#                                                        nA[E],
#                                                        cf)
        
# =============================================================================
# Replicate pz.tconv.tot2fpd25 function, but using PCHIP interpolation to get
#  CaCl2 osmotic coefficient at 298.15 K following RC97
        
        Ctot = tot[EL]
        CnC = nC[E]
        CnA = nA[E]
        Cmols = np.concatenate((Ctot*CnC,Ctot*CnA), axis=1)
        Cfpd = np.full_like(Ctot,np.nan)
        CT25 = np.full_like(Ctot,298.15, dtype='float64')
        
#        Cosm25 = pz.model.osm(Cmols,Eions,CT25,cf)
        
        # Use PCHIP interpolation to get calculated osm25 for CaCl2
        with open('pickles/fortest_CaCl2_10.pkl','rb') as f:
            rc97,F = pickle.load(f)
        pchip_CaCl2 = pchip(rc97.tot,rc97.osm)
        
        Cosm25 = pchip_CaCl2(Ctot)
        
        CiT25 = np.vstack([298.15])
        CiT00 = np.vstack([273.15])
        
        for i in range(len(Ctot)):
            
            if i/10. == np.round(i/10.):
                print('Getting FPD %d of %d...' % (i+1,len(Ctot)))
            
            imols = np.array([Cmols[i,:]])
            
# NOT QUITE RIGHT: pz.tconv.osm2osm still uses cf for thermal properties of
#                  CaCl2, which are not properly constrained there...
            
            Cfpd[i] = optimize.least_squares(lambda fpd: \
               (pz.tconv.osm2osm(Ctot[i],CnC,CnA,Eions,CiT00-fpd,CiT25,CiT25,
                                 cf,
                        pz.tconv.fpd2osm(imols,fpd)) - Cosm25[i]).ravel(),
                                            0., method='trf')['x'][0]
               
        fpdbase.loc[EL,'fpd_calc'] = Cfpd
        
# =============================================================================      
        
    else:
        fpdbase.loc[EL,'fpd_calc'] = pz.tconv.tot2fpd(tot[EL],
                                                      Eions,
                                                      nC[E],
                                                      nA[E],
                                                      cf)
    fpdbase.loc[EL,'dfpd'] = fpdbase.fpd[EL] - fpdbase.fpd_calc[EL]
    
    # Estimate uncertainties for each source
    fpderr_sys[ele] = {}
    fpderr_rdm[ele] = {}
    
    for src in fpdp.loc[ele].index:
        
        SL = np.logical_and(EL,fpdbase.src == src)
#        if ele == 'CaCl2':
#            SL = np.logical_and(SL,fpdbase.m <= 1.5)
        
        # Evaluate systematic component of error
        fpderr_sys[ele][src] = optimize.least_squares(lambda syserr: \
            syserr[1] * fpdbase[SL].m + USYS * syserr[0] - fpdbase[SL].dfpd,
                                             [0.,0.])['x']
        
        if USYS == 1:
            if sum(SL) < 6:
                fpderr_sys[ele][src][1] = 0
                fpderr_sys[ele][src][0] = optimize.least_squares(
                    lambda syserr: syserr - fpdbase[SL].dfpd,0.)['x'][0]

        fpdbase.loc[SL,'dfpd_sys'] \
            =  fpdbase.dfpd[SL] \
            - (fpdbase.m[SL] * fpderr_sys[ele][src][1] \
               +               fpderr_sys[ele][src][0])
                       
        # Evaluate random component of error
        fpderr_rdm[ele][src] = optimize.least_squares(lambda rdmerr: \
            rdmerr[1] * fpdbase[SL].m + rdmerr[0] \
            - np.abs(fpdbase[SL].dfpd_sys), [0,0])['x']
        
        if fpderr_rdm[ele][src][0] < 0:
            fpderr_rdm[ele][src][0] = 0
            fpderr_rdm[ele][src][1] = optimize.least_squares(lambda rdmerr: \
                rdmerr * fpdbase[SL].m \
                - np.abs(fpdbase[SL].dfpd_sys), 0.)['x']
        
        if (sum(SL) < 6) or (fpderr_rdm[ele][src][1] < 0):
            fpderr_rdm[ele][src][1] = 0
            fpderr_rdm[ele][src][0] = optimize.least_squares(lambda rdmerr: \
                rdmerr - np.abs(fpdbase[SL].dfpd_sys), 0.)['x'][0]
         
# Add 'all' fields for easier plotting in MATLAB
for ele in fpdp.index.levels[0]:
    Eksys = list(fpderr_sys[ele].keys())
    fpderr_sys[ele]['all_int'] = np.array( \
        [fpderr_sys[ele][src][0] for src in Eksys])
    fpderr_sys[ele]['all_grad'] = np.array( \
        [fpderr_sys[ele][src][1] for src in Eksys])
    Ekrdm = list(fpderr_rdm[ele].keys())
    fpderr_rdm[ele]['all_int'] = np.array( \
        [fpderr_rdm[ele][src][0] for src in Ekrdm])
    fpderr_rdm[ele]['all_grad'] = np.array( \
        [fpderr_rdm[ele][src][1] for src in Ekrdm])

#%% Generate fit splines for MATLAB
pshape_fpd = {'tot': np.vstack(np.linspace(0.001,3,100))**2}
pshape_fpd['t25'] = np.full_like(pshape_fpd['tot'],298.15)

# Define and differentiate conversion function
def fpd2osm25(tot,n1,n2,ions,fpd,T1,TR,cf):
    mols = np.concatenate((tot*n1,tot*n2),axis=1)
    return pz.tconv.osm2osm(tot,n1,n2,ions,273.15-fpd,T1,TR,cf,
                            pz.tconv.fpd2osm(mols,fpd))

dosm25_dfpd = egrad(fpd2osm25,argnum=4)


for ele in fpdp.index.levels[0]:
    
    Eions = pz.data.ele2ions(np.array([ele]))[0]
    _,_,_,EnC,EnA = pz.data.znu(np.array([ele]))
    
    pshape_fpd['fpd_' + ele] = pz.tconv.tot2fpd(pshape_fpd['tot'],
                                                Eions,EnC,EnA,cf)
    
    pshape_fpd['osm25_' + ele] = fpd2osm25(pshape_fpd['tot'],EnC,EnA,Eions,
                                           pshape_fpd['fpd_' + ele],
                                           pshape_fpd['t25'],
                                           pshape_fpd['t25'],
                                           cf)
    
    pshape_fpd['dosm25_' + ele] = dosm25_dfpd(pshape_fpd['tot'],EnC,EnA,Eions,
                                              pshape_fpd['fpd_' + ele],
                                              pshape_fpd['t25'],
                                              pshape_fpd['t25'],
                                              cf)

# Pickle outputs for simloop
with open('pickles/simpar_fpd.pkl','wb') as f:
    pickle.dump((fpdbase,fpderr_rdm,fpderr_sys),f)

# Save results for MATLAB figures
fpdbase.to_csv('pickles/simpar_fpd.csv')
savemat('pickles/simpar_fpd.mat',{'fpderr_sys':fpderr_sys,
                                  'fpderr_rdm':fpderr_rdm,
                                  'pshape_fpd':pshape_fpd})

print('FPD fit optimisation complete!')
