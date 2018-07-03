from  autograd import numpy as np
import pandas as pd
from scipy import optimize
from .constants import Mw
from . import model, tconv

##### DEGREE OF DISSOCIATION #################################################

# Dataset loading
def dis(datapath):
    
    disbase = pd.read_excel(datapath+'dis.xlsx', sheet_name='DIS data',
                            header=0, skiprows=2, usecols=5)
    disbase = prep(disbase)
    
    # Calculate bisulfate speciation
    L = disbase.ele == 'H2SO4'
    for var in ['TSO4','mSO4','mHSO4','mH']:
        disbase[var] = np.nan
    disbase.loc[L,'TSO4' ] = disbase.m[L]
    disbase.loc[L,'mSO4' ] = disbase.TSO4[L] * disbase.a_bisulfate[L]
    disbase.loc[L,'mHSO4'] = disbase.TSO4[L] - disbase.mSO4[L]
    disbase.loc[L,'mH'   ] = disbase.TSO4[L] + disbase.mSO4[L]
    
    return disbase

# Simulate expected values - CRP94 system H2SO4
def dis_sim_H2SO4(TSO4,T,cf):
    
    # Define minimisation function
    def minifun(mH,TSO4,T,cf):
        
        # Calculate [H+] and ionic speciation
        mH = np.vstack(mH)
        mHSO4 = 2*TSO4 - mH
        mSO4  = mH - TSO4
        
        # Create molality & ions arrays
        mols = np.concatenate((mH,mHSO4,mSO4), axis=1)
        ions = np.array(['H','HSO4','SO4'])
        
        # Calculate activity coefficients
        ln_acfs = model.ln_acfs(mols,ions,T,cf)
        gH    = np.exp(ln_acfs[:,0])
        gHSO4 = np.exp(ln_acfs[:,1])
        gSO4  = np.exp(ln_acfs[:,2])
        
        # Evaluate residuals
        return cf.getKeq(T, mH=mH,gH=gH, mHSO4=mHSO4,gHSO4=gHSO4, 
                       mSO4=mSO4,gSO4=gSO4)
        
    # Solve for mH
    mH = np.vstack(np.full_like(T,np.nan))
    
    for i in range(len(mH)):
        
        iT = np.array([T[i]])
        itots = np.array([TSO4[i,:]])
        
        mH[i] = optimize.least_squares(lambda mH: minifun(mH,itots,iT,cf),
                                       1.5*itots[0],
                                       bounds=(itots[0],2*itots[0]),
                                       method='trf',
                                       xtol=1e-12)['x']

    return mH


##### FREEZING POINT DEPRESSION ###############################################

def fpd(datapath):

    fpdbase = pd.read_excel(datapath+'fpd.xlsx', sheet_name='FPD data',
                            header=0, skiprows=2, usecols=6)

    # Calculate freezing point temperature from FPD
    fpdbase['t'] = 273.15 - fpdbase.fpd
    fpdbase = prep(fpdbase)

    # FPD to osmotic coefficient
    mols = np.array([(fpdbase.nC*fpdbase.m).values,
                     (fpdbase.nA*fpdbase.m).values]).transpose()
    fpdbase['osm'] = tconv.fpd2osm(mols,fpdbase.fpd.values)

    return fpdbase

##### VAPOUR PRESSURE LOWERING ################################################

def vpl(datapath):

    vplbase = pd.read_excel(datapath+'vpl.xlsx', sheet_name='VPL data',
                            header=0, skiprows=2, usecols=8)
    vplbase = prep(vplbase)

    # Osmotic coefficient from water activity
    vplbase['osm'] = -np.log(vplbase.aw) / (vplbase.nu * vplbase.m * Mw)

    return vplbase

##### GENERIC FUNCTIONS #######################################################
    
def prep(xxxbase):

    # Add extra variables
    xxxbase['sqm'] = np.sqrt(xxxbase.m)
    xxxbase['nu'],xxxbase['zC'],xxxbase['zA'],xxxbase['nC'],xxxbase['nA'] \
        = znu(xxxbase.ele)

    # Sort by electrolyte, then molality, then source
    xxxbase = xxxbase.sort_values(['ele', 'm', 'src'])

    return xxxbase

def znu(ele):

    # Define dicts
    zC = {'NaCl':+1, 'KCl':+1, 'Na2SO4':+1, 'CaCl2':+2, 'H2SO4':+1}
    zA = {'NaCl':-1, 'KCl':-1, 'Na2SO4':-2, 'CaCl2':-1, 'H2SO4':-2}
    nC = {'NaCl': 1, 'KCl': 1, 'Na2SO4': 2, 'CaCl2': 1, 'H2SO4': 2}
    nA = {'NaCl': 1, 'KCl': 1, 'Na2SO4': 1, 'CaCl2': 2, 'H2SO4': 1}

    # Return: nu, zC, zA, nuC, nuA
    return ele.map(nC) + ele.map(nA), \
           ele.map(zC), ele.map(zA), ele.map(nC), ele.map(nA)
           