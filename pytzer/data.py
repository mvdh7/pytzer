from autograd import numpy as np
import pandas as pd
from scipy import optimize
from .misc import pd2vs, ismember
from .     import model

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

# Simulate expected values - H2SO4
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

# Simulate expected values - H2SO4, solving for alpha
def dis_sim_H2SO4_alpha(TSO4,T,cf):
    
    # Define minimisation function solving for alpha
    def minifun(alpha,TSO4,T,cf):
        
        # Calculate [H+] and ionic speciation
        mSO4  = TSO4 * alpha
        mH    = TSO4 + mSO4
        mHSO4 = TSO4 - mSO4
        
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
        
    # Solve for alpha
    alpha = np.vstack(np.full_like(T,np.nan))
    
    for i in range(len(alpha)):
        
        iT = np.array([T[i]])
        itots = np.array([TSO4[i,:]])
        
        alpha[i] = optimize.least_squares(
                lambda alpha: minifun(alpha,itots,iT,cf),
                0.5,
                bounds=(0,1),
                method='trf',
                xtol=1e-12)['x']

    return alpha


##### FREEZING POINT DEPRESSION ###############################################

def fpd(datapath):

    fpdbase = pd.read_excel(datapath+'fpd.xlsx', sheet_name='FPD data',
                            header=0, skiprows=2, usecols=6)

    # Calculate freezing point temperature from FPD
    fpdbase['t'] = 273.15 - fpdbase.fpd
    T = pd2vs(fpdbase.t)
    
    # Calculate extras and sort
    fpdbase,mols,ions = prep(fpdbase)

    return fpdbase, mols, ions, T

##### VAPOUR PRESSURE LOWERING ################################################

def vpl(datapath):

    vplbase = pd.read_excel(datapath+'vpl.xlsx', sheet_name='VPL data',
                            header=0, skiprows=2, usecols=7)
    vplbase,mols,ions = prep(vplbase)
    T = pd2vs(vplbase.t)

    return vplbase, mols, ions, T

##### ELECTROMOTIVE FORCE #####################################################

def emf(datapath):

    emfbase = pd.read_excel(datapath+'emf.xlsx', sheet_name='EMF data',
                            header=0, skiprows=2, usecols=5)
    emfbase = prep(emfbase)

    return emfbase

##### GENERIC FUNCTIONS #######################################################

# Calculate some useful variables and sort database
def prep(xxxbase):

    # Add extra variables
    xxxbase['sqm'] = np.sqrt(xxxbase.m)
    xxxbase['nu'],xxxbase['zC'],xxxbase['zA'],xxxbase['nC'],xxxbase['nA'] \
        = znu(xxxbase.ele)

    # Sort by electrolyte, then molality, then source
    xxxbase = xxxbase.sort_values(['ele', 'm', 'src'])

    # Get mols and ions arrays for pz.model functions
    ions,_,_,idict = ele2ions(xxxbase.ele.values)
    
    mols = np.zeros((np.shape(xxxbase)[0],len(ions)))
    
    for ele in xxxbase.ele.unique():
        
        C = np.where(ions == idict[ele][0])
        A = np.where(ions == idict[ele][1])
        
        EL = xxxbase.ele == ele
        
        mols[EL,C] = xxxbase.m[EL] * xxxbase.nC[EL]
        mols[EL,A] = xxxbase.m[EL] * xxxbase.nA[EL]

    return xxxbase, mols, ions

# Calculate ionic charges and stoichiometry assuming complete dissociation
def znu(ele):
           
    # Define properties                   zC   zA   nC   nA
    zC_zA_nC_nA = {'BaCl2'  : np.float_([ +2 , -1 ,  1 ,  2 ]),
                   'CaCl2'  : np.float_([ +2 , -1 ,  1 ,  2 ]),
                   'H2SO4'  : np.float_([ +1 , -2 ,  2 ,  1 ]),
                   'KCl'    : np.float_([ +1 , -1 ,  1 ,  1 ]),
                   'KOH'    : np.float_([ +1 , -1 ,  1 ,  1 ]),
                   'MgCl2'  : np.float_([ +2 , -1 ,  1 ,  2 ]),
                   'Na2SO4' : np.float_([ +1 , -2 ,  2 ,  1 ]),
                   'NaCl'   : np.float_([ +1 , -1 ,  1 ,  1 ]),
                   'ZnBr2'  : np.float_([ +2 , -1 ,  1 ,  2 ]),
                   'ZnCl2'  : np.float_([ +2 , -1 ,  1 ,  2 ])}

    # Extract properties for input ele list from dict
    znus = np.array([zC_zA_nC_nA[ionpair] for ionpair in ele])
    
    # Prepare and return results
    zC = np.vstack(znus[:,0])
    zA = np.vstack(znus[:,1])
    nC = np.vstack(znus[:,2])
    nA = np.vstack(znus[:,3])
    
    nu = nC + nA
    
    return nu, zC, zA, nC, nA
          
# Return list of unique ions from list of electrolytes 
def ele2ions(ele):
    
    # Define ions in each electrolyte given full dissociation
    idict = {'BaCl2'  : np.array(['Ba', 'Cl' ]),
             'CaCl2'  : np.array(['Ca', 'Cl' ]),
             'H2SO4'  : np.array(['H' , 'SO4']),
             'KCl'    : np.array(['K' , 'Cl' ]),
             'KOH'    : np.array(['K' , 'OH' ]),
             'MgCl2'  : np.array(['Mg', 'Cl' ]),
             'Na2SO4' : np.array(['Na', 'SO4']),
             'NaCl'   : np.array(['Na', 'Cl' ]),
             'ZnBr2'  : np.array(['Zn', 'Br' ]),
             'ZnCl2'  : np.array(['Zn', 'Cl' ])}

    cats = np.unique(np.array([idict[ionpair][0] for ionpair in ele]).ravel())
    anis = np.unique(np.array([idict[ionpair][1] for ionpair in ele]).ravel())   
       
    ions = np.concatenate((cats,anis))

    return ions, cats, anis, idict
        
# Get a subset of electrolytes from a database
def subset_ele(xxxbase,mols,ions,T,subset):
    
    R = ismember(xxxbase.ele,subset)
    xxxbase = xxxbase[R]
    
    C = ismember(ions,ele2ions(subset)[0])
    
    mols = mols[R][:,C]
    ions = ions[C]
    T    = T[R]
    
    return xxxbase, mols, ions, T
