import autograd.numpy as np
from autograd import elementwise_grad as egrad
import pandas as pd
import coeffs

##### DICT OF COEFFICIENT FUNCTIONS ###########################################

# Set up dict of coefficient functions
cf = {coeff:{} for coeff in ['bC', 'theta', 'psi', 'dissoc']}

# Debye-Hueckel slope
cf['Aosm'] = coeffs.Aosm_M88

# betas and Cs as cf['bC']['cation-anion']
cf['bC']['Ca-Cl']  = coeffs.CaCl_M88
cf['bC']['Ca-SO4'] = coeffs.CaSO4_M88
cf['bC']['Na-Cl']  = coeffs.NaCl_M88
cf['bC']['Na-SO4'] = coeffs.NaSO4_M88

# thetas as cf['theta']['cation1-cation2'] with cations in alphabetical order
cf['theta']['Ca-Na']  = coeffs.CaNa_M88
cf['theta']['Cl-SO4'] = coeffs.ClSO4_M88

# psis as cf['psi']['cation1-cation2-anion'] with cations in alphabetical order
#   or as cf['psi']['cation-anion1-anion2']  with anions  in alphabetical order
cf['psi']['Ca-Na-Cl']  = coeffs.CaNaCl_M88
cf['psi']['Ca-Na-SO4'] = coeffs.CaNaSO4_M88
cf['psi']['Ca-Cl-SO4'] = coeffs.CaClSO4_M88
cf['psi']['Na-Cl-SO4'] = coeffs.NaClSO4_M88

# Dissociation constants
cf['dissoc']['Kw'] = coeffs.Kw_M88

##### FILE I/O ################################################################

def getIons(filename):
  
    # Import input conditions from .csv
    idf = pd.read_csv(filename, float_precision='round_trip')
        
    # Replace missing values with zero
    idf = idf.fillna(0)
    
    # Get temperatures
    T = idf.temp.values
    
    # Get ionic concentrations
    idf_tots = idf[idf.keys()[idf.keys() != 'temp']]
    tots = idf_tots.values
    
    # Get list of ions
    ions = idf_tots.keys()
       
    return T, tots, ions, idf

##### IONIC CHARGES ###########################################################

def getCharges(ions):

    z = {}
    
    z['Ca']  = np.float_(+2)
    z['H' ]  = np.float_(+1)
    z['K' ]  = np.float_(+1)
    z['Na']  = np.float_(+1)

    z['Cl']  = np.float_(-1)    
    z['OH']  = np.float_(-1)
    z['SO4'] = np.float_(-2)
    
    return np.array([z[ion] for ion in ions])

##### DEBYE-HUECKEL SLOPE #####################################################

b = np.float_(1.2)

def fG(T,I,cf): # CRP94 Eq. (AI1)
    
    return -4 * cf['Aosm'](T)[0] * I * np.log(1 + b*np.sqrt(I)) / b

##### PITZER MODEL SUBFUNCTIONS ###############################################

def g(x): # CRP94 Eq. (AI13)
    return 2 * (1 - (1 + x) * np.exp(-x)) / x**2

def h(x):  # CRP94 Eq. (AI15)
    return (6 - (6 + x*(6 + 3*x + x**2)) * np.exp(-x)) / x**4

def B(T,I,cf,iset): # CRP94 Eq. (AI8)
    
    b0,b1,b2,_,_,a1,a2,_,_ = cf['bC'][iset](T)

    return b0 + b1 * g(a1*np.sqrt(I)) + b2 * g(a2*np.sqrt(I))

def CT(T,I,cf,iset): # P91 Ch. 3 Eq. (53)
    
    _,_,_,C0,C1,_,_,o,_ = cf['bC'][iset](T)
    
    return C0 + 4 * C1 * h(o*np.sqrt(I))

##### EXCESS GIBBS ENERGY #####################################################
    
def Gex_nRT(mols,ions):
    
    # Ionic strength etc.
    zs = getCharges(ions)
    I = 0.5 * (np.sum(mols * zs**2, 1))
    Z = np.sum(mols * np.abs(zs), 1)
    
    # Separate cations and anions
    CL = zs > 0
    cats    = mols[:,CL]
    cations = ions[  CL]
#    zCs     = zs  [  CL]
    AL = zs < 0
    anis    = mols[:,AL]
    anions  = ions[  AL]
#    zAs     = zs  [  AL]
    
    # Begin with Debye-Hueckel component
    Gex_nRT = fG(T,I,cf)
    
    
    
    # Add c-a interactions
    for C,cation in enumerate(cations):
        for A,anion in enumerate(anions):

            iset= '-'.join([cation,anion])

            Gex_nRT = Gex_nRT + cats[:,C] * anis[:,A] \
                * (2*B(T,I,cf,iset) + Z*CT(T,I,cf,iset))
                
    # Add c-c' interactions
    for C0 in range(len(cations)):
        for C1 in range(C0+1,len(cations)):
            
            iset = [cations[C0],cations[C1]]
            iset.sort()
            iset= '-'.join(iset)
            
            Gex_nRT = Gex_nRT + cats[:,C0] * cats[:,C1] \
                * (2 * cf['theta'][iset](T)[0])# + pz.etheta(t,zC[C0],zC[C1],I))
                
    # Add c-c'-a interactions
            for A in range(len(anions)):
                
                itri = '-'.join([iset,anions[A]])
                                
                Gex_nRT = Gex_nRT + cats[:,C0] * cats[:,C1] \
                    * anis[:,A] * cf['psi'][itri](T)[0]

    # Add a-a' interactions
    for A0 in range(len(anions)):
        for A1 in range(A0+1,len(anions)):
            
            iset = [anions[A0],anions[A1]]
            iset.sort()
            iset= '-'.join(iset)
            
            Gex_nRT = Gex_nRT + anis[:,A0] * anis[:,A1] \
                * (2 * cf['theta'][iset](T)[0])# + pz.etheta(t,zC[C0],zC[C1],I))

    # Add c-a-a' interactions
            for C in range(len(cations)):
                
                itri = '-'.join([cations[C],iset])
                                
                Gex_nRT = Gex_nRT + anis[:,A0] * anis[:,A1] \
                    * cats[:,C] * cf['psi'][itri](T)[0]

    return Gex_nRT

# Derive activity coefficient function
ln_acfs = egrad(Gex_nRT)

##### TEST AREA ###############################################################
    
T,tots,ions,idf = getIons('ions_in.csv')
mols = np.copy(tots)

Gexs = Gex_nRT(mols,ions)
acfs = np.exp(ln_acfs(mols,ions))
