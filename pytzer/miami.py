import autograd.numpy as np
from autograd import elementwise_grad as egrad
from scipy.misc import derivative
import pandas as pd
from .constants import b, Mw, R

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
    ions = np.array(idf_tots.keys())
       
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
    
def Gex_nRT(mols,ions,T,cf):
    
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

# Derive osmotic coefficient function
def osm(mols,ions,T,cf):
    osmD = np.full_like(T,np.nan)
    for i in range(len(T)):
        osmD[i] = derivative(lambda ww: 
            ww * R*T[i] * Gex_nRT(np.array([mols[i,:]/ww]),ions,T[i],cf),
            np.array([1.]), dx=1e-8)[0]
    osm = 1 - osmD / (R * T * (np.sum(mols,axis=1)))
    
    return osm

# Convert osmotic coefficient to water activity
def osm2aw(mols,osm):
    return np.exp(-osm * Mw * np.sum(mols,axis=1))

# autograd doesn't seem to work due to broadcasting issues for osm derivative
#  hence scipy derivation here for now (which does work great)
#fx_osmD = egrad(lambda ww,Tw: ww * R*Tw * Gex_nRT(mols/ww,ions,Tw))
