import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd.extend import primitive, defvjp
#from scipy.misc import derivative
from .constants import b, Mw, R

##### IONIC CHARGES ###########################################################

def getCharges(ions):

    z = {}
    
    z['Ba'  ] = np.float_(+0)
    z['Ca'  ] = np.float_(+2)
    z['H'   ] = np.float_(+1)
    z['K'   ] = np.float_(+1)
    z['Mg'  ] = np.float_(+2)
    z['Na'  ] = np.float_(+1)
    z['Zn'  ] = np.float_(+2)

    z['Br'  ] = np.float_(-1)
    z['Cl'  ] = np.float_(-1)    
    z['OH'  ] = np.float_(-1)
    z['HSO4'] = np.float_(-1)
    z['SO4' ] = np.float_(-2)
    
    return np.array([z[ion] for ion in ions])


##### DEBYE-HUECKEL SLOPES ####################################################

def fG(T,I,cf): # from CRP94 Eq. (AI1)
    
    # T and I inputs must have the same shape for correct broadcasting
    
    # Override autograd differentiation of Aosm wrt. T by using AH, if AH is
    #  present in cf.dh
    if 'AH' in cf.dh.keys():
        @primitive
        def Aosm(T):
            return cf.dh['Aosm'](T)[0]
        def Aosm_vjp(ans,T): # P91 Ch. 3 Eq. (84)
            return lambda g: g * cf.dh['AH'](T) / (4 * R * T**2)   
        defvjp(Aosm,Aosm_vjp)
    
    # Just use autograd if AH is not provided
    else:
        Aosm = lambda T: cf.dh['Aosm'](T)[0]
    
    return -4 * np.vstack(Aosm(T)) * I * np.log(1 + b*np.sqrt(I)) / b

###
dfG_T_dT = egrad(lambda T,I,cf: fG(T,I,cf) * R) # for testing purposes only

def fL(T,I,cf,nu): # for testing purposes only

    return nu * cf.dh['AH'](T) * np.log(1 + b*np.sqrt(I)) / (2*b)


##### PITZER MODEL SUBFUNCTIONS ###############################################

def g(x): # CRP94 Eq. (AI13)
    return 2 * (1 - (1 + x) * np.exp(-x)) / x**2

def h(x):  # CRP94 Eq. (AI15)
    return (6 - (6 + x*(6 + 3*x + x**2)) * np.exp(-x)) / x**4


def B(T,I,cf,iset): # CRP94 Eq. (AI7)
    
    b0,b1,b2,_,_,a1,a2,_,_ = cf.bC[iset](T)

    return b0 + b1 * g(a1*np.sqrt(I)) + b2 * g(a2*np.sqrt(I))

def CT(T,I,cf,iset): # CRP94 Eq. (AI10)
    
    _,_,_,C0,C1,_,_,o,_ = cf.bC[iset](T)
    
    return C0 + 4 * C1 * h(o*np.sqrt(I))


##### UNSYMMETRIC MIXING ######################################################
    
def xij(T,I,z0,z1,cf):
    
    return 6 * z0*z1 * np.vstack(cf.dh['Aosm'](T)[0]) * np.sqrt(I)

def etheta(T,I,z0,z1,cf):
    
    x00 = xij(T,I,z0,z0,cf)
    x01 = xij(T,I,z0,z1,cf)
    x11 = xij(T,I,z1,z1,cf)
    
    etheta = z0*z1 * (cf.jfunc(x01)[0] \
                      - 0.5 * (cf.jfunc(x00)[0] + cf.jfunc(x11)[0])) / (4 * I)
    
    return etheta


##### EXCESS GIBBS ENERGY #####################################################
    
def Gex_nRT(mols,ions,T,cf):
    
    # Ionic strength etc.
    zs = getCharges(ions)
    I = np.vstack(0.5 * (np.sum(mols * zs**2, 1)))
    Z = np.vstack(np.sum(mols * np.abs(zs), 1))
    
    # Separate cations and anions
    CL = zs > 0
    cats    = mols[:,CL]
    cations = ions[  CL]
    zCs     = zs  [  CL]
    AL = zs < 0
    anis    = mols[:,AL]
    anions  = ions[  AL]
    zAs     = zs  [  AL]
    
    # Begin with Debye-Hueckel component
    Gex_nRT = fG(T,I,cf)
    
    # Add c-a interactions
    for C,cation in enumerate(cations):
        for A,anion in enumerate(anions):

            iset= '-'.join([cation,anion])

            Gex_nRT = Gex_nRT + np.vstack(cats[:,C] * anis[:,A]) \
                * (2*B(T,I,cf,iset) + Z*CT(T,I,cf,iset))
                
    # Add c-c' interactions
    for C0 in range(len(cations)):
        for C1 in range(C0+1,len(cations)):
            
            iset = [cations[C0],cations[C1]]
            iset.sort()
            iset= '-'.join(iset)
            
            Gex_nRT = Gex_nRT + np.vstack(cats[:,C0] * cats[:,C1]) \
                * 2 * cf.theta[iset](T)[0]
                
            if zCs[C0] != zCs[C1]:
                
                Gex_nRT = Gex_nRT + np.vstack(cats[:,C0] * cats[:,C1]) \
                    * 2 * etheta(T,I,zCs[C0],zCs[C1],cf)
                
    # Add c-c'-a interactions
            for A in range(len(anions)):
                
                itri = '-'.join([iset,anions[A]])
                                
                Gex_nRT = Gex_nRT + np.vstack(cats[:,C0] * cats[:,C1] \
                    * anis[:,A]) * cf.psi[itri](T)[0]

    # Add a-a' interactions
    for A0 in range(len(anions)):
        for A1 in range(A0+1,len(anions)):
            
            iset = [anions[A0],anions[A1]]
            iset.sort()
            iset= '-'.join(iset)
            
            Gex_nRT = Gex_nRT + np.vstack(anis[:,A0] * anis[:,A1]) \
                * 2 * cf.theta[iset](T)[0]
                
            if zAs[A0] != zAs[A1]:
                
                Gex_nRT = Gex_nRT + np.vstack(anis[:,A0] * anis[:,A1]) \
                    * 2 * etheta(T,I,zAs[A0],zAs[A1],cf)

    # Add c-a-a' interactions
            for C in range(len(cations)):
                
                itri = '-'.join([cations[C],iset])
                                
                Gex_nRT = Gex_nRT + np.vstack(anis[:,A0] * anis[:,A1] \
                    * cats[:,C]) * cf.psi[itri](T)[0]

    return Gex_nRT

##### SOLUTE ACTIVITY COEFFICIENT #############################################

# Determine activity coefficient function
ln_acfs = egrad(Gex_nRT)

def acfs(mols,ions,T,cf):
    
    return np.exp(ln_acfs(mols,ions,T,cf))

# Get mean activity coefficient for an M_(nM)X_(nX) electrolyte
def ln_acf2ln_acf_MX(ln_acfM,ln_acfX,nM,nX):
    
    return (nM * ln_acfM + nX * ln_acfX) / (nM + nX)

##### OSMOTIC COEFFICIENT and SOLVENT ACTIVITY ################################

# Osmotic coefficient derivative function - single electrolyte
def osmfunc(ww,mols,ions,T,cf):
    
    mols_ww = np.array([mols[:,E]/ww.ravel() \
                        for E in range(np.shape(mols)[1])]).transpose()
    
    return ww * R * T * Gex_nRT(mols_ww,ions,T,cf)

# Osmotic coefficient derivative - single electrolyte
osmD = egrad(osmfunc)

# Osmotic coefficient - single electrolyte
def osm(mols,ions,T,cf):
    
    ww = np.full_like(T,1, dtype='float64')
    
    return 1 - osmD(ww,mols,ions,T,cf) \
        / (R * T * np.vstack(np.sum(mols,axis=1)))

## Osmotic coefficient function - scipy derivative version
#def osm(mols,ions,T,cf):
#    osmD = np.full_like(T,np.nan)
#    for i in range(len(T)):
#        osmD[i] = derivative(lambda ww: 
#            ww * R*T[i] * Gex_nRT(np.array([mols[i,:]/ww]),ions,T[i],cf),
#            np.array([1.]), dx=1e-8)[0]
#    osm = 1 - osmD / (R * T * (np.sum(mols,axis=1)))
#    
#    return osm

# Convert osmotic coefficient to water activity
def osm2aw(mols,osm):
    return np.exp(-osm * Mw * np.vstack(np.sum(mols,axis=1)))

# Convert water activity to osmotic coefficient
def aw2osm(mols,aw):
    return -np.log(aw) / (Mw * np.vstack(np.sum(mols,axis=1)))

##### ENTHALPY and HEAT CAPACITY ##############################################
    
# Gex/T differential wrt. T
dGex_T_dT = egrad(Gex_nRT, argnum=2)

# Apparent relative molal enthalpy (single electrolyte)
def Lapp(tot,nC,nA,ions,T,cf):
    
    mC = (tot * nC).ravel()
    mA = (tot * nA).ravel()
    mols = np.vstack((mC,mA)).transpose()
    
    return -T**2 * dGex_T_dT(mols,ions,T,cf) * R / tot

# Apparent relative molal heat capacity (i.e. dL/dT; does not include Cp0 term)
Cpapp = egrad(Lapp, argnum=4)
