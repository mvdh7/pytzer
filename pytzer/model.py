from autograd.numpy import array, exp, full_like, log, shape, sqrt, vstack
from autograd.numpy import abs as np_abs
from autograd.numpy import sum as np_sum
from autograd import elementwise_grad as egrad
from .constants import b, Mw, R
from . import props


##### DEBYE-HUECKEL SLOPES ####################################################

def fG(T,I,cf): # from CRP94 Eq. (AI1)
        
    return -4 * vstack(cf.dh['Aosm'](T)[0]) * I * log(1 + b*sqrt(I)) / b


##### PITZER MODEL SUBFUNCTIONS ###############################################

def g(x): # CRP94 Eq. (AI13)
    return 2 * (1 - (1 + x) * exp(-x)) / x**2

def h(x):  # CRP94 Eq. (AI15)
    return (6 - (6 + x*(6 + 3*x + x**2)) * exp(-x)) / x**4


def B(T,I,cf,iset): # CRP94 Eq. (AI7)
    
    b0,b1,b2,_,_,a1,a2,_,_ = cf.bC[iset](T)

    return b0 + b1 * g(a1*sqrt(I)) + b2 * g(a2*sqrt(I))

def CT(T,I,cf,iset): # CRP94 Eq. (AI10)
    
    _,_,_,C0,C1,_,_,o,_ = cf.bC[iset](T)
    
    return C0 + 4 * C1 * h(o*sqrt(I))


##### UNSYMMETRIC MIXING ######################################################
    
def xij(T,I,z0,z1,cf):
    
    return 6 * z0*z1 * vstack(cf.dh['Aosm'](T)[0]) * sqrt(I)

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
    zs = props.charges(ions)[0]
    I = vstack(0.5 * (np_sum(mols * zs**2, 1)))
    Z = vstack(np_sum(mols * np_abs(zs), 1))
    
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

            Gex_nRT = Gex_nRT + vstack(cats[:,C] * anis[:,A]) \
                * (2*B(T,I,cf,iset) + Z*CT(T,I,cf,iset))
                
    # Add c-c' interactions
    for C0 in range(len(cations)):
        for C1 in range(C0+1,len(cations)):
            
            iset = [cations[C0],cations[C1]]
            iset.sort()
            iset= '-'.join(iset)
            
            Gex_nRT = Gex_nRT + vstack(cats[:,C0] * cats[:,C1]) \
                * 2 * cf.theta[iset](T)[0]
                
            if zCs[C0] != zCs[C1]:
                
                Gex_nRT = Gex_nRT + vstack(cats[:,C0] * cats[:,C1]) \
                    * 2 * etheta(T,I,zCs[C0],zCs[C1],cf)
                
    # Add c-c'-a interactions
            for A in range(len(anions)):
                
                itri = '-'.join([iset,anions[A]])
                                
                Gex_nRT = Gex_nRT + vstack(cats[:,C0] * cats[:,C1] \
                    * anis[:,A]) * cf.psi[itri](T)[0]

    # Add a-a' interactions
    for A0 in range(len(anions)):
        for A1 in range(A0+1,len(anions)):
            
            iset = [anions[A0],anions[A1]]
            iset.sort()
            iset= '-'.join(iset)
            
            Gex_nRT = Gex_nRT + vstack(anis[:,A0] * anis[:,A1]) \
                * 2 * cf.theta[iset](T)[0]
                
            if zAs[A0] != zAs[A1]:
                
                Gex_nRT = Gex_nRT + vstack(anis[:,A0] * anis[:,A1]) \
                    * 2 * etheta(T,I,zAs[A0],zAs[A1],cf)

    # Add c-a-a' interactions
            for C in range(len(cations)):
                
                itri = '-'.join([cations[C],iset])
                                
                Gex_nRT = Gex_nRT + vstack(anis[:,A0] * anis[:,A1] \
                    * cats[:,C]) * cf.psi[itri](T)[0]

    return Gex_nRT



##### SOLUTE ACTIVITY COEFFICIENT #############################################


# Determine activity coefficient function
ln_acfs = egrad(Gex_nRT)

def acfs(mols,ions,T,cf):
    
    return exp(ln_acfs(mols,ions,T,cf))


# Get mean activity coefficient for an M_(nM)X_(nX) electrolyte
def ln_acf2ln_acf_MX(ln_acfM,ln_acfX,nM,nX):
    
    return (nM * ln_acfM + nX * ln_acfX) / (nM + nX)



##### OSMOTIC COEFFICIENT and SOLVENT ACTIVITY ################################


# Osmotic coefficient derivative function - single electrolyte
def osmfunc(ww,mols,ions,T,cf):
    
    mols_ww = array([mols[:,E]/ww.ravel() \
                        for E in range(shape(mols)[1])]).transpose()
    
    return ww * R * T * Gex_nRT(mols_ww,ions,T,cf)


# Osmotic coefficient derivative - single electrolyte
osmD = egrad(osmfunc)


# Osmotic coefficient - single electrolyte
def osm(mols,ions,T,cf):
    
    ww = full_like(T,1, dtype='float64')
    
    return 1 - osmD(ww,mols,ions,T,cf) \
        / (R * T * vstack(np_sum(mols,axis=1)))


# Convert osmotic coefficient to water activity
def osm2aw(mols,osm):
    return exp(-osm * Mw * vstack(np_sum(mols,axis=1)))


# Convert water activity to osmotic coefficient
def aw2osm(mols,aw):
    return -log(aw) / (Mw * vstack(np_sum(mols,axis=1)))
