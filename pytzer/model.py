# pytzer: Pitzer model for chemical activities in aqueous solutions
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)

from autograd.numpy import array, exp, full_like, log, \
                           shape, sqrt, vstack, zeros_like
from autograd.numpy import abs as np_abs
from autograd.numpy import sum as np_sum
from autograd import elementwise_grad as egrad
from .constants import b, Mw, R
from . import props


#==============================================================================
#=================================================== Debye-Hueckel slopes =====


def fG(T,I,cfdict): # from CRP94 Eq. (AI1)

    return -4 * vstack(cfdict.dh['Aosm'](T)[0]) * I * log(1 + b*sqrt(I)) / b


#==============================================================================
#============================================== Pitzer model subfunctions =====


def g(x): # CRP94 Eq. (AI13)
    return 2 * (1 - (1 + x) * exp(-x)) / x**2

def h(x):  # CRP94 Eq. (AI15)
    return (6 - (6 + x*(6 + 3*x + x**2)) * exp(-x)) / x**4


def B(T,I,cfdict,iset): # CRP94 Eq. (AI7)

    b0,b1,b2,_,_,alph1,alph2,_,_ = cfdict.bC[iset](T)

    return b0 + b1 * g(alph1*sqrt(I)) + b2 * g(alph2*sqrt(I))

def CT(T,I,cfdict,iset): # CRP94 Eq. (AI10)

    _,_,_,C0,C1,_,_,omega,_ = cfdict.bC[iset](T)

    return C0 + 4 * C1 * h(omega*sqrt(I))


#==============================================================================
#============================================= Unsymmetrical mixing terms =====


def xij(T,I,z0,z1,cfdict):

    return 6 * z0*z1 * vstack(cfdict.dh['Aosm'](T)[0]) * sqrt(I)

def etheta(T,I,z0,z1,cfdict):

    x00 = xij(T,I,z0,z0,cfdict)
    x01 = xij(T,I,z0,z1,cfdict)
    x11 = xij(T,I,z1,z1,cfdict)

    etheta = z0*z1 * (cfdict.jfunc(x01) \
             - 0.5 * (cfdict.jfunc(x00) + cfdict.jfunc(x11))) / (4 * I)

    return etheta


#==============================================================================
#==================================================== Excess Gibbs energy =====


def Istr(mols,zs):
    return vstack(0.5 * (np_sum(mols * zs**2, 1)))


def Gex_nRT(mols,ions,T,cfdict, Izero=False):

    # Ionic strength etc.
    zs,cations,anions,neutrals = props.charges(ions)
    I = Istr(mols,zs)
    Z = vstack(np_sum(mols * np_abs(zs), 1))

    # Separate out cations, anions and neutrals
    NL = zs == 0
    CL = zs >  0
    AL = zs <  0

    # Concentrations
    neus = mols[:,NL]
    cats = mols[:,CL]
    anis = mols[:,AL]

    # Charges
    zCs = zs[CL]
    zAs = zs[AL]

    # Initialise with zeros
    Gex_nRT = zeros_like(T)

    # Don't do ionic calculations if Izero is requested
    if not Izero:

        # Begin with Debye-Hueckel component
        Gex_nRT = Gex_nRT + fG(T,I,cfdict)

        # Loop through cations
        for C0, cation0 in enumerate(cations):

            # Add c-a interactions
            for A, anion in enumerate(anions):

                iset = '-'.join((cation0,anion))

                Gex_nRT = Gex_nRT + vstack(cats[:,C0] * anis[:,A]) \
                    * (2*B(T,I,cfdict,iset) + Z*CT(T,I,cfdict,iset))

                # Add n-c-a interactions
                for N, neutral in enumerate(neutrals):

                    inca = '-'.join((neutral,iset))

                    Gex_nRT = Gex_nRT \
                            + vstack(neus[:,N] * cats[:,C0] * anis[:,A]) \
                            * cfdict.eta[inca](T)[0]

            # Add c-c' interactions
            for xC1, cation1 in enumerate(cations[C0+1:]):

                C1 = xC1 + C0 + 1

                iset = [cation0,cation1]
                iset.sort()
                iset= '-'.join(iset)

                Gex_nRT = Gex_nRT + vstack(cats[:,C0] * cats[:,C1]) \
                    * 2 * cfdict.theta[iset](T)[0]

                # Unsymmetrical mixing terms
                if zCs[C0] != zCs[C1]:

                    Gex_nRT = Gex_nRT + vstack(cats[:,C0] * cats[:,C1]) \
                        * 2 * etheta(T,I,zCs[C0],zCs[C1],cfdict)

                # Add c-c'-a interactions
                for A, anion in enumerate(anions):

                    itri = '-'.join((iset,anion))

                    Gex_nRT = Gex_nRT + vstack(cats[:,C0] * cats[:,C1] \
                        * anis[:,A]) * cfdict.psi[itri](T)[0]

            # Add n-c interactions
            for N, neutral in enumerate(neutrals):

                inc = '-'.join((neutral,cation0))

                Gex_nRT = Gex_nRT + 2 * vstack(neus[:,N] * cats[:,C0]) \
                                      * cfdict.lambd[inc](T)[0]

        # Loop through anions
        for A0, anion0 in enumerate(anions):

            # Add a-a' interactions
            for xA1, anion1 in enumerate(anions[A0+1:]):

                A1 = xA1 + A0 + 1

                iset = [anion0,anion1]
                iset.sort()
                iset = '-'.join(iset)

                Gex_nRT = Gex_nRT + vstack(anis[:,A0] * anis[:,A1]) \
                    * 2 * cfdict.theta[iset](T)[0]

                # Unsymmetrical mixing terms
                if zAs[A0] != zAs[A1]:

                    Gex_nRT = Gex_nRT + vstack(anis[:,A0] * anis[:,A1]) \
                        * 2 * etheta(T,I,zAs[A0],zAs[A1],cfdict)

                # Add c-a-a' interactions
                for C, cation in enumerate(cations):

                    itri = '-'.join((cation,iset))

                    Gex_nRT = Gex_nRT + vstack(anis[:,A0] * anis[:,A1] \
                        * cats[:,C]) * cfdict.psi[itri](T)[0]

            # Add n-a interactions
            for N, neutral in enumerate(neutrals):

                ina = '-'.join((neutral,anion0))

                Gex_nRT = Gex_nRT + 2 * vstack(neus[:,N] * anis[:,A0]) \
                                      * cfdict.lambd[ina](T)[0]

    # Add neutral-only interactions
    for N0, neutral0 in enumerate(neutrals):
        
        # n-n' including n-n
        for N1, neutral1 in enumerate(neutrals):
            
            inn = '-'.join((neutral0,neutral1))
            
            Gex_nRT = Gex_nRT + 2 * vstack(neus[:,N0] * neus[:,N1]) \
                                  * cfdict.lambd[inn](T)[0]

        # n-n-n
        innn = '-'.join((neutral0,neutral0,neutral0))

        Gex_nRT = Gex_nRT + vstack(neus[:,N0]**3) * cfdict.mu[innn](T)[0]


    return Gex_nRT


#==============================================================================
#=========================================== Solute activity coefficients =====


# Determine activity coefficient function
ln_acfs = egrad(Gex_nRT)

def acfs(mols,ions,T,cfdict, Izero=False):
    return exp(ln_acfs(mols,ions,T,cfdict, Izero))


# Get mean activity coefficient for an M_(nM)X_(nX) electrolyte
def ln_acf2ln_acf_MX(ln_acfM,ln_acfX,nM,nX):
    return (nM * ln_acfM + nX * ln_acfX) / (nM + nX)


#==============================================================================
#=============================== Osmotic coefficient and solvent activity =====


#---------------------------------------------------- Osmotic coefficient -----

# Osmotic coefficient derivative function - single electrolyte
def _osmfunc(ww,mols,ions,T,cfdict, Izero=False):

    mols_ww = array([mols[:,E]/ww.ravel() \
                        for E in range(shape(mols)[1])]).transpose()

    return ww * R * T * Gex_nRT(mols_ww,ions,T,cfdict, Izero)

# Osmotic coefficient derivative - single electrolyte
_osmD = egrad(_osmfunc)

# Osmotic coefficient - single electrolyte
def osm(mols,ions,T,cfdict, Izero=False):

    ww = full_like(T,1, dtype='float64')

    return 1 - _osmD(ww,mols,ions,T,cfdict, Izero) \
        / (R * T * vstack(np_sum(mols,axis=1)))


#------------------------------------------------------- Solvent activity -----

# Convert osmotic coefficient to water activity
def osm2aw(mols,osm):
    return exp(-osm * Mw * vstack(np_sum(mols,axis=1)))

# Convert water activity to osmotic coefficient
def aw2osm(mols,aw):
    return -log(aw) / (Mw * vstack(np_sum(mols,axis=1)))


#==============================================================================
