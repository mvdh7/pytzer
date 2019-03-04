# pytzer: Pitzer model for chemical activities in aqueous solutions
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)

from autograd.numpy import array, exp, full_like, log, \
                           shape, sqrt, vstack, zeros_like
from autograd.numpy import abs as np_abs
from autograd.numpy import any as np_any
from autograd.numpy import sum as np_sum
from autograd import elementwise_grad as egrad
from .constants import b, Mw, R
from . import props


#==============================================================================
#=================================================== Debye-Hueckel slopes =====


def fG(tempK, I, cflib): # from CRP94 Eq. (AI1)

    return -4 * cflib.dh['Aosm'](tempK)[0] * I * log(1 + b*sqrt(I)) / b


#==============================================================================
#============================================== Pitzer model subfunctions =====


def g(x): # CRP94 Eq. (AI13)
    return 2 * (1 - (1 + x) * exp(-x)) / x**2

def h(x):  # CRP94 Eq. (AI15)
    return (6 - (6 + x*(6 + 3*x + x**2)) * exp(-x)) / x**4


def B(tempK, I, cflib, iset): # CRP94 Eq. (AI7)

    b0, b1, b2, _,_, alph1, alph2, _,_ = cflib.bC[iset](tempK)

    return b0 + b1 * g(alph1 * sqrt(I)) + b2 * g(alph2 * sqrt(I))

def CT(tempK, I, cflib, iset): # CRP94 Eq. (AI10)

    _,_,_, C0, C1, _,_, omega, _ = cflib.bC[iset](tempK)

    return C0 + 4 * C1 * h(omega * sqrt(I))


#==============================================================================
#============================================= Unsymmetrical mixing terms =====


def xij(tempK, I, z0, z1, cflib):

    return 6 * z0*z1 * cflib.dh['Aosm'](tempK)[0] * sqrt(I)

def etheta(tempK, I, z0, z1, cflib):

    x00 = xij(tempK, I, z0, z0, cflib)
    x01 = xij(tempK, I, z0, z1, cflib)
    x11 = xij(tempK, I, z1, z1, cflib)

    etheta = z0*z1 * (cflib.jfunc(x01) \
             - 0.5 * (cflib.jfunc(x00) + cflib.jfunc(x11))) / (4 * I)

    return etheta


#==============================================================================
#==================================================== Excess Gibbs energy =====


def Istr(mols, zs):
    return 0.5 * np_sum(mols * zs**2, axis=0)

def Zstr(mols, zs):
    return np_sum(mols * np_abs(zs), axis=0)


def Gex_nRT(mols, ions, tempK, cflib, Izero=False):

    # Ionic strength etc.
    zs, cations, anions, neutrals = props.charges(ions)
    zs = vstack(zs)
    I = Istr(mols, zs)
    Z = Zstr(mols, zs)

    # Split up concentrations
    if np_any(zs == 0):
        neus = vstack([mols[N] for N,_ in enumerate(zs) if zs[N] == 0])
    else:
        neus = []

    if np_any(zs > 0):
        cats = vstack([mols[C] for C,_ in enumerate(zs) if zs[C] > 0])
    else:
        cats = []

    if np_any(zs < 0):
        anis = vstack([mols[A] for A,_ in enumerate(zs) if zs[A] < 0])
    else:
        anis = []

    # Split up charges
    zCs = vstack(zs[zs > 0])
    zAs = vstack(zs[zs < 0])

    # Initialise with zeros
    Gex_nRT = zeros_like(tempK)

    # Don't do ionic calculations if Izero is requested
    if not Izero:

        # Begin with Debye-Hueckel component
        Gex_nRT = Gex_nRT + fG(tempK, I, cflib)

        # Loop through cations
        for C0, cation0 in enumerate(cations):

            # Add c-a interactions
            for A, anion in enumerate(anions):

                iset = '-'.join((cation0,anion))

                Gex_nRT = Gex_nRT + cats[C0] * anis[A] \
                    * (2*B(tempK, I, cflib, iset) + Z*CT(tempK, I, cflib, iset))

            # Add c-c' interactions
            for xC1, cation1 in enumerate(cations[C0+1:]):

                C1 = xC1 + C0 + 1

                iset = [cation0, cation1]
                iset.sort()
                iset= '-'.join(iset)

                Gex_nRT = Gex_nRT + cats[C0] * cats[C1] \
                    * 2 * cflib.theta[iset](tempK)[0]

                # Unsymmetrical mixing terms
                if zCs[C0] != zCs[C1]:

                    Gex_nRT = Gex_nRT + cats[C0] * cats[C1] \
                        * 2 * etheta(tempK, I, zCs[C0], zCs[C1], cflib)

                # Add c-c'-a interactions
                for A, anion in enumerate(anions):

                    itri = '-'.join((iset, anion))

                    Gex_nRT = Gex_nRT + cats[C0] * cats[C1] * anis[A] \
                        * cflib.psi[itri](tempK)[0]

        # Loop through anions
        for A0, anion0 in enumerate(anions):

            # Add a-a' interactions
            for xA1, anion1 in enumerate(anions[A0+1:]):

                A1 = xA1 + A0 + 1

                iset = [anion0, anion1]
                iset.sort()
                iset = '-'.join(iset)

                Gex_nRT = Gex_nRT + anis[A0] * anis[A1] \
                    * 2 * cflib.theta[iset](tempK)[0]

                # Unsymmetrical mixing terms
                if zAs[A0] != zAs[A1]:

                    Gex_nRT = Gex_nRT + anis[A0] * anis[A1] \
                        * 2 * etheta(tempK, I, zAs[A0], zAs[A1], cflib)

                # Add c-a-a' interactions
                for C, cation in enumerate(cations):

                    itri = '-'.join((cation, iset))

                    Gex_nRT = Gex_nRT + anis[A0] * anis[A1] * cats[C] \
                        * cflib.psi[itri](tempK)[0]

    # Add neutral interactions
    for N0, neutral0 in enumerate(neutrals):

        # Add n-c interactions
        for C, cation in enumerate(cations):

            inc = '-'.join((neutral0, cation))

            Gex_nRT = Gex_nRT + neus[N0] * cats[C] \
                * 2 * cflib.lambd[inc](tempK)[0]

            # Add n-c-a interactions
            for A, anion in enumerate(anions):

                inca = '-'.join((inc, anion))

                Gex_nRT = Gex_nRT + neus[N0] * cats[C] * anis[A] \
                    * cflib.zeta[inca](tempK)[0]

        # Add n-a interactions
        for A, anion in enumerate(anions):

            ina = '-'.join((neutral0, anion))

            Gex_nRT = Gex_nRT + neus[N0] * anis[A] \
                * 2 * cflib.lambd[ina](tempK)[0]

        # n-n' excluding n-n
        for N1, neutral1 in enumerate(neutrals[N0+1:]):

            inn = [neutral0, neutral1]
            inn.sort()
            inn = '-'.join(inn)

            Gex_nRT = Gex_nRT + neus[N0] * neus[N1] \
                * 2 * cflib.lambd[inn](tempK)[0]

        # n-n
        inn = '-'.join((neutral0, neutral0))

        Gex_nRT = Gex_nRT + neus[N0]**2 * cflib.lambd[inn](tempK)[0]

        # n-n-n
        innn = '-'.join((neutral0,neutral0,neutral0))

        Gex_nRT = Gex_nRT + neus[N0]**3 * cflib.mu[innn](tempK)[0]


    return Gex_nRT


#==============================================================================
#=========================================== Solute activity coefficients =====


# Determine activity coefficient function
ln_acfs = egrad(Gex_nRT)

def acfs(mols, ions, tempK, cflib, Izero=False):
    return exp(ln_acfs(mols, ions, tempK, cflib, Izero))


# Get mean activity coefficient for an M_(nM)X_(nX) electrolyte
def ln_acf2ln_acf_MX(ln_acfM, ln_acfX, nM, nX):
    return (nM * ln_acfM + nX * ln_acfX) / (nM + nX)


#==============================================================================
#=============================== Osmotic coefficient and solvent activity =====


#---------------------------------------------------- Osmotic coefficient -----

# Osmotic coefficient
def osm(mols, ions, tempK, cflib, Izero=False):

    ww = full_like(tempK,1.0)

    return 1 - egrad(
        lambda ww: ww * Gex_nRT(mols/ww, ions, tempK, cflib, Izero))(ww) \
        / np_sum(mols, axis=0)


#--------------------------------------------------------- Water activity -----

# Water activity - direct
def lnaw(mols, ions, tempK, cflib, Izero=False):

    ww = full_like(tempK, 1.0)

    return (egrad(
        lambda ww: ww * Gex_nRT(mols/ww, ions, tempK, cflib, Izero))(ww) \
        - np_sum(mols, axis=0)) * Mw


def aw(mols, ions, tempK, cflib, Izero=False):

    return exp(lnaw(mols, ions, tempK, cflib, Izero))


#------------------------------------------------------------ Conversions -----

# Convert osmotic coefficient to water activity
def osm2aw(mols, osm):
    return exp(-osm * Mw * np_sum(mols, axis=0))

# Convert water activity to osmotic coefficient
def aw2osm(mols, aw):
    return -log(aw) / (Mw * np_sum(mols, axis=0))


#==============================================================================
