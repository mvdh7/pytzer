# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
"""Calculate solution properties using the Pitzer model."""
from autograd.numpy import exp, full_like, log, sqrt, vstack, zeros_like
from autograd.numpy import abs as np_abs
from autograd.numpy import any as np_any
from autograd.numpy import sum as np_sum
from autograd import elementwise_grad as egrad
from .constants import b, Mw
from .libraries import Seawater
from . import properties

# Debye-Hueckel slope
def fG(tempK, pres, I, prmlib): # from CRP94 Eq. (AI1)
    """Calculate the Debye-Hueckel component of the excess Gibbs energy."""
    return -4 * prmlib.dh['Aosm'](tempK, pres)[0] * I * log(1 + b*sqrt(I)) / b

# Pitzer model subfunctions
def g(x):
    """g function following CRP94 Eq. (AI13)."""
    return 2*(1 - (1 + x)*exp(-x)) / x**2

def h(x):
    """h function following CRP94 Eq. (AI15)."""
    return (6 - (6 + x*(6 + 3*x + x**2)) * exp(-x)) / x**4

def B(I, b0, b1, b2, alph1, alph2):
    """B function following CRP94 Eq. (AI7)."""
    return b0 + b1*g(alph1*sqrt(I)) + b2*g(alph2*sqrt(I))

def CT(I, C0, C1, omega):
    """CT function following CRP94 Eq. (AI10)."""
    return C0 + 4*C1*h(omega*sqrt(I))

# Unsymmetrical mixing terms
def xij(tempK, pres, I, z0, z1, prmlib):
    """xij function for unsymmetrical mixing."""
    return 6 * z0*z1 * prmlib.dh['Aosm'](tempK, pres)[0] * sqrt(I)

def etheta(tempK, pres, I, z0, z1, prmlib):
    """etheta function for unsymmetrical mixing."""
    x00 = xij(tempK, pres, I, z0, z0, prmlib)
    x01 = xij(tempK, pres, I, z0, z1, prmlib)
    x11 = xij(tempK, pres, I, z1, z1, prmlib)
    etheta = z0*z1 * (prmlib.jfunc(x01)
        - 0.5 * (prmlib.jfunc(x00) + prmlib.jfunc(x11))) / (4 * I)
    return etheta

# Ionic strength
def Istr(mols, zs):
    """Calculate the ionic strength."""
    return 0.5 * np_sum(mols * zs**2, axis=0)

def Zstr(mols, zs):
    """Calculate the Z function."""
    return np_sum(mols * np_abs(zs), axis=0)

# Excess Gibbs energy
def Gex_nRT(mols, ions, tempK, pres, prmlib=Seawater, Izero=False):
    """Calculate the excess Gibbs energy of a solution."""
    # Note that oceanographers record ocean pressure as only due to the water,
    # so at the sea surface pressure = 0 dbar, but the atmospheric pressure
    # should also be taken into account for this model
    # Ionic strength etc.
    zs, cations, anions, neutrals = properties.charges(ions)
    zs = vstack(zs)
    I = Istr(mols, zs)
    Z = Zstr(mols, zs)
    # Split up concentrations
    if np_any(zs == 0):
        neus = vstack([mols[N] for N, _ in enumerate(zs) if zs[N] == 0])
    else:
        neus = []
    if np_any(zs > 0):
        cats = vstack([mols[C] for C, _ in enumerate(zs) if zs[C] > 0])
    else:
        cats = []
    if np_any(zs < 0):
        anis = vstack([mols[A] for A, _ in enumerate(zs) if zs[A] < 0])
    else:
        anis = []
    # Initialise with zeros
    Gex_nRT = zeros_like(tempK)
    # Don't do ionic calculations if Izero is requested
    if not Izero:
        # Split up charges
        zCs = vstack(zs[zs > 0])
        zAs = vstack(zs[zs < 0])
        # Begin with Debye-Hueckel component
        Gex_nRT = Gex_nRT + fG(tempK, pres, I, prmlib)
        # Loop through cations
        for CX, cationx in enumerate(cations):
            # Add c-a interactions
            for A, anion in enumerate(anions):
                iset = '-'.join((cationx, anion))
                b0, b1, b2, C0, C1, alph1, alph2, omega, _ = \
                    prmlib.bC[iset](tempK, pres)
                Gex_nRT = (Gex_nRT + cats[CX]*anis[A] *
                    (2*B(I, b0, b1, b2, alph1, alph2) +
                        Z*CT(I, C0, C1, omega)))
            # Add c-c' interactions
            for xCY, cationy in enumerate(cations[CX+1:]):
                CY = xCY + CX + 1
                iset = [cationx, cationy]
                iset.sort()
                iset= '-'.join(iset)
                Gex_nRT = (Gex_nRT + cats[CX]*cats[CY] *
                    2*prmlib.theta[iset](tempK, pres)[0])
                # Unsymmetrical mixing terms
                if zCs[CX] != zCs[CY]:
                    Gex_nRT = (Gex_nRT + cats[CX]*cats[CY] *
                        2*etheta(tempK, pres, I, zCs[CX], zCs[CY], prmlib))
                # Add c-c'-a interactions
                for A, anion in enumerate(anions):
                    itri = '-'.join((iset, anion))
                    Gex_nRT = (Gex_nRT + cats[CX]*cats[CY]*anis[A] *
                        prmlib.psi[itri](tempK, pres)[0])
        # Loop through anions
        for AX, anionx in enumerate(anions):
            # Add a-a' interactions
            for xAY, aniony in enumerate(anions[AX+1:]):
                AY = xAY + AX + 1
                iset = [anionx, aniony]
                iset.sort()
                iset = '-'.join(iset)
                Gex_nRT = (Gex_nRT + anis[AX]*anis[AY] *
                    2*prmlib.theta[iset](tempK, pres)[0])
                # Unsymmetrical mixing terms
                if zAs[AX] != zAs[AY]:
                    Gex_nRT = (Gex_nRT + anis[AX]*anis[AY] *
                        2*etheta(tempK, pres, I, zAs[AX], zAs[AY], prmlib))
                # Add c-a-a' interactions
                for C, cation in enumerate(cations):
                    itri = '-'.join((cation, iset))
                    Gex_nRT = (Gex_nRT + anis[AX]*anis[AY]*cats[C] *
                        prmlib.psi[itri](tempK, pres)[0])
    # Add neutral interactions
    for NX, neutralx in enumerate(neutrals):
        # Add n-c interactions
        for C, cation in enumerate(cations):
            inc = '-'.join((neutralx, cation))
            Gex_nRT = (Gex_nRT + neus[NX]*cats[C] *
                2*prmlib.lambd[inc](tempK, pres)[0])
            # Add n-c-a interactions
            for A, anion in enumerate(anions):
                inca = '-'.join((inc, anion))
                Gex_nRT = (Gex_nRT + neus[NX]*cats[C]*anis[A] *
                    prmlib.zeta[inca](tempK, pres)[0])
        # Add n-a interactions
        for A, anion in enumerate(anions):
            ina = '-'.join((neutralx, anion))
            Gex_nRT = (Gex_nRT + neus[NX]*anis[A] *
                2*prmlib.lambd[ina](tempK, pres)[0])
        # n-n' excluding n-n
        for xNY, neutraly in enumerate(neutrals[NX+1:]):
            NY = xNY + NX + 1
            inn = [neutralx, neutraly]
            inn.sort()
            inn = '-'.join(inn)
            Gex_nRT = (Gex_nRT + neus[NX]*neus[NY] *
                2*prmlib.lambd[inn](tempK, pres)[0])
        # n-n
        inn = '-'.join((neutralx, neutralx))
        Gex_nRT = Gex_nRT + neus[NX]**2 * prmlib.lambd[inn](tempK, pres)[0]
        # n-n-n
        innn = '-'.join((neutralx,neutralx,neutralx))
        Gex_nRT = Gex_nRT + neus[NX]**3 * prmlib.mu[innn](tempK, pres)[0]
    return Gex_nRT

# Solute activity coefficients
def ln_acfs(mols, ions, tempK, pres, prmlib=Seawater, Izero=False):
    """Calculate the natural logarithms of the activity coefficients
    of all solutes.
    """
    return egrad(Gex_nRT)(mols, ions, tempK, pres, prmlib, Izero)

def acfs(mols, ions, tempK, pres, prmlib=Seawater, Izero=False):
    """Calculate the activity coefficients of all solutes."""
    return exp(ln_acfs(mols, ions, tempK, pres, prmlib, Izero))

def ln_acf2ln_acf_MX(ln_acfM, ln_acfX, nM, nX):
    """Calculate the mean activity coefficient for an electrolyte."""
    return (nM*ln_acfM + nX*ln_acfX)/(nM + nX)

# Osmotic coefficient
def osm(mols, ions, tempK, pres, prmlib=Seawater, Izero=False):
    """Calculate the osmotic coefficient."""
    ww = full_like(tempK, 1.0)
    return (1 - egrad(lambda ww:
        ww*Gex_nRT(mols/ww, ions, tempK, pres, prmlib, Izero))(ww)
            /np_sum(mols, axis=0))

# Water activity
def lnaw(mols, ions, tempK, pres, prmlib=Seawater, Izero=False):
    """Calculate the natural log of the water activity."""
    ww = full_like(tempK, 1.0)
    return (egrad(lambda ww:
        ww*Gex_nRT(mols/ww, ions, tempK, pres, prmlib, Izero))(ww)
        - np_sum(mols, axis=0))*Mw

def aw(mols, ions, tempK, pres, prmlib=Seawater, Izero=False):
    """Calculate the water activity."""
    return exp(lnaw(mols, ions, tempK, pres, prmlib, Izero))

# Conversions
def osm2aw(mols, osm):
    """Convert osmotic coefficient to water activity."""
    return exp(-osm*Mw*np_sum(mols, axis=0))

def aw2osm(mols, aw):
    """Convert water activity to osmotic coefficient."""
    return -log(aw)/(Mw*np_sum(mols, axis=0))
