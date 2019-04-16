# Pytzer: Pitzer model for chemical activities in aqueous solutions
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)

from autograd.numpy import array, log, size, sqrt, transpose, zeros
from autograd.numpy import abs as np_abs
from . import properties
from .cflibs import Seawater
from .constants import b
from .model import g, h
from .jfuncs import P75_eq47 as jfunc

"""Matrix version of the Pitzer model."""

def Istr(mols, zs):
    """Calculate the ionic strength."""
    return mols @ transpose(zs**2) / 2

def Zstr(mols, zs):
    """Calculate the Z function."""
    return mols @ transpose(np_abs(zs))

def fG(Aosm, I):
    """Calculate the Debye-Hueckel component of the excess Gibbs energy."""
    return -4*I*Aosm * log(1 + b*sqrt(I)) / b

def B(cats, anis, I, b0, b1, b2, alph1, alph2):
    """B function following CRP94 Eq. (AI7)."""
    Bmx = b0 + b1*g(alph1*sqrt(I)) + b2*g(alph2*sqrt(I))
    return cats @ Bmx @ transpose(anis)
    
def CT(cats, anis, I, C0, C1, omega):
    """CT function following CRP94 Eq. (AI10)."""
    CTmx = C0 + 4*C1*h(omega*sqrt(I))
    return cats @ CTmx @ transpose(anis)

def xij(Aosm, I, zs):
    """xij function for unsymmetrical mixing."""
    return 6*Aosm*sqrt(I)*(transpose(zs) @ zs)

def xi(Aosm, I, zs):
    """xi function for unsymmetrical mixing."""
    return 6*Aosm*sqrt(I) * zs**2

def xj(Aosm, I, zs):
    """xj function for unsymmetrical mixing."""
    return 6*Aosm*sqrt(I) * transpose(zs**2)

def etheta(Aosm, I, zs):
    """etheta function for unsymmetrical mixing."""
    x01 = xij(Aosm, I, zs)
    x00 = xi(Aosm, I, zs)
    x11 = xj(Aosm, I, zs)
    return (transpose(zs) @ zs) * (jfunc(x01) 
        - (jfunc(x00) + jfunc(x11))/2) / (4*I)

def Gex_nRT(mols, zs, Aosm, b0mx, b1mx, b2mx, C0mx, C1mx,
        alph1mx, alph2mx, omegamx, thetamxcc, thetamxaa):
    """Calculate the excess Gibbs energy of a solution."""
    I = Istr(mols, zs)
    Z = Zstr(mols, zs)
    cats = array([mols[zs > 0]])
    anis = array([mols[zs < 0]])
    zcats = array([zs[zs > 0]])
    zanis = array([zs[zs < 0]])
    return fG(Aosm, I) \
        + 2*B(cats, anis, I, b0mx, b1mx, b2mx, alph1mx, alph2mx) \
        + Z*CT(cats, anis, I, C0mx, C1mx, omegamx) \
        + cats @ (thetamxcc + etheta(Aosm, I, zcats)) @ transpose(cats) \
        + anis @ (thetamxaa + etheta(Aosm, I, zanis)) @ transpose(anis)

def assemble(ions, tempK, pres, cflib=Seawater):
    """Assemble coefficient matrices."""
    zs, cations, anions, _ = properties.charges(ions)
    zs = transpose(zs)
    Aosm = cflib.dh['Aosm'](tempK, pres)[0][0]
    b0mx = zeros((size(cations), size(anions)))
    b1mx = zeros((size(cations), size(anions)))
    b2mx = zeros((size(cations), size(anions)))
    C0mx = zeros((size(cations), size(anions)))
    C1mx = zeros((size(cations), size(anions)))
    alph1mx = zeros((size(cations), size(anions)))
    alph2mx = zeros((size(cations), size(anions)))
    omegamx = zeros((size(cations), size(anions)))
    thetamxcc = zeros((size(cations), size(cations)))
    thetamxaa = zeros((size(anions), size(anions)))
    for CX, cationx in enumerate(cations):
        for A, anion in enumerate(anions):
            iset = '-'.join((cationx, anion))
            b0mx[CX, A], b1mx[CX, A], b2mx[CX, A], C0mx[CX, A], C1mx[CX, A], \
                    alph1mx[CX, A], alph2mx[CX, A], omegamx[CX, A], _ \
                = cflib.bC[iset](tempK, pres)
        for xCY, cationy in enumerate(cations[CX+1:]):
            CY = xCY + CX + 1
            iset = [cationx, cationy]
            iset.sort()
            iset= '-'.join(iset)
            thetamxcc[CX, CY] = thetamxcc[CY, CX] \
                = cflib.theta[iset](tempK, pres)[0]
    for AX, anionx in enumerate(anions):
        for xAY, aniony in enumerate(anions[AX+1:]):
            AY = xAY + AX + 1
            iset = [anionx, aniony]
            iset.sort()
            iset = '-'.join(iset)
            thetamxaa[AX, AY] = thetamxaa[AY, AX] \
                = cflib.theta[iset](tempK, pres)[0]
    return zs, Aosm, b0mx, b1mx, b2mx, C0mx, C1mx, alph1mx, alph2mx, omegamx, \
        thetamxcc, thetamxaa