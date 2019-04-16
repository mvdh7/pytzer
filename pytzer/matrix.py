# Pytzer: Pitzer model for chemical activities in aqueous solutions
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)

from autograd.numpy import array, log, size, sqrt, transpose, zeros
from autograd.numpy import abs as np_abs
from . import props
from .cflibs import Seawater
from .constants import b
from .model import g, h

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

def Gex_nRT(mols, zs, Aosm, b0mx, b1mx, b2mx, C0mx, C1mx,
        alph1mx, alph2mx, omegamx):
    I = Istr(mols, zs)
    Z = Zstr(mols, zs)
    cats = array([mols[zs > 0]])
    anis = array([mols[zs < 0]])
    return fG(Aosm, I) \
        + 2*B(cats, anis, I, b0mx, b1mx, b2mx, alph1mx, alph2mx) \
        + Z*CT(cats, anis, I, C0mx, C1mx, omegamx)

def assemble(ions, tempK, pres, cflib=Seawater):
    """Assemble coefficient matrices."""
    zs, cations, anions, _ = props.charges(ions)
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
    for C, cation in enumerate(cations):
        for A, anion in enumerate(anions):
            ca = '-'.join((cation, anion))
            b0mx[C, A], b1mx[C, A], b2mx[C, A], C0mx[C, A], C1mx[C, A], \
                    alph1mx[C, A], alph2mx[C, A], omegamx[C, A], _ \
                = cflib.bC[ca](tempK, pres)
    return zs, Aosm, b0mx, b1mx, b2mx, C0mx, C1mx, alph1mx, alph2mx, omegamx