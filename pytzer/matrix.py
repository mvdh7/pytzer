# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
"""Pitzer model implemented using matrix notation."""
from scipy.special import comb
from autograd import elementwise_grad as egrad
from autograd.numpy import (array, exp, log, ones, size, sqrt, transpose,
    triu_indices, zeros)
from autograd.numpy import abs as np_abs
from autograd.numpy import sum as np_sum
from . import properties
from .libraries import Seawater
from .constants import b, Mw
from .model import g, h
from .unsymmetrical import P75_eq47 as jfunc

def Istr(mols, zs):
    """Ionic strength."""
    return mols @ transpose(zs**2) / 2

def Zstr(mols, zs):
    """Z function."""
    return mols @ transpose(np_abs(zs))

def fG(Aosm, I):
    """Debye-Hueckel component of the excess Gibbs energy."""
    return -4*I*Aosm * log(1 + b*sqrt(I)) / b

def BCT(I, Z, b0, b1, b2, alph1, alph2, C0, C1, omega):
    """Combined B and CT terms."""
    return (b0 + b1*g(alph1*sqrt(I)) + b2*g(alph2*sqrt(I)) +
        (C0 + 4*C1*h(omega*sqrt(I)))*Z/2)

def xij(Aosm, I, zs):
    """xij function for unsymmetrical mixing."""
    return (transpose(zs) @ zs) * 6*Aosm*sqrt(I)

def xi(Aosm, I, zs):
    """xi function for unsymmetrical mixing."""
    return zs**2 * 6*Aosm*sqrt(I)

def xj(Aosm, I, zs):
    """xj function for unsymmetrical mixing."""
    return transpose(zs**2) * 6*Aosm*sqrt(I)

def etheta(Aosm, I, zs):
    """E-theta function for unsymmetrical mixing."""
    x01 = xij(Aosm, I, zs)
    x00 = xi(Aosm, I, zs)
    x11 = xj(Aosm, I, zs)
    return (transpose(zs) @ zs) * (jfunc(x01)
        - (jfunc(x00) + jfunc(x11))/2) / (4*I)

def Gex_nRT(mols, allmxs):
    """Excess Gibbs energy of a solution."""
    (zs, Aosm, b0mx, b1mx, b2mx, C0mx, C1mx, alph1mx, alph2mx, omegamx,
        thetamx, lambdamx, psimxcca, psimxcaa, zetamx, mumx) = allmxs
    I = Istr(mols, zs)
    cats = array([mols[zs > 0]])
    anis = array([mols[zs < 0]])
    neus = array([mols[zs == 0]])
    catsanis = array([(transpose(cats) @ anis).ravel()])
    if I == 0:
        Gex_nRT = (mols @ lambdamx @ transpose(mols) +
            neus @ zetamx @ transpose(catsanis) + neus**3 @ mumx)[0]
    else:
        Z = Zstr(mols, zs)
        zcats = array([zs[zs > 0]])
        zanis = array([zs[zs < 0]])
        catscats = array([(transpose(cats) @ cats)
            [triu_indices(len(cats[0]), k=1)]])
        anisanis = array([(transpose(anis) @ anis)
            [triu_indices(len(anis[0]), k=1)]])
        Gex_nRT = (fG(Aosm, I) + mols @ (
            BCT(I, Z, b0mx, b1mx, b2mx, alph1mx, alph2mx, C0mx, C1mx, omegamx)
                + thetamx + lambdamx) @ transpose(mols) +
            cats @ etheta(Aosm, I, zcats) @ transpose(cats) +
            anis @ etheta(Aosm, I, zanis) @ transpose(anis) +
            catscats @ psimxcca @ transpose(anis) +
            anisanis @ psimxcaa @ transpose(cats) +
            neus @ zetamx @ transpose(catsanis) + neus**3 @ mumx)
    return Gex_nRT[0]

def ln_acfs(mols, allmxs):
    """Natural logarithms of the activity coefficients of all solutes."""
    return egrad(Gex_nRT)(mols, allmxs)[0]

def acfs(mols, allmxs):
    """Activity coefficients of all solutes."""
    return exp(ln_acfs(mols, allmxs))

def lnaw(mols, allmxs):
    """Natural log of the water activity."""
    ww = 1.0
    return (egrad(lambda ww:
        ww*Gex_nRT(mols/ww, allmxs))(ww) - np_sum(mols, axis=1))*Mw

def aw(mols, allmxs):
    """Water activity."""
    return exp(lnaw(mols, allmxs))

def assemble(ions, tempK, pres, prmlib=Seawater):
    """Assemble parameter matrices."""
    zs, cations, anions, neutrals = properties.charges(ions)
    zs = transpose(zs)
    Aosm = prmlib.dh['Aosm'](tempK, pres)[0][0]
    iisize = (size(ions), size(ions))
    b0mx = zeros(iisize)
    b1mx = zeros(iisize)
    b2mx = zeros(iisize)
    C0mx = zeros(iisize)
    C1mx = zeros(iisize)
    alph1mx = -9*ones(iisize)
    alph2mx = -9*ones(iisize)
    omegamx = -9*ones(iisize)
    thetamx = zeros(iisize)
    lambdamx = zeros(iisize)
    psimxcca = zeros((int(comb(size(cations), 2)), size(anions)))
    psimxcaa = zeros((int(comb(size(anions), 2)), size(cations)))
    zetamx = zeros((size(neutrals), size(cations)*size(anions)))
    mumx = zeros((size(neutrals), 1))
    # Pairwise matrices:
    for IX, ionx in enumerate(ions):
        for IY, iony in enumerate(ions):
            if ionx in cations and iony in anions:
                iset = '-'.join((ionx, iony))
                (b0mx[IX, IY], b1mx[IX, IY], b2mx[IX, IY], C0mx[IX, IY],
                    C1mx[IX, IY], alph1mx[IX, IY], alph2mx[IX, IY],
                    omegamx[IX, IY], _)  = prmlib.bC[iset](tempK, pres)
            elif ionx in anions and iony in cations:
                iset = '-'.join((iony, ionx))
                (b0mx[IX, IY], b1mx[IX, IY], b2mx[IX, IY], C0mx[IX, IY],
                    C1mx[IX, IY], alph1mx[IX, IY], alph2mx[IX, IY],
                    omegamx[IX, IY], _)  = prmlib.bC[iset](tempK, pres)
            elif (((ionx in cations and iony in cations) or
                    (ionx in anions and iony in anions)) and
                    ionx != iony):
                iset = [ionx, iony]
                iset.sort()
                iset = '-'.join(iset)
                thetamx[IX, IY] = prmlib.theta[iset](tempK, pres)[0]
            elif ionx in neutrals:
                iset = [ionx, iony]
                if iony in neutrals:
                    iset.sort()
                iset = '-'.join(iset)
                lambdamx[IX, IY] = prmlib.lambd[iset](tempK, pres)[0]
            elif iony in neutrals:
                iset = [iony, ionx]
                if ionx in neutrals:
                    iset.sort()
                iset = '-'.join(iset)
                lambdamx[IX, IY] = prmlib.lambd[iset](tempK, pres)[0]
    # Ionic triplet interactions:
    CC = 0
    for CX, cationx in enumerate(cations):
        for xCY, cationy in enumerate(cations[CX+1:]):
            iset = [cationx, cationy]
            iset.sort()
            iset= '-'.join(iset)
            for A, anion in enumerate(anions):
                iset3 = '-'.join((iset, anion))
                psimxcca[CC, A] = prmlib.psi[iset3](tempK, pres)[0]
            CC = CC + 1
    AA = 0
    for AX, anionx in enumerate(anions):
        for xAY, aniony in enumerate(anions[AX+1:]):
            iset = [anionx, aniony]
            iset.sort()
            iset = '-'.join(iset)
            for C, cation in enumerate(cations):
                iset3 = '-'.join((cation, iset))
                psimxcaa[AA, C] = prmlib.psi[iset3](tempK, pres)[0]
            AA = AA + 1
    # Neutral triplets:
    for N, neutral in enumerate(neutrals):
        iset3 = '-'.join((neutral, neutral, neutral))
        mumx[N, 0] = prmlib.mu[iset3](tempK, pres)[0]
        for C, cation in enumerate(cations):
            for A, anion in enumerate(anions):
                iset3 = '-'.join((neutral, cation, anion))
                zetamx[N, C*size(anions)+A] = (
                    prmlib.zeta[iset3](tempK, pres)[0])
    return (zs, Aosm, b0mx, b1mx, b2mx, C0mx, C1mx, alph1mx, alph2mx, omegamx,
        thetamx, lambdamx, psimxcca, psimxcaa, zetamx, mumx)
