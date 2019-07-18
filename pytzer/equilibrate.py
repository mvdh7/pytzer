# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
"""Solve for the equilibrium state."""
from copy import deepcopy
from scipy.optimize import minimize
from autograd import elementwise_grad as egrad
from autograd.numpy import array, exp, full, full_like, log, sqrt, transpose
from autograd.numpy import sum as np_sum
from . import dissociation, libraries, matrix, properties

def sig01(x):
    """Numerically stable logistic sigmoid function."""
    if x < 0:
        return exp(x)/(1 + exp(x))
    else:
        return 1/(1 + exp(-x))

def varmols(eqstate, tots1, fixmols1, eles, fixions, fixcharges):
    """Calculate variable molalities from solver targets."""
    if 't_HSO4' in eles:
        tHSO4 = tots1[eles == 't_HSO4'][0]
        if tHSO4 > 0:
            aHSO4 = sig01(eqstate[1])
            mHSO4 = tHSO4*aHSO4
            mSO4 = tHSO4 - mHSO4
            zbHSO4 = -mHSO4 - 2*mSO4
        else: # if tHSO4 <= 0
            mHSO4 = 0
            mSO4 = 0
            zbHSO4 = 0
    if 't_Mg' in eles:
        tMg = tots1[eles == 't_Mg'][0]
        if tMg > 0:
            aMgOH = sig01(eqstate[2])
            mMg = tMg*aMgOH
            mMgOH = tMg - mMg
            zbMg = 2*mMg + mMgOH
        else: # if tMg <= 0
            mMg = 0
            mMgOH = 0
            zbMg = 0
    if 't_trisH' in eles:
        ttrisH = tots1[eles == 't_trisH'][0]
        if ttrisH > 0:
            atrisH = sig01(eqstate[3])
            mtrisH = ttrisH * atrisH
            mtris = ttrisH - mtrisH
            zbtrisH = mtrisH
        else: # if ttrisH <= 0
            mtrisH = 0
            mtris = 0
            zbtrisH = 0
    zbalance = np_sum(fixmols1*fixcharges) + zbHSO4 + zbMg + zbtrisH
    dissociatedH2O = exp(-eqstate[0])
    mOH = (zbalance + sqrt(zbalance**2 + dissociatedH2O))/2
    mH = mOH - zbalance
    return mH, mOH, mHSO4, mSO4, mMg, mMgOH, mtris, mtrisH

def GibbsH2O(lnaw, mH, lnacfH, mOH, lnacfOH, lnkH2O):
    """Evaluate the Gibbs energy for water dissocation."""
    return lnacfH + log(mH) + lnacfOH + log(mOH) - lnaw - lnkH2O

def GibbsHSO4(mH, lnacfH, mSO4, lnacfSO4, mHSO4, lnacfHSO4, lnkHSO4):
    """Evaluate the Gibbs energy for the bisulfate-sulfate equilibrium."""
    return (lnacfH + log(mH) + lnacfSO4 + log(mSO4) - lnacfHSO4 - log(mHSO4)
        - lnkHSO4)

def GibbsMg(mMg, lnacfMg, mMgOH, lnacfMgOH, mOH, lnacfOH, lnkMg):
    """Evaluate the Gibbs energy for the magnesium-MgOH+ equilibrium."""
    return (lnacfMg + log(mMg) + lnacfOH + log(mOH) - lnacfMgOH - log(mMgOH)
        + lnkMg)

def GibbsTrisH(mH, lnacfH, mtris, lnacftris, mtrisH, lnacftrisH, lnktrisH):
    """Evaluate the Gibbs energy for the tris-trisH+ equilibrium."""
    return (lnacftris + log(mtris) - lnacftrisH - log(mtrisH) + lnacfH
        + log(mH) - lnktrisH)

def GibbsComponents(eqstate, tots1, fixmols1, eles, allions, fixions,
        fixcharges, allmxs, lnkHSO4, lnkH2O, lnkMg, lnktrisH, ideal=False):
    """Evaluate the Gibbs energy for each component equilibrium."""
    mH, mOH, mHSO4, mSO4, mMg, mMgOH, mtris, mtrisH = varmols(
        eqstate, tots1, fixmols1, eles, fixions, fixcharges)
    allmols = deepcopy(fixmols1)
    for ele in eles:
        if ele == 't_Mg':
            allmols = [*allmols, mMg, mMgOH]
        elif ele == 't_HSO4':
            allmols = [*allmols, mHSO4, mSO4]
        elif ele == 't_trisH':
            allmols = [*allmols, mtrisH, mtris]
    allmols = array([[*allmols, mH, mOH]])
    # Get activities:
    if ideal:
        lnaw = 0
        lnacfH = 0
        lnacfOH = 0
        lnacfHSO4 = 0
        lnacfSO4 = 0
        lnacfMg = 0
        lnacfMgOH = 0
        lnacfTris = 0
        lnacfTrisH = 0
    else:
        lnaw = matrix.lnaw(allmols, allmxs)
        lnacfs = matrix.ln_acfs(allmols, allmxs)
        lnacfH = lnacfs[allions == 'H']
        lnacfOH = lnacfs[allions == 'OH']
        lnacfHSO4 = lnacfs[allions == 'HSO4']
        lnacfSO4 = lnacfs[allions == 'SO4']
        lnacfMg = lnacfs[allions == 'Mg']
        lnacfMgOH = lnacfs[allions == 'MgOH']
        lnacfTris = lnacfs[allions == 'tris']
        lnacfTrisH = lnacfs[allions == 'trisH']
    # Evaluate equilibrium states:
    gH2O = GibbsH2O(lnaw, mH, lnacfH, mOH, lnacfOH, lnkH2O)
    if tots1[eles == 't_HSO4'] > 0:
        gHSO4 = GibbsHSO4(mH, lnacfH, mSO4, lnacfSO4, mHSO4, lnacfHSO4,
            lnkHSO4)
    else:
        gHSO4 = eqstate[1]
    if tots1[eles == 't_Mg'] > 0:
        gMg = GibbsMg(mMg, lnacfMg, mMgOH, lnacfMgOH, mOH, lnacfOH, lnkMg)
    else:
        gMg = eqstate[2]
    if tots1[eles == 't_trisH'] > 0:
        gtrisH = GibbsTrisH(mH, lnacfH, mtris, lnacfTris, mtrisH, lnacfTrisH,
            lnktrisH)
    else:
        gtrisH = eqstate[3]
    return gH2O, gHSO4, gMg, gtrisH

def Gibbs(eqstate, tots1, fixmols1, eles, allions, fixions, fixcharges, allmxs,
        lnkHSO4, lnkH2O, lnkMg, lnktrisH, ideal=False):
    """Evaluate the total Gibbs energy to be minimised for all equilibria."""
    gH2O, gHSO4, gMg, gtrisH = GibbsComponents(eqstate, tots1, fixmols1, eles,
        allions, fixions, fixcharges, allmxs, lnkHSO4, lnkH2O, lnkMg,
        lnktrisH, ideal)
    return gHSO4**2 + gH2O**2 + gMg**2 + gtrisH**2

_GibbsGrad = egrad(Gibbs)

def solve(eqstate_guess, tots1, fixmols1, eles, allions, fixions, allmxs,
        lnkHSO4, lnkH2O, lnkMg, lnktrisH, ideal=False):
    """Solve for the solution's equilibrium state."""
    fixcharges = transpose(properties.charges(fixions)[0])
    Gargs = (tots1, fixmols1, eles, allions, fixions, fixcharges, allmxs,
        lnkHSO4, lnkH2O, lnkMg, lnktrisH, ideal)
    eqstate = minimize(
        lambda eqstate: Gibbs(eqstate, *Gargs),
        eqstate_guess,
        method='BFGS',
        jac=lambda eqstate: _GibbsGrad(eqstate, *Gargs),
    )
    return eqstate

def solvequick(eqstate_guess, tots1, fixmols1, eles, allions, fixions, allmxs,
        lnkHSO4, lnkH2O, lnkMg, lnktrisH):
    """Solve ideal case first to speed up computation."""
    Sargs = (tots1, fixmols1, eles, allions, fixions, allmxs, lnkHSO4, lnkH2O,
        lnkMg, lnktrisH)
    eqstate_ideal = solve(eqstate_guess, *Sargs, ideal=True)['x']
    eqstate = solve(eqstate_ideal, *Sargs, ideal=False)
    return eqstate

def solveloop(eqstate_guess, tots, fixmols, eles, fixions, tempK, pres,
        prmlib=libraries.Seawater):
    """Run solver through a loop of input data."""
    eqstates = full((len(tots[0]), len(eqstate_guess)), 0.0)
    allions = properties.getallions(eles, fixions)
    allmols = full((len(tots[0]), len(allions)), 0.0)
    _lnkHSO4 = dissociation.HSO4_CRP94(tempK, pres)
    _lnkH2O = dissociation.H2O_MF(tempK, pres)
    _lnkMg = dissociation.Mg_CW91(tempK, pres)
    _lnktrisH = dissociation.trisH_BH64(tempK, pres)
    for L in range(len(tots[0])):
        print('Solving {} of {}...'.format(L+1, len(tots[0])))
        allmxs = matrix.assemble(allions, array([tempK[L]]), array([pres[L]]),
            prmlib)
        Largs = (eqstate_guess, tots[:, L], fixmols[:, L], eles,
            allions, fixions, allmxs, _lnkHSO4[L], _lnkH2O[L], _lnkMg[L],
            _lnktrisH[L])
        if L == 0:
            eqstates[L] = solvequick(*Largs)['x']
        else:
            eqstates[L] = solve(*Largs)['x']
        eqstate_guess = eqstates[L]
        allmols[L] = eqstate2mols(
            eqstates[L], tots[:, L], fixmols[:, L], eles, fixions)[0]
    print('Solving complete!')
    return transpose(allmols), allions, eqstates

def eqstate2mols(eqstate, tots1, fixmols1, eles, fixions):
    """Convert eqstate solution to arrays required for Pytzer functions."""
    fixcharges = transpose(properties.charges(fixions)[0])
    mH, mOH, mHSO4, mSO4, mMg, mMgOH, mtris, mtrisH = varmols(
        eqstate, tots1, fixmols1, eles, fixions, fixcharges)
    allions = properties.getallions(eles, fixions)
    allmols = full_like(allions, 0.0, dtype='float64')
    for i, ion in enumerate(allions):
        if ion in fixions:
            allmols[i] = fixmols1[fixions == ion]
        elif ion == 'H':
            allmols[i] = mH
        elif ion == 'OH':
            allmols[i] = mOH
        elif ion == 'HSO4':
            allmols[i] = mHSO4
        elif ion == 'SO4':
            allmols[i] = mSO4
        elif ion == 'Mg':
            allmols[i] = mMg
        elif ion == 'MgOH':
            allmols[i] = mMgOH
        elif ion == 'tris':
            allmols[i] = mtris
        elif ion == 'trisH':
            allmols[i] = mtrisH
    return allmols, allions