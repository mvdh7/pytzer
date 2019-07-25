# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
"""Solve for the equilibrium state."""
from copy import deepcopy
from scipy.optimize import minimize
from autograd import elementwise_grad as egrad
from autograd.numpy import (array, exp, full, full_like, log, nan, sqrt,
    transpose)
from autograd.numpy import sum as np_sum
from . import libraries, matrix, properties

def _sig01(x):
    """Numerically stable logistic sigmoid function."""
    if x < 0:
        return exp(x)/(1 + exp(x))
    else:
        return 1/(1 + exp(-x))

def _varmols(eqstate, tots1, fixmols1, eles, fixions, fixcharges):
    """Calculate variable molalities from solver targets."""
    for e, ele in enumerate(eles):
        if ele == 't_HSO4':
            tHSO4 = tots1[e]
            if tHSO4 > 0:
                aHSO4 = _sig01(eqstate[e])
                mHSO4 = tHSO4*aHSO4
                mSO4 = tHSO4 - mHSO4
                zbHSO4 = -mHSO4 - 2*mSO4
            else:
                mHSO4 = 0.0
                mSO4 = 0.0
                zbHSO4 = 0.0
        elif ele == 't_Mg':
            tMg = tots1[e]
            if tMg > 0:
                aMgOH = _sig01(eqstate[e])
                mMg = tMg*aMgOH
                mMgOH = tMg - mMg
                zbMg = 2*mMg + mMgOH
            else:
                mMg = 0.0
                mMgOH = 0.0
                zbMg = 0.0
        elif ele == 't_trisH':
            ttrisH = tots1[e]
            if ttrisH > 0:
                atrisH = _sig01(eqstate[e])
                mtrisH = ttrisH * atrisH
                mtris = ttrisH - mtrisH
                zbtrisH = mtrisH
            else:
                mtrisH = 0.0
                mtris = 0.0
                zbtrisH = 0.0
    if 't_HSO4' not in eles:
        mHSO4 = 0.0
        mSO4 = 0.0
        zbHSO4 = 0.0
    if 't_Mg' not in eles:
        mMg = 0.0
        mMgOH = 0.0
        zbMg = 0.0
    if 't_trisH' not in eles:
        mtrisH = 0.0
        mtris = 0.0
        zbtrisH = 0.0
    if len(fixcharges) == 0:
        zbfixed = 0.0
    else:
        zbfixed = np_sum(fixmols1*fixcharges)
    zbalance = zbfixed + zbHSO4 + zbMg + zbtrisH
    if len(eqstate) == len(tots1) + 1:
        dissociatedH2O = exp(-eqstate[-1])
    elif len(eqstate) == len(tots1):
        dissociatedH2O = 0.0
    else:
        print('WARNING: eqstate and tots1 dimensions are not compatible!')
    mOH = (zbalance + sqrt(zbalance**2 + dissociatedH2O))/2
    mH = mOH - zbalance
    return mH, mOH, mHSO4, mSO4, mMg, mMgOH, mtris, mtrisH

def _GibbsH2O(lnaw, mH, lnacfH, mOH, lnacfOH, lnkH2O):
    """Evaluate the Gibbs energy for water dissocation."""
    return lnacfH + log(mH) + lnacfOH + log(mOH) - lnaw - lnkH2O

def _GibbsHSO4(mH, lnacfH, mSO4, lnacfSO4, mHSO4, lnacfHSO4, lnkHSO4):
    """Evaluate the Gibbs energy for the bisulfate-sulfate equilibrium."""
    return (lnacfH + log(mH) + lnacfSO4 + log(mSO4) - lnacfHSO4 - log(mHSO4)
        - lnkHSO4)

def _GibbsMg(mMg, lnacfMg, mMgOH, lnacfMgOH, mOH, lnacfOH, lnkMg):
    """Evaluate the Gibbs energy for the magnesium-MgOH+ equilibrium."""
    return (lnacfMg + log(mMg) + lnacfOH + log(mOH) - lnacfMgOH - log(mMgOH)
        + lnkMg)

def _GibbstrisH(mH, lnacfH, mtris, lnacftris, mtrisH, lnacftrisH, lnktrisH):
    """Evaluate the Gibbs energy for the tris-trisH+ equilibrium."""
    return (lnacftris + log(mtris) - lnacftrisH - log(mtrisH) + lnacfH
        + log(mH) - lnktrisH)

def _GibbsComponents(eqstate, tots1, fixmols1, eles, allions, fixions,
        fixcharges, allmxs, lnks, ideal=False):
    """Evaluate the Gibbs energy for each component equilibrium."""
    mH, mOH, mHSO4, mSO4, mMg, mMgOH, mtris, mtrisH = _varmols(
        eqstate, tots1, fixmols1, eles, fixions, fixcharges)
    allmols = deepcopy(fixmols1)
    # The order in which molality values are added to allmols within each
    # equilibrium in the following statements MUST comply with the order in
    # which the corresponding ions were added to allions by the function
    # pytzer.properties.getallions, which is defined by their order in the
    # entries of the pytzer.properties._eq2ions dict.
    for e, ele in enumerate(eles):
        if ele == 't_HSO4':
            allmols = [*allmols, mHSO4, mSO4]
            lnkHSO4 = lnks[e]
        elif ele == 't_Mg':
            allmols = [*allmols, mMg, mMgOH]
            lnkMg = lnks[e]
        elif ele == 't_trisH':
            allmols = [*allmols, mtrisH, mtris]
            lnktrisH = lnks[e]
    allmols = array([[*allmols, mH, mOH]])
    # Get activities:
    if ideal:
        lnaw = 0.0
        lnacfH = 0.0
        lnacfOH = 0.0
        lnacfHSO4 = 0.0
        lnacfSO4 = 0.0
        lnacfMg = 0.0
        lnacfMgOH = 0.0
        lnacftris = 0.0
        lnacftrisH = 0.0
    else:
        lnaw = matrix.lnaw(allmols, allmxs)
        lnacfs = matrix.ln_acfs(allmols, allmxs)
        lnacfH = lnacfs[allions == 'H']
        lnacfOH = lnacfs[allions == 'OH']
        lnacfHSO4 = lnacfs[allions == 'HSO4']
        lnacfSO4 = lnacfs[allions == 'SO4']
        lnacfMg = lnacfs[allions == 'Mg']
        lnacfMgOH = lnacfs[allions == 'MgOH']
        lnacftris = lnacfs[allions == 'tris']
        lnacftrisH = lnacfs[allions == 'trisH']
    # Evaluate equilibrium states:
    if len(eqstate) == len(tots1) + 1:
        lnkH2O = lnks[-1]
        gH2O = _GibbsH2O(lnaw, mH, lnacfH, mOH, lnacfOH, lnkH2O)
    else:
        gH2O = 0.0
    if tots1[eles == 't_HSO4'] > 0:
        gHSO4 = _GibbsHSO4(mH, lnacfH, mSO4, lnacfSO4, mHSO4, lnacfHSO4,
            lnkHSO4)
    else:
        gHSO4 = 0.0
    if tots1[eles == 't_Mg'] > 0:
        gMg = _GibbsMg(mMg, lnacfMg, mMgOH, lnacfMgOH, mOH, lnacfOH, lnkMg)
    else:
        gMg = 0.0
    if tots1[eles == 't_trisH'] > 0:
        gtrisH = _GibbstrisH(mH, lnacfH, mtris, lnacftris, mtrisH, lnacftrisH,
            lnktrisH)
    else:
        gtrisH = 0.0
    return gH2O, gHSO4, gMg, gtrisH

def _Gibbs(eqstate, tots1, fixmols1, eles, allions, fixions, fixcharges,
        allmxs, lnks, ideal=False):
    """Evaluate the total Gibbs energy to be minimised for all equilibria."""
    gH2O, gHSO4, gMg, gtrisH = _GibbsComponents(eqstate, tots1, fixmols1, eles,
        allions, fixions, fixcharges, allmxs, lnks, ideal)
    return gHSO4**2 + gH2O**2 + gMg**2 + gtrisH**2

_GibbsGrad = egrad(_Gibbs)

def solve(eqstate_guess, tots1, fixmols1, eles, allions, fixions, allmxs,
        lnks, ideal=False):
    """Solve for the solution's equilibrium state."""
    fixcharges = transpose(properties.charges(fixions)[0])
    Gargs = (tots1, fixmols1, eles, allions, fixions, fixcharges, allmxs, lnks,
        ideal)
    eqstate = minimize(
        lambda eqstate: _Gibbs(eqstate, *Gargs),
        eqstate_guess,
        method='BFGS',
        jac=lambda eqstate: _GibbsGrad(eqstate, *Gargs),
    )
    return eqstate

def solvequick(eqstate_guess, tots1, fixmols1, eles, allions, fixions, allmxs,
        lnks):
    """Solve ideal case first to speed up computation."""
    Sargs = (tots1, fixmols1, eles, allions, fixions, allmxs, lnks)
    eqstate_ideal = solve(eqstate_guess, *Sargs, ideal=True)['x']
    eqstate = solve(eqstate_ideal, *Sargs, ideal=False)
    return eqstate

def solveloop(eqstate_guess, tots, fixmols, eles, fixions, tempK, pres,
        prmlib=libraries.Seawater):
    """Run solver through a loop of input data."""
    eqstates = full((len(tots[0]), len(eqstate_guess)), 0.0)
    allions = properties.getallions(eles, fixions)
    allmols = full((len(tots[0]), len(allions)), 0.0)
    lnks = full((len(eqstate_guess), len(tempK)), nan)
    for e, ele in enumerate(eles):
        if ele == 't_HSO4':
            lnks[e] = prmlib.lnk['HSO4'](tempK, pres)
        elif ele == 't_Mg':
            lnks[e] = prmlib.lnk['Mg'](tempK, pres)
        elif ele == 't_trisH':
            lnks[e] = prmlib.lnk['trisH'](tempK, pres)
    if len(eqstate_guess) == len(eles) + 1:
        lnks[-1] = prmlib.lnk['H2O'](tempK, pres)
    for L in range(len(tots[0])):
        print('Solving {} of {}...'.format(L+1, len(tots[0])))
        allmxs = matrix.assemble(allions, array([tempK[L]]), array([pres[L]]),
            prmlib)
        tots1 = tots[:, L]
        if len(fixions) > 0:
            fixmols1 = fixmols[:, L]
        else:
            fixmols1 = fixmols
        lnks1 = lnks[:, L]
        Largs = (eqstate_guess, tots1, fixmols1, eles, allions, fixions,
            allmxs, lnks1)
        if L == 0:
            eqstates[L] = solvequick(*Largs)['x']
        else:
            eqstates[L] = solve(*Largs)['x']
        eqstate_guess = eqstates[L]
        allmols[L] = eqstate2mols(
            eqstates[L], tots1, fixmols1, eles, fixions)[0]
    print('Solving complete!')
    return transpose(allmols), allions, eqstates

def eqstate2mols(eqstate, tots1, fixmols1, eles, fixions):
    """Convert eqstate solution to arrays required for Pytzer functions."""
    fixcharges = transpose(properties.charges(fixions)[0])
    mH, mOH, mHSO4, mSO4, mMg, mMgOH, mtris, mtrisH = _varmols(
        eqstate, tots1, fixmols1, eles, fixions, fixcharges)
    allions = properties.getallions(eles, fixions)
    allmols = full_like(allions, 0.0, dtype='float64')
    for i, ion in enumerate(allions):
        if len(fixions) > 0:
            if ion in fixions:
                allmols[i] = fixmols1[fixions == ion]
        if ion not in fixions:
            if ion == 'H':
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
