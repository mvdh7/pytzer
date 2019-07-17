# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
"""Solve for the equilibrium state."""
from copy import deepcopy
from scipy.optimize import minimize
from autograd import elementwise_grad as egrad
from autograd.numpy import array, exp, log, sqrt
from autograd.numpy import sum as np_sum
from . import matrix

def sig01(x):
    """Numerically stable logistic sigmoid function."""
    if x < 0:
        return exp(x)/(1 + exp(x))
    else:
        return 1/(1 + exp(-x))

def varmols(eqstate, tots, fixmols, eles, fixions, fixcharges):
    """Calculate variable molalities from solver targets."""
    if 't_HSO4' in eles:
        tHSO4 = tots[eles == 't_HSO4'][0]
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
        tMg = tots[eles == 't_Mg'][0]
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
        ttrisH = tots[eles == 't_trisH'][0]
        if ttrisH > 0:
            atrisH = sig01(eqstate[3])
            mtrisH = ttrisH * atrisH
            mtris = ttrisH - mtrisH
            zbtrisH = mtrisH
        else: # if ttrisH <= 0
            mtrisH = 0
            mtris = 0
            zbtrisH = 0
    zbalance = np_sum(fixmols*fixcharges) + zbHSO4 + zbMg + zbtrisH
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

def GibbsMg(mMg, lnacfMg, mMgOH, lnacfMgOH, mOH, lnacfOH, lnkMgOH):
    """Evaluate the Gibbs energy for the magnesium-MgOH+ equilibrium."""
    return (lnacfMg + log(mMg) + lnacfOH + log(mOH) - lnacfMgOH - log(mMgOH)
        + lnkMgOH)

def GibbsTrisH(mH, lnacfH, mtris, lnacftris, mtrisH, lnacftrisH, lnktrisH):
    """Evaluate the Gibbs energy for the tris-trisH+ equilibrium."""
    return (lnacftris + log(mtris) - lnacftrisH - log(mtrisH) + lnacfH
        + log(mH) - lnktrisH)

def GibbsComponents(eqstate, tots, fixmols, eles, fixions, fixcharges,
        allions, allmxs, lnkHSO4, lnkH2O, lnkMgOH, lnktrisH, ideal=False):
    """Evaluate the Gibbs energy for each component equilibrium."""
    mH, mOH, mHSO4, mSO4, mMg, mMgOH, mtris, mtrisH = varmols(
        eqstate, tots, fixmols, eles, fixions, fixcharges)
    allmols = deepcopy(fixmols)
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
    if tots[eles == 't_HSO4'] > 0:
        gHSO4 = GibbsHSO4(mH, lnacfH, mSO4, lnacfSO4, mHSO4, lnacfHSO4,
            lnkHSO4)
    else:
        gHSO4 = eqstate[1]
    if tots[eles == 't_Mg'] > 0:
        gMg = GibbsMg(mMg, lnacfMg, mMgOH, lnacfMgOH, mOH, lnacfOH, lnkMgOH)
    else:
        gMg = eqstate[2]
    if tots[eles == 't_trisH'] > 0:
        gtrisH = GibbsTrisH(mH, lnacfH, mtris, lnacfTris, mtrisH, lnacfTrisH,
            lnktrisH)
    else:
        gtrisH = eqstate[3]
    return gH2O, gHSO4, gMg, gtrisH

def Gibbs(eqstate, tots, fixmols, eles, fixions, fixcharges, allions, allmxs,
        lnkHSO4, lnkH2O, lnkMgOH, lnktrisH, ideal=False):
    """Evaluate the total Gibbs energy to be minimised for all equilibria."""
    gH2O, gHSO4, gMg, gtrisH = GibbsComponents(eqstate, tots, fixmols, eles,
        fixions, fixcharges, allions, allmxs, lnkHSO4, lnkH2O, lnkMgOH,
        lnktrisH, ideal)
    return gHSO4**2 + gH2O**2 + gMg**2 + gtrisH**2

_GibbsGrad = egrad(Gibbs)

def solve(eqstate_guess, tots, fixmols, eles, fixions, fixcharges, allions,
        allmxs, lnkHSO4, lnkH2O, lnkMgOH, lnktrisH, ideal=False):
    """Solve for the solution's equilibrium state."""
    Gargs = (tots, fixmols, eles, fixions, fixcharges, allions,
        allmxs, lnkHSO4, lnkH2O, lnkMgOH, lnktrisH, ideal)
    eqstate = minimize(
        lambda eqstate: Gibbs(eqstate, *Gargs),
        eqstate_guess,
        method='BFGS',
        jac=lambda eqstate: _GibbsGrad(eqstate, *Gargs),
    )
    return eqstate

def solvequick(eqstate_guess, tots, fixmols, eles, fixions, fixcharges,
        allions, allmxs, lnkHSO4, lnkH2O, lnkMgOH, lnktrisH):
    """Solve ideal case first to speed up computation."""
    Gargs = (tots, fixmols, eles, fixions, fixcharges, allions, allmxs,
        lnkHSO4, lnkH2O, lnkMgOH, lnktrisH)
    eqstate_ideal = solve(eqstate_guess, *Gargs, ideal=True)['x']
    eqstate = solve(eqstate_ideal, *Gargs, ideal=False)
    return eqstate
