from copy import deepcopy
from scipy.optimize import minimize
from autograd import numpy as np
from autograd import elementwise_grad as egrad
from autograd.numpy import array, exp, log, sqrt
from autograd.numpy import sum as np_sum
import pytzer as pz

tots, fixmols, eles, fixions, tempK, pres = pz.io.gettots(
    'testfiles/seawaterMZF93.csv')

def sig01(x):
    """Numerically stable logistic sigmoid function."""
    if x < 0:
        return exp(x)/(1 + exp(x))
    else:
        return 1/(1 + exp(-x))

def varmols(pmXe, tots, fixmols, eles, fixions, fixcharges):
    """Calculate variable molalities from solver targets."""
    if 't_HSO4' in eles:
        tHSO4 = tots[eles == 't_HSO4'][0]
        if tHSO4 > 0:
            aHSO4 = sig01(pmXe[1])
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
            aMgOH = sig01(pmXe[2])
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
            atrisH = sig01(pmXe[3])
            mtrisH = ttrisH * atrisH
            mtris = ttrisH - mtrisH
            zbtrisH = mtrisH
        else: # if ttrisH <= 0
            mtrisH = 0
            mtris = 0
            zbtrisH = 0
    zbalance = np_sum(fixmols*fixcharges) + zbHSO4 + zbMg + zbtrisH
    dissociatedH2O = exp(-pmXe[0])
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

def GibbsComponents(pmXe, tots, fixmols, eles, fixions, fixcharges,
        allions, allmxs, lnkHSO4, lnkH2O, lnkMgOH, lnktrisH):
    """Evaluate the Gibbs energy for each component equilibrium."""
    mH, mOH, mHSO4, mSO4, mMg, mMgOH, mtris, mtrisH = varmols(
        pmXe, tots, fixmols, eles, fixions, fixcharges)
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
    lnaw = pz.matrix.lnaw(allmols, allmxs)
    lnacfs = pz.matrix.ln_acfs(allmols, allmxs)
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
        gHSO4 = pmXe[1]
    if tots[eles == 't_Mg'] > 0:
        gMg = GibbsMg(mMg, lnacfMg, mMgOH, lnacfMgOH, mOH, lnacfOH, lnkMgOH)
    else:
        gMg = pmXe[2]
    if tots[eles == 't_trisH'] > 0:
        gtrisH = GibbsTrisH(mH, lnacfH, mtris, lnacfTris, mtrisH, lnacfTrisH,
            lnktrisH)
    else:
        gtrisH = pmXe[3]
    return gH2O, gHSO4, gMg, gtrisH

def Gibbs(pmXe, tots, fixmols, eles, fixions, fixcharges, allions, allmxs,
        lnkHSO4, lnkH2O, lnkMgOH, lnktrisH):
    """Evaluate the total Gibbs energy to be minimised for all equilibria."""
    gH2O, gHSO4, gMg, gtrisH = GibbsComponents(pmXe, tots, fixmols, eles,
        fixions, fixcharges, allions, allmxs, lnkHSO4, lnkH2O, lnkMgOH,
        lnktrisH)
    return gHSO4**2 + gH2O**2 + gMg**2 + gtrisH**2

def GibbsComponentsIdeal(pmXe, tots, fixmols, eles, fixions, fixcharges,
        allions, allmxs, lnkHSO4, lnkH2O, lnkMgOH, lnktrisH):
    """Evaluate the Gibbs energy for each component equilibrium in the 'ideal'
    case where all activity coefficients are unity.
    """
    mH, mOH, mHSO4, mSO4, mMg, mMgOH, mtris, mtrisH = varmols(
        pmXe, tots, fixmols, eles, fixions, fixcharges)
    allmols = deepcopy(fixmols)
    for ele in eles:
        if ele == 't_Mg':
            allmols = [*allmols, mMg, mMgOH]
        elif ele == 't_HSO4':
            allmols = [*allmols, mHSO4, mSO4]
        elif ele == 't_trisH':
            allmols = [*allmols, mtrisH, mtris]
    allmols = array([[*allmols, mH, mOH]])
    # 'Get' activities:
    lnaw = 0
    lnacfH = 0
    lnacfOH = 0
    lnacfHSO4 = 0
    lnacfSO4 = 0
    lnacfMg = 0
    lnacfMgOH = 0
    lnacfTris = 0
    lnacfTrisH = 0
    # Evaluate equilibrium states:
    gH2O = GibbsH2O(lnaw, mH, lnacfH, mOH, lnacfOH, lnkH2O)
    if tots[eles == 't_HSO4'] > 0:
        gHSO4 = GibbsHSO4(mH, lnacfH, mSO4, lnacfSO4, mHSO4, lnacfHSO4,
            lnkHSO4)
    else:
        gHSO4 = pmXe[1]
    if tots[eles == 't_Mg'] > 0:
        gMg = GibbsMg(mMg, lnacfMg, mMgOH, lnacfMgOH, mOH, lnacfOH, lnkMgOH)
    else:
        gMg = pmXe[2]
    if tots[eles == 't_trisH'] > 0:
        gtrisH = GibbsTrisH(mH, lnacfH, mtris, lnacfTris, mtrisH, lnacfTrisH,
            lnktrisH)
    else:
        gtrisH = pmXe[3]
    return gH2O, gHSO4, gMg, gtrisH

def GibbsIdeal(pmXe, tots, fixmols, eles, fixions, fixcharges, allions, allmxs,
        lnkHSO4, lnkH2O, lnkMgOH, lnktrisH):
    """Evaluate the total Gibbs energy to be minimised for all equilibria
    in the 'ideal' case where all activity coefficients are unity.
    """
    gH2O, gHSO4, gMg, gtrisH = GibbsComponentsIdeal(pmXe, tots, fixmols, eles,
        fixions, fixcharges, allions, allmxs, lnkHSO4, lnkH2O, lnkMgOH,
        lnktrisH)
    return gHSO4**2 + gH2O**2 + gMg**2 + gtrisH**2

GibbsGrad = egrad(Gibbs)
GibbsIdealGrad = egrad(GibbsIdeal)

def solveIdeal(pmXe_guess, tots, fixmols, eles, fixions, fixcharges, allions,
        allmxs, lnkHSO4, lnkH2O, lnkMgOH, lnktrisH):
    pmXe = minimize(
        lambda pmXe: GibbsIdeal(pmXe, tots, fixmols, eles, fixions,
            fixcharges, allions, allmxs, lnkHSO4, lnkH2O, lnkMgOH, lnktrisH),
        pmXe_guess,
        method='BFGS',
        jac=lambda pmXe: GibbsIdealGrad(pmXe, tots, fixmols, eles, fixions,
            fixcharges, allions, allmxs, lnkHSO4, lnkH2O, lnkMgOH, lnktrisH),
    )
    return pmXe

def solve(pmXe_guess, tots, fixmols, eles, fixions, fixcharges, allions,
        allmxs, lnkHSO4, lnkH2O, lnkMgOH, lnktrisH):
    pmXe = minimize(
        lambda pmXe: Gibbs(pmXe, tots, fixmols, eles, fixions,
            fixcharges, allions, allmxs, lnkHSO4, lnkH2O, lnkMgOH, lnktrisH),
        pmXe_guess,
        method='BFGS',
        jac=lambda pmXe: GibbsGrad(pmXe, tots, fixmols, eles, fixions,
            fixcharges, allions, allmxs, lnkHSO4, lnkH2O, lnkMgOH, lnktrisH),
    )
    return pmXe

#from autograd import hessian, jacobian
#GibbsJac = jacobian(Gibbs)
#GibbsHess = hessian(Gibbs)

L = 5
Ltots = tots[:, L]
Lfixmols = fixmols[:, L]
LtempK = array([tempK[L]])
Lpres = array([pres[L]])
pmXe = [
    29.978530891580323,
    -16.505227391910964,
    9.55561822596657,
    0.0003713411227473853,
]
fixcharges = np.transpose(pz.properties.charges(fixions)[0])
varmolin = varmols(pmXe, Ltots, Lfixmols, eles, fixions, fixcharges)

allions = pz.properties.getallions(eles, fixions)
allmxs = pz.matrix.assemble(allions, LtempK, Lpres,
    cflib=pz.cflibs.MarChemSpec)
lnkHSO4 = pz.dissociation.HSO4_CRP94(LtempK, Lpres)
lnkH2O = pz.dissociation.H2O_MF(LtempK, Lpres)
lnkMgOH = pz.dissociation.Mg_CW91(LtempK, Lpres)
lnktrisH = pz.dissociation.trisH_BH64(LtempK, Lpres)

Gcomp = GibbsComponents(pmXe, Ltots, Lfixmols, eles, fixions, fixcharges,
    allions, allmxs, lnkHSO4, lnkH2O, lnkMgOH, lnktrisH)
Gtot = Gibbs(pmXe, Ltots, Lfixmols, eles, fixions, fixcharges, allions, allmxs,
    lnkHSO4, lnkH2O, lnkMgOH, lnktrisH)

varmolout = varmols(pmXe, Ltots, Lfixmols, eles, fixions, fixcharges)
pHF = -np.log10(varmolout[0])

def solveboost(pmXe_guess, Ltots, Lfixmols, eles, fixions, fixcharges, allions,
        allmxs, lnkHSO4, lnkH2O, lnkMgOH, lnktrisH):
    pmXe_ideal = solveIdeal(pmXe_guess, Ltots, Lfixmols, eles, fixions,
        fixcharges, allions, allmxs, lnkHSO4, lnkH2O, lnkMgOH, lnktrisH)['x']
    pmXe = solve(pmXe_ideal, Ltots, Lfixmols, eles, fixions, fixcharges,
          allions, allmxs, lnkHSO4, lnkH2O, lnkMgOH, lnktrisH)
    return pmXe

pmXe_guess = [30, 0, 0, 0]
fullsolve = solve(pmXe_guess, Ltots, Lfixmols, eles, fixions, fixcharges,
    allions, allmxs, lnkHSO4, lnkH2O, lnkMgOH, lnktrisH)
boostsolve = solveboost(pmXe_guess, Ltots, Lfixmols, eles, fixions, fixcharges,
    allions, allmxs, lnkHSO4, lnkH2O, lnkMgOH, lnktrisH)
