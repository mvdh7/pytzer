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

def _varmols(eqstate, tots1, fixmols1, eles, equilibria, fixions, fixcharges):
    """Calculate variable molalities from solver targets."""
    # Get total concentrations
    for e, ele in enumerate(eles):
        if ele == 't_Mg':
            tMg = tots1[e]
        elif ele == 't_HSO4':
            tHSO4 = tots1[e]
        elif ele == 't_trisH':
            ttrisH = tots1[e]
        elif ele == 't_H2CO3':
            tH2CO3 = tots1[e]
        elif ele == 't_BOH3':
            tBOH3 = tots1[e]
    # Loop manually through equilibria
    eqindex = equilibria.index
    if len(eles) > 0:
        # Magnesium non-parasites
        if 't_Mg' in eles:
            if 'MgOH' in equilibria:
                q = eqindex('MgOH')
                aMgOH = _sig01(eqstate[q])
                mMgOH = tMg*aMgOH
            else:
                mMgOH = 0
            mMg = tMg - mMgOH
        else:
            mMgOH = 0
            mMg = 0
        # Bisulfate series
        if 't_HSO4' in eles:
            q = eqindex('HSO4')
            aHSO4 = _sig01(eqstate[q])
            mHSO4 = tHSO4*aHSO4
            mSO4 = tHSO4 - mHSO4
        else:
            mHSO4 = 0
            mSO4 = 0
        # TrisH+ series
        if 't_trisH' in eles:
            q = eqindex('trisH')
            atrisH = _sig01(eqstate[q])
            mtrisH = ttrisH*atrisH
            mtris = ttrisH - mtrisH
        else:
            mtrisH = 0
            mtris = 0
        # Carbonic acid series
        if 't_H2CO3' in eles:
            q = eqindex('H2CO3')
            aH2CO3 = _sig01(eqstate[q])
            q = eqindex('HCO3')
            aHCO3 = _sig01(eqstate[q])
            mCO2 = tH2CO3*aH2CO3
            mHCO3 = (tH2CO3 - mCO2)*aHCO3
            mCO3 = tH2CO3 - mCO2 - mHCO3
            # Metal-carbonate parasites
            if 't_Mg' in eles:
                if 'MgCO3' in equilibria:
                    q = eqindex('MgCO3')
                    aMgCO3 = _sig01(eqstate[q])
                    if mMg > mHCO3:
                        mMgCO3 = aMgCO3*mHCO3
                    else:
                        mMgCO3 = aMgCO3*mMg
                    mCO3 = mCO3 - mMgCO3
                    mMg = mMg - mMgCO3
                else:
                    mMgCO3 = 0
        else:
            mCO2 = 0
            mHCO3 = 0
            mCO3 = 0
            mMgCO3 = 0
        # Boric acid series
        if 't_BOH3' in eles:
            q = eqindex('BOH3')
            aBOH3 = _sig01(eqstate[q])
            mBOH3 = tBOH3*aBOH3
            mBOH4 = tBOH3 - mBOH3
        else:
            mBOH3 = 0
            mBOH4 = 0
    # Charge balances
    zbHSO4 = -mHSO4 - 2*mSO4
    zbMg = 2*mMg + mMgOH
    zbtrisH = mtrisH
    zbH2CO3 = -(mHCO3 + 2*mCO3)
    zbBOH3 = -mBOH4
    if len(fixcharges) == 0:
        zbfixed = 0
    else:
        zbfixed = np_sum(fixmols1*fixcharges)
    zbalance = zbfixed + zbHSO4 + zbMg + zbtrisH + zbH2CO3 + zbBOH3
    if 'H2O' in equilibria:
        q = eqindex('H2O')
        dissociatedH2O = exp(-eqstate[q])
    else:
        dissociatedH2O = 0
    mOH = (zbalance + sqrt(zbalance**2 + dissociatedH2O))/2
    mH = mOH - zbalance
    return (mH, mOH, mHSO4, mSO4, mMg, mMgOH, mtris, mtrisH, mCO2, mHCO3, mCO3,
        mMgCO3, mBOH3, mBOH4)

#def _varmols_original(eqstate, tots1, fixmols1, eles, fixions, fixcharges):
#    """Calculate variable molalities from solver targets."""
#    # Seems long-winded, but it's necessary for Autograd
#    q = 0
#    if len(eles) > 0:
#        for e, ele in enumerate(eles):
#            if ele == 't_HSO4':
#                tHSO4 = tots1[e]
#                if tHSO4 > 0:
#                    aHSO4 = _sig01(eqstate[q])
#                    mHSO4 = tHSO4*aHSO4
#                    mSO4 = tHSO4 - mHSO4
#                else:
#                    mHSO4 = 0.0
#                    mSO4 = 0.0
#                q += 1
#            elif ele == 't_Mg':
#                tMg = tots1[e]
#                if tMg > 0:
#                    aMgOH = _sig01(eqstate[q])
#                    mMg = tMg*aMgOH
#                    mMgOH = tMg - mMg
#                else:
#                    mMg = 0.0
#                    mMgOH = 0.0
#                q += 1
#            elif ele == 't_trisH':
#                ttrisH = tots1[e]
#                if ttrisH > 0:
#                    atrisH = _sig01(eqstate[q])
#                    mtrisH = ttrisH*atrisH
#                    mtris = ttrisH - mtrisH
#                else:
#                    mtrisH = 0.0
#                    mtris = 0.0
#                q += 1
#            elif ele == 't_H2CO3':
#                tH2CO3 = tots1[e]
#                if tH2CO3 > 0:
#                    aH2CO3 = _sig01(eqstate[q])
#                    aHCO3 = _sig01(eqstate[q+1])
#                    mCO2 = tH2CO3*aH2CO3
#                    mHCO3 = (tH2CO3 - mCO2)*aHCO3
#                    mCO3 = tH2CO3 - mCO2 - mHCO3
#                else:
#                    mCO2 = 0.0
#                    mHCO3 = 0.0
#                    mCO3 = 0.0
#                q += 2
#            elif ele == 't_BOH3':
#                tBOH3 = tots1[e]
#                if tBOH3 > 0:
#                    aBOH3 = _sig01(eqstate[q])
#                    mBOH3 = tBOH3*aBOH3
#                    mBOH4 = tBOH3 - mBOH3
#                else:
#                    mBOH3 = 0.0
#                    mBOH4 = 0.0
#                q += 1
#        if 't_HSO4' not in eles:
#            mHSO4 = 0.0
#            mSO4 = 0.0
#        if 't_Mg' not in eles:
#            mMg = 0.0
#            mMgOH = 0.0
#        if 't_trisH' not in eles:
#            mtrisH = 0.0
#            mtris = 0.0
#        if 't_H2CO3' not in eles:
#            mCO2 = 0.0
#            mHCO3 = 0.0
#            mCO3 = 0.0
#        if 't_BOH3' not in eles:
#            mBOH3 = 0.0
#            mBOH4 = 0.0
#    else:
#        mHSO4 = 0.0
#        mSO4 = 0.0
#        mMg = 0.0
#        mMgOH = 0.0
#        mtrisH = 0.0
#        mtris = 0.0
#        mCO2 = 0.0
#        mHCO3 = 0.0
#        mCO3 = 0.0
#        mBOH3 = 0.0
#        mBOH4 = 0.0
#    zbHSO4 = -mHSO4 - 2*mSO4
#    zbMg = 2*mMg + mMgOH
#    zbtrisH = mtrisH
#    zbH2CO3 = -(mHCO3 + 2*mCO3)
#    zbBOH3 = -mBOH4
#    if len(fixcharges) == 0:
#        zbfixed = 0.0
#    else:
#        zbfixed = np_sum(fixmols1*fixcharges)
#    zbalance = zbfixed + zbHSO4 + zbMg + zbtrisH + zbH2CO3 + zbBOH3
#    if len(eqstate) == q+1:
#        dissociatedH2O = exp(-eqstate[-1])
#    elif len(eqstate) == q:
#        dissociatedH2O = 0.0
#    else:
#        print('WARNING: eqstate and tots1 dimensions are not compatible!')
#    mOH = (zbalance + sqrt(zbalance**2 + dissociatedH2O))/2
#    mH = mOH - zbalance
#    return (mH, mOH, mHSO4, mSO4, mMg, mMgOH, mtris, mtrisH, mCO2, mHCO3, mCO3,
#        mBOH3, mBOH4)

def _GibbsH2O(lnaw, mH, lnacfH, mOH, lnacfOH, lnkH2O):
    """Evaluate the Gibbs energy for water dissocation."""
    return lnacfH + log(mH) + lnacfOH + log(mOH) - lnaw - lnkH2O

def _GibbsHSO4(mH, lnacfH, mSO4, lnacfSO4, mHSO4, lnacfHSO4, lnkHSO4):
    """Evaluate the Gibbs energy for the bisulfate-sulfate equilibrium."""
    return (lnacfH + log(mH) + lnacfSO4 + log(mSO4) - lnacfHSO4 - log(mHSO4)
        - lnkHSO4)

def _GibbsMgOH(mMg, lnacfMg, mMgOH, lnacfMgOH, mOH, lnacfOH, lnkMgOH):
    """Evaluate the Gibbs energy for the magnesium-MgOH+ equilibrium."""
    return (lnacfMg + log(mMg) + lnacfOH + log(mOH) - lnacfMgOH - log(mMgOH)
        + lnkMgOH)

def _GibbstrisH(mH, lnacfH, mtris, lnacftris, mtrisH, lnacftrisH, lnktrisH):
    """Evaluate the Gibbs energy for the tris-trisH+ equilibrium."""
    return (lnacftris + log(mtris) - lnacftrisH - log(mtrisH) + lnacfH
        + log(mH) - lnktrisH)

def _GibbsH2CO3(lnaw, mH, lnacfH, mHCO3, lnacfHCO3, mCO2, lnacfCO2, lnkH2CO3):
    """Evaluate the Gibbs energy for the H2CO3-bicarbonate equilibrium."""
    return (lnacfH + log(mH) + lnacfHCO3 + log(mHCO3) - lnacfCO2 - log(mCO2)
        - lnaw - lnkH2CO3)

def _GibbsHCO3(mH, lnacfH, mHCO3, lnacfHCO3, mCO3, lnacfCO3, lnkHCO3):
    """Evaluate the Gibbs energy for the bicarbonate-carbonate equilibrium."""
    return (lnacfH + log(mH) + lnacfCO3 + log(mCO3) - lnacfHCO3 - log(mHCO3)
        - lnkHCO3)

def _GibbsBOH3(lnaw, lnacfBOH4, mBOH4, lnacfBOH3, mBOH3, lnacfH, mH, lnkBOH3):
    """Evaluate the Gibbs energy for the boric acid equilibrium."""
    return (lnacfBOH4 + log(mBOH4) + lnacfH + log(mH) - lnacfBOH3 -
        log(mBOH3) - lnaw - lnkBOH3)

#def _GibbsMgCO3():

def _GibbsComponents_original(eqstate, tots1, fixmols1, eles, allions, fixions,
        fixcharges, allmxs, lnks, equilibria, ideal=False):
    """Evaluate the Gibbs energy for each component equilibrium."""
    (mH, mOH, mHSO4, mSO4, mMg, mMgOH, mtris, mtrisH, mCO2, mHCO3, mCO3,
        mMgCO3, mBOH3, mBOH4) = _varmols(eqstate, tots1, fixmols1, eles,
        equilibria, fixions, fixcharges)
    allmols = deepcopy(fixmols1)
    # The order in which molality values are added to allmols within each
    # equilibrium in the following statements MUST comply with the order in
    # which the corresponding ions were added to allions by the function
    # pytzer.properties.getallions, which is defined by their order in the
    # entries of the pytzer.properties._eq2ions dict.
    q = 0
    for ele in eles:
        if ele == 't_HSO4':
            allmols = [*allmols, mHSO4, mSO4]
            lnkHSO4 = lnks[q]
            q += 1
        elif ele == 't_Mg':
            allmols = [*allmols, mMg, mMgOH]
            lnkMgOH = lnks[q]
            q += 1
        elif ele == 't_trisH':
            allmols = [*allmols, mtrisH, mtris]
            lnktrisH = lnks[q]
            q += 1
        elif ele == 't_H2CO3':
            allmols = [*allmols, mCO2, mHCO3, mCO3]
            lnkH2CO3 = lnks[q]
            lnkHCO3 = lnks[q+1]
            q += 2
        elif ele == 't_BOH3':
            allmols = [*allmols, mBOH3, mBOH4]
            lnkBOH3 = lnks[q]
            q += 1
    allmols = array([[*allmols, mH, mOH]])
    solveH2O = len(eqstate) == q+1
    # Get activities:
    if ideal:
        lnaw = 0
        lnacfH = 0
        lnacfOH = 0
        lnacfHSO4 = 0
        lnacfSO4 = 0
        lnacfMg = 0
        lnacfMgOH = 0
        lnacftris = 0
        lnacftrisH = 0
        lnacfCO2 = 0
        lnacfHCO3 = 0
        lnacfCO3 = 0
        lnacfBOH3 = 0
        lnacfBOH4 = 0
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
        lnacfCO2 = lnacfs[allions == 'CO2']
        lnacfHCO3 = lnacfs[allions == 'HCO3']
        lnacfCO3 = lnacfs[allions == 'CO3']
        lnacfBOH3 = lnacfs[allions == 'BOH3']
        lnacfBOH4 = lnacfs[allions == 'BOH4']
    # Evaluate equilibrium states:
    if solveH2O:
        lnkH2O = lnks[-1]
        gH2O = _GibbsH2O(lnaw, mH, lnacfH, mOH, lnacfOH, lnkH2O)
    else:
        gH2O = 0.0
    if len(eles) > 0:
        if tots1[eles == 't_HSO4'] > 0:
            gHSO4 = _GibbsHSO4(mH, lnacfH, mSO4, lnacfSO4, mHSO4, lnacfHSO4,
                lnkHSO4)
        else:
            gHSO4 = 0.0
        if tots1[eles == 't_Mg'] > 0:
            gMg = _GibbsMgOH(mMg, lnacfMg, mMgOH, lnacfMgOH, mOH, lnacfOH,
                lnkMgOH)
        else:
            gMg = 0.0
        if tots1[eles == 't_trisH'] > 0:
            gtrisH = _GibbstrisH(mH, lnacfH, mtris, lnacftris, mtrisH,
                lnacftrisH, lnktrisH)
        else:
            gtrisH = 0.0
        if tots1[eles == 't_H2CO3'] > 0:
            gH2CO3 = _GibbsH2CO3(lnaw, mH, lnacfH, mHCO3, lnacfHCO3, mCO2,
                lnacfCO2, lnkH2CO3)
            gHCO3 = _GibbsHCO3(mH, lnacfH, mHCO3, lnacfHCO3, mCO3, lnacfCO3,
                lnkHCO3)
        else:
            gH2CO3 = 0.0
            gHCO3 = 0.0
        if tots1[eles == 't_BOH3'] > 0:
            gBOH3 = _GibbsBOH3(lnaw, lnacfBOH4, mBOH4, lnacfBOH3, mBOH3,
                lnacfH, mH, lnkBOH3)
        else:
            gBOH3 = 0.0
    else:
        gHSO4 = 0.0
        gMg = 0.0
        gtrisH = 0.0
        gH2CO3 = 0.0
        gHCO3 = 0.0
        gBOH3 = 0.0
    return gH2O, gHSO4, gMg, gtrisH, gH2CO3, gHCO3, gBOH3

def _GibbsComponents(eqstate, tots1, fixmols1, eles, allions, fixions,
        fixcharges, allmxs, lnks, equilibria, eqions, ideal=False):
    """Evaluate the Gibbs energy for each component equilibrium."""
    (mH, mOH, mHSO4, mSO4, mMg, mMgOH, mtris, mtrisH, mCO2, mHCO3, mCO3,
        mMgCO3, mBOH3, mBOH4) = _varmols(eqstate, tots1, fixmols1, eles,
        equilibria, fixions, fixcharges)
    allmols = deepcopy(fixmols1)
    for ion in eqions:
        if ion == 'H':
            allmols = [*allmols, mH]
        elif ion == 'OH':
            allmols = [*allmols, mOH]
        elif ion == 'HSO4':
            allmols = [*allmols, mHSO4]
        elif ion == 'SO4':
            allmols = [*allmols, mSO4]
        elif ion == 'Mg':
            allmols = [*allmols, mMg]
        elif ion == 'MgOH':
            allmols = [*allmols, mMgOH]
        elif ion == 'tris':
            allmols = [*allmols, mtris]
        elif ion == 'trisH':
            allmols = [*allmols, mtrisH]
        elif ion == 'CO2':
            allmols = [*allmols, mCO2]
        elif ion == 'HCO3':
            allmols = [*allmols, mHCO3]
        elif ion == 'CO3':
            allmols = [*allmols, mCO3]
        elif ion == 'MgCO3':
            allmols = [*allmols, mMgCO3]
        elif ion == 'BOH3':
            allmols = [*allmols, mBOH3]
        elif ion == 'BOH4':
            allmols = [*allmols, mBOH4]
    allmols = array([allmols])
    # Get activities:
    if ideal:
        lnaw = 0
        lnacfH = 0
        lnacfOH = 0
        lnacfHSO4 = 0
        lnacfSO4 = 0
        lnacfMg = 0
        lnacfMgOH = 0
        lnacftris = 0
        lnacftrisH = 0
        lnacfCO2 = 0
        lnacfHCO3 = 0
        lnacfCO3 = 0
        lnacfBOH3 = 0
        lnacfBOH4 = 0
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
        lnacfCO2 = lnacfs[allions == 'CO2']
        lnacfHCO3 = lnacfs[allions == 'HCO3']
        lnacfCO3 = lnacfs[allions == 'CO3']
        lnacfBOH3 = lnacfs[allions == 'BOH3']
        lnacfBOH4 = lnacfs[allions == 'BOH4']
    # Evaluate equilibrium states:
    eqindex = equilibria.index
    if 'H2O' in equilibria:
        q = eqindex('H2O')
        lnkH2O = lnks[q]        
        gH2O = _GibbsH2O(lnaw, mH, lnacfH, mOH, lnacfOH, lnkH2O)
    else:
        gH2O = 0
    if 'HSO4' in equilibria:
        q = eqindex('HSO4')
        lnkHSO4 = lnks[q]
        gHSO4 = _GibbsHSO4(mH, lnacfH, mSO4, lnacfSO4, mHSO4, lnacfHSO4,
            lnkHSO4)
    else:
        gHSO4 = 0
    if 'MgOH' in equilibria:
        q = eqindex('MgOH')
        lnkMgOH = lnks[q]
        gMg = _GibbsMgOH(mMg, lnacfMg, mMgOH, lnacfMgOH, mOH, lnacfOH, lnkMgOH)
    else:
        gMg = 0
    if 'trisH' in equilibria:
        q = eqindex('trisH')
        lnktrisH = lnks[q]
        gtrisH = _GibbstrisH(mH, lnacfH, mtris, lnacftris, mtrisH, lnacftrisH,
            lnktrisH)
    else:
        gtrisH = 0
    if 'H2CO3' in equilibria:
        q = eqindex('H2CO3')
        lnkH2CO3 = lnks[q]
        gH2CO3 = _GibbsH2CO3(lnaw, mH, lnacfH, mHCO3, lnacfHCO3, mCO2,
            lnacfCO2, lnkH2CO3)
    else:
        gH2CO3 = 0
    if 'HCO3' in equilibria:
        q = eqindex('HCO3')
        lnkHCO3 = lnks[q]
        gHCO3 = _GibbsHCO3(mH, lnacfH, mHCO3, lnacfHCO3, mCO3, lnacfCO3,
            lnkHCO3)
    else:
        gHCO3 = 0
    if 'BOH3' in equilibria:
        q = eqindex('BOH3')
        lnkBOH3 = lnks[q]
        gBOH3 = _GibbsBOH3(lnaw, lnacfBOH4, mBOH4, lnacfBOH3, mBOH3,
            lnacfH, mH, lnkBOH3)
    else:
        gBOH3 = 0
    return gH2O, gHSO4, gMg, gtrisH, gH2CO3, gHCO3, gBOH3

def _Gibbs(eqstate, tots1, fixmols1, eles, allions, fixions, fixcharges,
        allmxs, lnks1, equilibria, eqions, ideal=False):
    """Evaluate the total Gibbs energy to be minimised for all equilibria."""
    gH2O, gHSO4, gMg, gtrisH, gH2CO3, gHCO3, gBOH3 = _GibbsComponents(eqstate,
        tots1, fixmols1, eles, allions, fixions, fixcharges, allmxs, lnks1,
        equilibria, eqions, ideal)
    return (gH2O**2 + gHSO4**2 + gMg**2 + gtrisH**2 + gH2CO3**2 + gHCO3**2 +
        gBOH3**2)

_GibbsGrad = egrad(_Gibbs)

def solve(eqstate_guess, tots1, fixmols1, eles, allions, fixions, allmxs,
        lnks1, equilibria, eqions, ideal=False):
    """Solve for the solution's equilibrium state."""
    fixcharges = transpose(properties.charges(fixions)[0])
    Gargs = (tots1, fixmols1, eles, allions, fixions, fixcharges, allmxs,
        lnks1, equilibria, eqions, ideal)
    eqstate = minimize(
        lambda eqstate: _Gibbs(eqstate, *Gargs),
        eqstate_guess,
        method='BFGS',
        jac=lambda eqstate: _GibbsGrad(eqstate, *Gargs),
    )
    return eqstate

def solvequick(eqstate_guess, tots1, fixmols1, eles, allions, fixions, allmxs,
        lnks1, equilibria, eqions):
    """Solve ideal case first to speed up computation."""
    Sargs = (tots1, fixmols1, eles, allions, fixions, allmxs, lnks1,
        equilibria, eqions)
    eqstate_ideal = solve(eqstate_guess, *Sargs, ideal=True)['x']
    eqstate = solve(eqstate_ideal, *Sargs, ideal=False)
    return eqstate

# Within _parasites, dict LHS is the lnks function, RHS is the ions to be added
_parasites = {
    't_Mg': {
        'always': ('Mg',),
        'MgOH': ('MgOH',),
        'MgF': ('MgF',),
        'MgCO3': ('MgCO3',),
    },
    't_Ca': {
        'always': ('Ca',),
        'CaCO3': ('CaCO3',),
    },
}
# Within _serieses, all the ions are always added
_serieses = {
    't_H2CO3': {
        'ions': ('CO2', 'HCO3', 'CO3',),
        'lnks': ('H2CO3', 'HCO3',),
    },
    't_BOH3': {
        'ions': ('BOH3', 'BOH4',),
        'lnks': ('BOH3',),
    },
    't_HSO4': {
        'ions': ('HSO4', 'SO4',),
        'lnks': ('HSO4',),
    },
    't_trisH': {
        'ions': ('trisH', 'tris',),
        'lnks': ('trisH',),
    },
}

def get_eqions(eles, prmlib):
    """Get list of equilibrating ions."""
    eqions = []
    for ele in eles:
        if ele in _parasites:
            eqions.extend(_parasites[ele]['always'])
            for parasite in _parasites[ele]:
                if parasite in prmlib.lnk:
                    eqions.extend(_parasites[ele][parasite])
        elif ele in _serieses:
            eqions.extend(_serieses[ele]['ions'])
    eqions.extend(['H', 'OH'])
    return eqions

def get_equilibria(eles, tempK1, pres1, prmlib):
    """Generate eqstate_guess and lists of ln(K) functions and equilibria."""
    q = 0
    equilibria = []
    lnkfuncs = []
    for ele in eles:
        if ele in _parasites:
            for parasite in _parasites[ele]:
                if parasite in prmlib.lnk:
                    q += 1
                    equilibria.append(parasite)
                    lnkfuncs.append(prmlib.lnk[parasite])
    for ele in eles:
        if ele in _serieses:
            q += len(_serieses[ele]['lnks'])
            for series in _serieses[ele]['lnks']:
                equilibria.append(series)
                lnkfuncs.append(prmlib.lnk[series])
    eqstate_guess = [0.0 for _ in range(q)]
    if 'H2O' in prmlib.lnk:
        equilibria.append('H2O')
        lnkfuncs.append(prmlib.lnk['H2O'])
        eqstate_guess.append(30.0)
    return eqstate_guess, lnkfuncs, equilibria

def get_lnks(eles, tempK1, pres1, prmlib):
    """Get ln(K) values for one set of conditions."""
    lnks = []
    for ele in eles:
        if ele in _parasites:
            for parasite in _parasites[ele]:
                if parasite in prmlib.lnk:
                    lnks.append(prmlib.lnk[parasite](tempK1, pres1))
    for ele in eles:
        if ele in _serieses:
            for series in _serieses[ele]['lnks']:
                lnks.append(prmlib.lnk[series](tempK1, pres1))
    if 'H2O' in prmlib.lnk:
        lnks.append(prmlib.lnk['H2O'](tempK1, pres1))
    return lnks

def countlnks(eles, prmlib):
    """Count the number of lnk values required and prepare eqstate_guess."""
    q = 0
    for ele in eles:
        q += len(properties._eq2ions[ele]) - 1
    eqstate_guess = [0.0 for _ in range(q)]
    if 'H2O' in prmlib.lnk:
        if q == 0:
            eqstate_guess = [30.0]
        else:
            eqstate_guess.append(30.0)
    return eqstate_guess

def getlnks(eqstate_guess, eles, tempK, pres, prmlib):
    """Get arrays of lnk values for solver functions."""
    lnks = full((len(eqstate_guess), len(tempK)), nan)
    q = 0
    for ele in eles:
        if ele == 't_HSO4':
            lnks[q] = prmlib.lnk['HSO4'](tempK, pres)
            q += 1
        elif ele == 't_Mg':
            lnks[q] = prmlib.lnk['MgOH'](tempK, pres)
            q += 1
        elif ele == 't_trisH':
            lnks[q] = prmlib.lnk['trisH'](tempK, pres)
            q += 1
        elif ele == 't_H2CO3':
            lnks[q] = prmlib.lnk['H2CO3'](tempK, pres)
            lnks[q+1] = prmlib.lnk['HCO3'](tempK, pres)
            q += 2
        elif ele == 't_BOH3':
            lnks[q] = prmlib.lnk['BOH3'](tempK, pres)
            q += 1
    if len(eqstate_guess) == q+1:
        lnks[-1] = prmlib.lnk['H2O'](tempK, pres)
    return lnks

def _oosetup(tots, eles, fixions, tempK, pres, prmlib):
    """Prepare for solveloop and solvepool."""
    eqstate_guess, eqions, equilibria = get_equilibria(eles, tempK[0], pres[0],
        prmlib)
    eqstates = full((len(tempK), len(eqstate_guess)), 0.0)
    allions = properties.getallions(eles, fixions)
    allmols = full((len(tempK), len(allions)), 0.0)
    lnks = getlnks(eqstate_guess, eles, tempK, pres, prmlib)
    return eqstates, allmols, allions, lnks, equilibria, eqions

def solveloop(eqstate_guess, tots, fixmols, eles, fixions, tempK, pres,
        prmlib=libraries.Seawater):
    """Run solver through a loop of input data."""
    eqstates, allmols, allions, lnks, equilibria, eqions = _oosetup(
        tots, eles, fixions, tempK, pres, prmlib)
    for L in range(len(tempK)):
        print('Solving {} of {}...'.format(L+1, len(tempK)))
        allmxs = matrix.assemble(allions, array([tempK[L]]), array([pres[L]]),
            prmlib)
        if len(eles) > 0:
            tots1 = tots[:, L]
        else:
            tots1 = tots
        if len(fixions) > 0:
            fixmols1 = fixmols[:, L]
        else:
            fixmols1 = fixmols
        lnks1 = lnks[:, L]
        Largs = (eqstate_guess, tots1, fixmols1, eles, allions, fixions,
            allmxs, lnks1, equilibria, eqions)
        if L == 0:
            eqstates[L] = solvequick(*Largs)['x']
        else:
            eqstates[L] = solve(*Largs)['x']
        eqstate_guess = eqstates[L]
        allmols[L] = eqstate2mols(
            eqstates[L], tots1, fixmols1, eles, fixions, equilibria)[0]
    print('Solving complete!')
    return transpose(allmols), allions, eqstates

def eqstate2mols(eqstate, tots1, fixmols1, eles, fixions, equilibria):
    """Convert eqstate solution to arrays required for Pytzer functions."""
    fixcharges = transpose(properties.charges(fixions)[0])
    (mH, mOH, mHSO4, mSO4, mMg, mMgOH, mtris, mtrisH, mCO2, mHCO3, mCO3,
        mMgCO3, mBOH3, mBOH4) = _varmols(eqstate, tots1, fixmols1, eles,
        equilibria, fixions, fixcharges)
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
            elif ion == 'CO2':
                allmols[i] = mCO2
            elif ion == 'HCO3':
                allmols[i] = mHCO3
            elif ion == 'CO3':
                allmols[i] = mCO3
            elif ion == 'MgCO3':
                allmols[i] = mMgCO3
            elif ion == 'BOH3':
                allmols[i] = mBOH3
            elif ion == 'BOH4':
                allmols[i] = mBOH4
    return allmols, allions
