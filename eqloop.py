import pytzer as pz
from pytzer.equilibrate import _sig01
from autograd import numpy as np
from autograd.numpy import exp, sqrt
from autograd.numpy import sum as np_sum

# User inputs eles and lnks
eles = [
    't_BOH3',
    't_H2CO3',
    't_Mg',
]
tots1 = np.array([1.0, 1.0, 1.0])
fixions = ['Na', 'Cl']
fixmols1 = np.array([[1.0, 1.0]])
fixcharges = np.array([[+1.0, -1.0]])
prmlib = pz.libraries.Seawater

eqions = pz.equilibrate.get_eqions(eles, prmlib)
eqstate_guess, lnks, equilibria, parasites, serieses = \
    pz.equilibrate.get_equilibria(eles, 298.15, 10, prmlib)

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

(mH, mOH, mHSO4, mSO4, mMg, mMgOH, mtris, mtrisH, mCO2, mHCO3, mCO3,
    mMgCO3, mBOH3, mBOH4) = _varmols(eqstate_guess, tots1, fixmols1, eles,
    equilibria, fixions, fixcharges)