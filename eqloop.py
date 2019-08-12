#from copy import deepcopy
import pytzer as pz
from pytzer import matrix
#from pytzer.equilibrate import (_varmols, _GibbsBOH3, _GibbsH2CO3, _GibbsH2O,
#    _GibbsHCO3, _GibbsHSO4, _GibbsMgOH, _GibbstrisH)
from autograd import numpy as np
from autograd.numpy import array#, exp, sqrt
#from autograd.numpy import sum as np_sum

# User inputs eles and lnks
eles = [
    't_BOH3',
    't_H2CO3',
    't_Mg',
    
]
tots1 = np.array([1.0, 1.0, 1.0])
fixions = ['Na', 'Cl']
fixmols1 = np.array([1.0, 1.0])
fixcharges = np.array([[+1.0, -1.0]])
prmlib = pz.libraries.Seawater

eqions = pz.equilibrate.get_eqions(eles, prmlib)
eqstate_guess, lnks, equilibria = pz.equilibrate.get_equilibria(
    eles, 298.15, 10, prmlib)

(mH, mOH, mHSO4, mSO4, mMg, mMgOH, mtris, mtrisH, mCO2, mHCO3, mCO3,
    mMgCO3, mBOH3, mBOH4) = pz.equilibrate._varmols(eqstate_guess, tots1,
    fixmols1, eles, equilibria, fixions, fixcharges)

allions = array([*fixions, *eqions])
prmlib.add_zeros(allions)
allmxs = matrix.assemble(allions, array([298.15]), array([10]), prmlib=prmlib)
#gH2O, gHSO4, gMg, gtrisH, gH2CO3, gHCO3, gBOH3 = \
#    pz.equilibrate._GibbsComponents(eqstate, tots1, fixmols1, eles,
#    allions, fixions, fixcharges, allmxs, lnks, equilibria, eqions,
#    ideal=False)
test = pz.equilibrate._Gibbs(eqstate_guess, tots1, fixmols1, eles, allions, fixions, fixcharges,
        allmxs, lnks, equilibria, eqions, ideal=False)

#eqstate_guess = pz.equilibrate.solve(eqstate_guess, tots1, fixmols1, eles, allions, fixions, allmxs,
#    lnks, equilibria, eqions, ideal=True)['x']
eqstate_guess = [-12.5, -1.8, -3.9, 1.6, 29.5]
eqstate = pz.equilibrate.solve(eqstate_guess, tots1, fixmols1, eles, allions,
    fixions, allmxs, lnks, equilibria, eqions, ideal=False)['x']
gFINAL = pz.equilibrate._Gibbs(eqstate, tots1, fixmols1, eles, allions,
    fixions, fixcharges, allmxs, lnks, equilibria, eqions, ideal=False)
