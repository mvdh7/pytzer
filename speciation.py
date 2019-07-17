from copy import deepcopy
from scipy.optimize import minimize
from autograd import numpy as np
from autograd import elementwise_grad as egrad
from autograd.numpy import array, exp, log, sqrt
from autograd.numpy import sum as np_sum
import pytzer as pz

tots, fixmols, eles, fixions, tempK, pres = pz.io.gettots(
    'testfiles/seawaterMZF93.csv')

L = 5
Ltots = tots[:, L]
Lfixmols = fixmols[:, L]
LtempK = array([tempK[L]])
Lpres = array([pres[L]])
eqstate_Julia = [
    29.978530891580323,
    -16.505227391910964,
    9.55561822596657,
    0.0003713411227473853,
]
fixcharges = np.transpose(pz.properties.charges(fixions)[0])
varmolin = pz.equilibrate.varmols(eqstate_Julia, Ltots, Lfixmols, eles,
    fixions, fixcharges)

allions = pz.properties.getallions(eles, fixions)
allmxs = pz.matrix.assemble(allions, LtempK, Lpres,
    cflib=pz.cflibs.MarChemSpec)
lnkHSO4 = pz.dissociation.HSO4_CRP94(LtempK, Lpres)
lnkH2O = pz.dissociation.H2O_MF(LtempK, Lpres)
lnkMgOH = pz.dissociation.Mg_CW91(LtempK, Lpres)
lnktrisH = pz.dissociation.trisH_BH64(LtempK, Lpres)

Gcomp = pz.equilibrate.GibbsComponents(eqstate_Julia, Ltots, Lfixmols, eles,
    fixions, fixcharges, allions, allmxs, lnkHSO4, lnkH2O, lnkMgOH, lnktrisH)
Gtot = pz.equilibrate.Gibbs(eqstate_Julia, Ltots, Lfixmols, eles, fixions,
    fixcharges, allions,allmxs, lnkHSO4, lnkH2O, lnkMgOH, lnktrisH)

varmolout = pz.equilibrate.varmols(eqstate_Julia, Ltots, Lfixmols, eles,
    fixions, fixcharges)
pHF = -np.log10(varmolout[0])

eqstate_guess = [30, -16, 10, 0]
fullsolve = pz.equilibrate.solve(eqstate_guess, Ltots, Lfixmols, eles, fixions,
    fixcharges, allions, allmxs, lnkHSO4, lnkH2O, lnkMgOH, lnktrisH)
quicksolve = pz.equilibrate.solvequick(eqstate_guess, Ltots, Lfixmols, eles,
    fixions, fixcharges, allions, allmxs, lnkHSO4, lnkH2O, lnkMgOH, lnktrisH)
