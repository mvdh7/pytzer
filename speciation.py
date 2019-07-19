from autograd import numpy as np
import pytzer as pz

tots, fixmols, eles, fixions, tempK, pres = pz.io.gettots(
    'testfiles/trisASWequilibrium.csv')
fixcharges = np.transpose(pz.properties.charges(fixions)[0])

L = 5
Ltots = tots[:, L]
Lfixmols = fixmols[:, L]
LtempK = np.array([tempK[L]])
Lpres = np.array([pres[L]])
eqstate_Julia = [
    29.978530891580323,
    -16.505227391910964,
    9.55561822596657,
    0.0003713411227473853,
]
varmolin = pz.equilibrate._varmols(eqstate_Julia, Ltots, Lfixmols, eles,
    fixions, fixcharges)

allions = pz.properties.getallions(eles, fixions)
allmxs = pz.matrix.assemble(allions, LtempK, Lpres,
    prmlib=pz.libraries.MarChemSpec)
lnkHSO4 = pz.dissociation.HSO4_CRP94(LtempK, Lpres)
lnkH2O = pz.dissociation.H2O_MF(LtempK, Lpres)
lnkMg = pz.dissociation.Mg_CW91(LtempK, Lpres)
lnktrisH = pz.dissociation.trisH_BH64(LtempK, Lpres)

Gcomp = pz.equilibrate._GibbsComponents(eqstate_Julia, Ltots, Lfixmols, eles,
    allions, fixions, fixcharges, allmxs, lnkHSO4, lnkH2O, lnkMg, lnktrisH)
Gtot = pz.equilibrate._Gibbs(eqstate_Julia, Ltots, Lfixmols, eles, allions,
    fixions, fixcharges, allmxs, lnkHSO4, lnkH2O, lnkMg, lnktrisH)

varmolout = pz.equilibrate._varmols(eqstate_Julia, Ltots, Lfixmols, eles,
    fixions, fixcharges)
pHF = -np.log10(varmolout[0])

eqstate_guess = [30, 0, 0, 0]
#fullsolve = pz.equilibrate.solve(eqstate_guess, Ltots, Lfixmols, eles, fixions,
#    allions, allmxs, lnkHSO4, lnkH2O, lnkMg, lnktrisH)
#quicksolve = pz.equilibrate.solvequick(eqstate_guess, Ltots, Lfixmols, eles,
#    fixions, allions, allmxs, lnkHSO4, lnkH2O, lnkMg, lnktrisH)
#allmols, allions, eqstates = pz.equilibrate.solveloop(eqstate_guess, tots,
#    fixmols, eles, fixions, tempK, pres,)

allmols, allions, tempK, pres, prmlib, Gex_nRT, osm, aw, acfs, eqstates \
    = pz.blackbox_equilibrate('testfiles/trisASWequilibrium.csv')
