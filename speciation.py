#from copy import deepcopy
from sys import path
if '../calkulate' not in path:
    path.append('../calkulate')
import calkulate as calk
import pytzer as pz
import numpy as np
#from scipy.optimize import minimize

#mols, ions, tempK, pres, prmlib, Gex_nRT, osm, aw, acfs = \
#    pz.blackbox('testfiles/MilleroStandardEquilibrated.csv')

filename = 'testfiles/MilleroStandard.csv'
#filename = 'testfiles/trisASWequilibrium.csv'
allmols, allions, tempK, pres, prmlib, Gex_nRT, osm, aw, acfs, eqstates \
    = pz.blackbox_equilibrate(filename, prmlib=pz.libraries.Seawater)
pH = -np.log10(allmols[allions == 'H']).ravel()

#%% Calculate stoichiometric K*s & compare
mH = allmols[allions == 'H'][0][0]
mHSO4 = allmols[allions == 'HSO4'][0][0]
mSO4 = allmols[allions == 'SO4'][0][0]
mHT = mH + mHSO4
mOH = allmols[allions == 'OH'][0][0]
mHCO3 = allmols[allions == 'HCO3'][0][0]
mCO3 = allmols[allions == 'CO3'][0][0]
mCO2 = allmols[allions == 'CO2'][0][0]
mBOH4 = allmols[allions == 'BOH4'][0][0]
mBOH3 = allmols[allions == 'BOH3'][0][0]
pKstarH2O_pz = -np.log10(mHT*mOH)
pKstarH2O = -np.log10(calk.dissociation.kH2O_T_DSC07(298.15, 35))
pKstarH2CO3_pz = -np.log10(mHCO3*mHT/mCO2)
pKstarH2CO3 = -np.log10(calk.dissociation.ksH2CO3_T_LDK00(298.15, 35)[0])
pKstarHCO3_pz = -np.log10(mCO3*mHT/mHCO3)
pKstarHCO3 = -np.log10(calk.dissociation.ksH2CO3_T_LDK00(298.15, 35)[1])
pKstarBOH3_pz = -np.log10(mBOH4*mHT/mBOH3)
pKstarBOH3 = -np.log10(calk.dissociation.kBOH3_T_D90a(298.15, 35))
pKstarHSO4_pz = -np.log10(mSO4*mH/mHSO4)
pKstarHSO4 = -np.log10(calk.dissociation.kHSO4_F_D90b(298.15, 35))

#%%
#tots, fixmols, eles, fixions, tempK, pres = pz.io.gettots(filename)
#allions = pz.properties.getallions(eles, fixions)
#prmlib = deepcopy(pz.libraries.MIAMI)
#prmlib.add_zeros(allions) # just in case
## Solve for equilibria
#q = 0
#for ele in eles:
#    q += len(pz.properties._eq2ions[ele]) - 1
#eqstate_guess = [0.0 for _ in range(q)]
#if q == 0:
#    eqstate_guess = [30.0]
#else:
#    eqstate_guess.append(30.0)
#eqstates, allmols, allions, lnks = pz.equilibrate._oosetup(eqstate_guess, tots,
#    eles, fixions, tempK, pres, prmlib)
#for L in range(len(tempK)):
#    print('Solving {} of {}...'.format(L+1, len(tempK)))
#    allmxs = pz.matrix.assemble(allions, np.array([tempK[L]]),
#        np.array([pres[L]]), prmlib)
#    if len(eles) > 0:
#        tots1 = tots[:, L]
#    else:
#        tots1 = tots
#    if len(fixions) > 0:
#        fixmols1 = fixmols[:, L]
#    else:
#        fixmols1 = fixmols
#    lnks1 = lnks[:, L]
#    Largs = (eqstate_guess, tots1, fixmols1, eles, allions, fixions,
#        allmxs, lnks1)
#    fixcharges = np.transpose(pz.properties.charges(fixions)[0])
#    Gargs = (tots1, fixmols1, eles, allions, fixions, fixcharges, allmxs, lnks1,
#        False)
##    
#    # from _GibbsComponents
#    eqstate = eqstate_guess
#    mH, mOH, mHSO4, mSO4, mMg, mMgOH, mtris, mtrisH, mCO2, mHCO3, mCO3, mBOH3, mBOH4 = \
#        pz.equilibrate._varmols(eqstate, tots1, fixmols1, eles, fixions, fixcharges)
#    allmols = deepcopy(fixmols1)
#    q = 0
#    for ele in eles:
#        if ele == 't_HSO4':
#            allmols = [*allmols, mHSO4, mSO4]
#            lnkHSO4 = lnks[q]
#            q += 1
#        elif ele == 't_Mg':
#            allmols = [*allmols, mMg, mMgOH]
#            lnkMg = lnks[q]
#            q += 1
#        elif ele == 't_trisH':
#            allmols = [*allmols, mtrisH, mtris]
#            lnktrisH = lnks[q]
#            q += 1
#        elif ele == 't_H2CO3':
#            allmols = [*allmols, mCO2, mHCO3, mCO3]
#            lnkH2CO3 = lnks[q]
#            lnkHCO3 = lnks[q+1]
#            q += 2
#        elif ele == 't_BOH3':
#            allmols = [*allmols, mBOH3, mBOH4]
#            lnkBOH3 = lnks[q]
#            q += 1
#    allmols = np.array([[*allmols, mH, mOH]])
#    solveH2O = len(eqstate) == q+1
#    ideal = False
#    if ideal:
#        lnaw = 0.0
#        lnacfH = 0.0
#        lnacfOH = 0.0
#        lnacfHSO4 = 0.0
#        lnacfSO4 = 0.0
#        lnacfMg = 0.0
#        lnacfMgOH = 0.0
#        lnacftris = 0.0
#        lnacftrisH = 0.0
#        lnacfCO2 = 0.0
#        lnacfHCO3 = 0.0
#        lnacfCO3 = 0.0
#        lnacfBOH3 = 0.0
#        lnacfBOH4 = 0.0
#    else:
#        lnaw = pz.matrix.lnaw(allmols, allmxs)
#        lnacfs = pz.matrix.ln_acfs(allmols, allmxs)
#        lnacfH = lnacfs[allions == 'H']
#        lnacfOH = lnacfs[allions == 'OH']
#        lnacfHSO4 = lnacfs[allions == 'HSO4']
#        lnacfSO4 = lnacfs[allions == 'SO4']
#        lnacfMg = lnacfs[allions == 'Mg']
#        lnacfMgOH = lnacfs[allions == 'MgOH']
#        lnacftris = lnacfs[allions == 'tris']
#        lnacftrisH = lnacfs[allions == 'trisH']
#        lnacfCO2 = lnacfs[allions == 'CO2']
#        lnacfHCO3 = lnacfs[allions == 'HCO3']
#        lnacfCO3 = lnacfs[allions == 'CO3']
#        lnacfBOH3 = lnacfs[allions == 'BOH3']
#        lnacfBOH4 = lnacfs[allions == 'BOH4']
    
#    gH2O, gHSO4, gMg, gtrisH, gH2CO3, gHCO3 = pz.equilibrate._GibbsComponents(
#        eqstate_guess, *Gargs)
##    test = pz.equilibrate._Gibbs(eqstate_guess, *Gargs)
##    eqstate = minimize(
##        lambda eqstate: pz.equilibrate._Gibbs(eqstate, *Gargs),
##        eqstate_guess,
##        method='BFGS',
##        jac=lambda eqstate: pz.equilibrate._GibbsGrad(eqstate, *Gargs),
##    )
#    
##    if L == 0:
##        eqstates[L] = pz.equilibrate.solvequick(*Largs)['x']
##    else:
##        eqstates[L] = pz.equilibrate.solve(*Largs)['x']
##    eqstate_guess = eqstates[L]
##    allmols[L] = pz.equilibrate.eqstate2mols(
##        eqstates[L], tots1, fixmols1, eles, fixions)[0]
#    
#    
##allmols, allions, eqstates = pz.equilibrate.solveloop(eqstate_guess, tots,
##    fixmols, eles, fixions, tempK, pres, prmlib=prmlib)
#
#
##tots, fixmols, eles, fixions, tempK, pres = pz.io.gettots(eqfile)
#
##filename = 'testfiles/CRP94 Table 8.csv'
### Import test dataset
##tots, fixmols, eles, fixions, tempK, pres = pz.io.gettots(filename)
##allions = pz.properties.getallions(eles, fixions)
##prmlib = deepcopy(pz.libraries.MIAMI)
##prmlib.add_zeros(allions) # just in case
##prmlib.lnk['HSO4'] = pz.dissociation.HSO4_CRP94
### Solve for equilibria
##eqstate_guess = [0.0]
##allmols, allions, eqstates = pz.equilibrate.solveloop(eqstate_guess, tots,
##    fixmols, eles, fixions, tempK, pres, prmlib=prmlib)
##
###%%
##alpha = np.round(allmols[1]/tots[0], decimals=5)
