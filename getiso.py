from copy import deepcopy
from pandas import read_excel
from numpy import (append, concatenate, float_, unique, vstack)
#from numpy import abs as np_abs
#from numpy import max as np_max
from numpy import sum as np_sum
import pytzer as pz
#from scipy.special import factorial
import numpy as np
prmlib = deepcopy(pz.libraries.MarChemSpec)
prmlib.bC['Na-Cl'] = pz.parameters.bC_Na_Cl_A92ii
prmlib.bC['K-Cl'] = pz.parameters.bC_K_Cl_A99
prmlib.lnk['HSO4'] = pz.dissociation.HSO4_CRP94

# Read in isopiestic data
e2i = pz.properties._ele2ions
filename = '../pytzer-support/datasets/isonew.xlsx'
isonew = read_excel(filename, skiprows=2)
isobase = isonew[['src', 'tempK', 'pres']]
isodict = {}
for r in range(len(isonew.index)):
    irow = isonew.iloc[r, 7:]
    isodict[r] = {}
    for i, icell in enumerate(irow):
        if type(icell) is str:
            irc = isodict[r][icell] = {}
            celes = icell.split('-')
            irc['tots'] = irow[i-len(celes):i].values.astype('float64')
            irc_ions = concatenate([e2i[cele][0] for cele in celes])
            irc_mols = concatenate([np.array(e2i[cele][1])*irc['tots'][c]
                for c, cele in enumerate(celes)])
            # Get unique ions without disturbing order
            seen = set()
            seen_add = seen.add
            irc['ions'] = [ion for ion in irc_ions
                if not (ion in seen or seen_add(ion))]
            irc['mols'] = vstack([np_sum(irc_mols[irc_ions == ion])
                for ion in irc['ions']])
            irc['tempK'] = float_([isonew.tempK[r]])
            irc['pres'] = float_([isonew.pres[r]])
            irc['src'] = isonew.src[r]

#%% Get all data for a particular electrolyte
testele = 'KCl'
testdict = {irow: isodict[irow] for irow in isodict.keys()
    if testele in isodict[irow].keys()}
testions = unique(concatenate([testdict[irow][icell]['ions']
    for irow  in testdict.keys() for icell in testdict[irow].keys()]))
prmlib.ions = unique(append(prmlib.ions, testions))
prmlib.add_zeros(prmlib.ions)

# Calculate activities and difference from test ele
#from time import time
#go = time()
for i, irow in enumerate(testdict.keys()):
    if (i+1)%20 == 0:
        print('Solving {} of {}...'.format(i+1, len(testdict.keys())))
    for icell in testdict[irow].keys():
        trc = testdict[irow][icell]
        # Calculate ionic strengths
        trc['zs'] = pz.properties.charges(trc['ions'])[0]
        trc['Istr'] = pz.model.Istr(trc['mols'], trc['zs'])
        # Solve equilibria
        if icell == 'H2SO4':
            allions = np.array(trc['ions'])
            allmxs = pz.matrix.assemble(allions, trc['tempK'], trc['pres'],
                prmlib=prmlib)
            lnks = [prmlib.lnk['HSO4'](trc['tempK'])]
            eqstate_guess = [0.0]
            tots1 = trc['tots']
            fixmols1 = np.array([])
            eles = np.array(['t_HSO4'])
            fixions = np.array([])
            eqstate = pz.equilibrate.solve(eqstate_guess, tots1, fixmols1,
                eles, allions, fixions, allmxs, lnks, ideal=False)['x']
            trc['mols'] = np.vstack(pz.equilibrate.eqstate2mols(eqstate, tots1,
                fixmols1, eles, fixions)[0])
        trc['aw'] = pz.model.aw(trc['mols'], trc['ions'], trc['tempK'],
            trc['pres'], prmlib=prmlib, Izero=trc['Istr'] == 0)
        trc['osm'] = pz.model.osm(trc['mols'], trc['ions'], trc['tempK'],
            trc['pres'], prmlib=prmlib, Izero=trc['Istr'] == 0)
for irow in testdict.keys():
    for icell in testdict[irow].keys():
        if icell != testele:
            trc = testdict[irow][icell]
            trc['del_aw'] = trc['aw'] - testdict[irow][testele]['aw']
            trc['del_osm'] = trc['osm'] - testdict[irow][testele]['osm']
#print(time() - go)

#%% Get arrays for plotting
t_dictrow = vstack([irow for irow  in testdict.keys()
    for icell in testdict[irow].keys() if icell != testele])
t_elemix = vstack([icell for irow  in testdict.keys()
    for icell in testdict[irow].keys() if icell != testele])
t_testtot = vstack([testdict[irow][testele]['tots']
    for irow in testdict.keys() for icell in testdict[irow].keys()
    if icell != testele])
t_testIstr = vstack([testdict[irow][testele]['Istr']
    for irow in testdict.keys() for icell in testdict[irow].keys()
    if icell != testele])
t_tempK = vstack([testdict[irow][testele]['tempK'] for irow in testdict.keys()
    for icell in testdict[irow].keys() if icell != testele])
t_pres = vstack([testdict[irow][testele]['pres'] for irow in testdict.keys()
    for icell in testdict[irow].keys() if icell != testele])
t_delaw = vstack([testdict[irow][icell]['del_aw'] for irow in testdict.keys()
    for icell in testdict[irow].keys() if icell != testele])
t_testaw = vstack([testdict[irow][testele]['aw']
    for irow in testdict.keys() for icell in testdict[irow].keys()
    if icell != testele])
t_delosm = vstack([testdict[irow][icell]['del_osm'] for irow in testdict.keys()
    for icell in testdict[irow].keys() if icell != testele])
t_testosm = vstack([testdict[irow][testele]['osm']
    for irow in testdict.keys() for icell in testdict[irow].keys()
    if icell != testele])
t_eleaw = vstack([testdict[irow][icell]['aw']
    for irow in testdict.keys() for icell in testdict[irow].keys()
    if icell != testele])
t_eleosm = vstack([testdict[irow][icell]['osm']
    for irow in testdict.keys() for icell in testdict[irow].keys()
    if icell != testele])
t_src = vstack([testdict[irow][testele]['src']
    for irow in testdict.keys() for icell in testdict[irow].keys()
    if icell != testele])

#%% Plot results
from matplotlib import pyplot as plt

fig, ax = plt.subplots(2, 1)
L = np.logical_and.reduce((
    np.logical_or.reduce((
#        t_elemix == 'MgCl2',
#        t_elemix == 'KCl',
#        t_elemix == 'H2SO4',
#        t_elemix == 'CaCl2',
        t_elemix == 'NaCl',
    )),
#    t_tempK >= 273.15,
#    t_tempK <= 323.15,
    t_tempK == 298.15,
))
H = np.logical_and(L,
    t_elemix == 'NaCl',
)
sim_mNaCl = np.arange(0.01, 2.5, 0.01)**2
sim_mols = np.array([sim_mNaCl, sim_mNaCl])
sim_ions = np.array(['K', 'Cl'])
sim_aw = pz.model.aw(sim_mols, sim_ions, np.full_like(sim_mNaCl, 298.15),
    np.full_like(sim_mNaCl, 10.1325), prmlib=prmlib, Izero=False)
ax[0].plot(sim_mNaCl, sim_aw, c='k')
ax[0].scatter(t_testtot[L], t_eleaw[L])
ax[0].scatter(t_testtot[H], t_eleaw[H], c='r')
ax[1].plot(sim_mNaCl, 0*sim_mNaCl, c='k')
ax[1].scatter(t_testtot[L], t_delaw[L])
ax[1].scatter(t_testtot[H], t_delaw[H], c='r')

#%% neat NaCl
fig, ax = plt.subplots(2, 1)
L = np.logical_and.reduce((
    np.logical_or.reduce((
#        t_elemix == 'MgCl2',
        t_elemix == 'KCl',
#        t_elemix == 'H2SO4',
#        t_elemix == 'CaCl2',
#        t_elemix == 'NaCl',
#        t_elemix == 'K2SO4',
#        t_elemix == 'Na2SO4',
#        t_elemix == 'MgSO4',
#        t_elemix == 'KCl-CaCl2',
    )),
#    t_tempK >= 273.15,
#    t_tempK <= 323.15,
    t_tempK == 298.15,
))
figsrcs = np.unique(t_src[L])
sim_mNaCl = np.arange(0.01, 2.25, 0.01)**2
sim_mols = np.array([sim_mNaCl, sim_mNaCl])
sim_ions = np.array(['K', 'Cl'])
sim_tempK = np.full_like(sim_mNaCl, 298.15)
sim_pres = np.full_like(sim_mNaCl, 10.1325)
sim_aw = pz.model.aw(sim_mols, sim_ions, sim_tempK, sim_pres, prmlib=prmlib,
    Izero=False)
ax[0].plot(np.sqrt(sim_mNaCl), sim_aw, c='k')
ax[1].plot(np.sqrt(sim_mNaCl), 0*sim_mNaCl, c='k')
for src in figsrcs:
    S = np.logical_and(L, t_src==src)
    ax[0].scatter(np.sqrt(t_testtot[S]), t_eleaw[S], label=src)
    ax[1].scatter(np.sqrt(t_testtot[S]), t_delaw[S]*1e4, label=src)
ax[1].set_ylim(np.max(np.abs(t_delaw[L]))*np.array([-1, 1])*1.1e4)
ax[0].legend()
for x in [0, 1]:
    ax[x].set_xlim((0, np.max(np.sqrt(sim_mNaCl))))
    ax[x].set_xlabel('sqrt[$m$(KCl) / mol$\cdot$kg$^{-1}$]')
ax[0].set_ylabel('Water activity ($a_w$)')
ax[1].set_ylabel('$\Delta a_w \cdot 10^4$')