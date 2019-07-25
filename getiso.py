from copy import deepcopy
from pandas import read_excel
from numpy import (append, array, concatenate, float_, logical_not, mean,
    ones_like, pi, size, sqrt, std, unique, vstack)
#from numpy import abs as np_abs
#from numpy import max as np_max
from numpy import sum as np_sum
import pytzer as pz
#from scipy.special import factorial
import numpy as np
prmlib = deepcopy(pz.libraries.Seawater)

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
            seen = set()
            seen_add = seen.add
            irc['ions'] = [ion for ion in irc_ions
                if not (ion in seen or seen_add(ion))]
            irc['mols'] = vstack([np_sum(irc_mols[irc_ions == ion])
                for ion in irc['ions']])
            irc['tempK'] = float_([isonew.tempK[r]])
            irc['pres'] = float_([isonew.pres[r]])

#%% Get all data for a particular electrolyte
testele = 'NaCl'
testdict = {irow: isodict[irow] for irow in isodict.keys()
    if testele in isodict[irow].keys()}
testions = unique(concatenate([testdict[irow][icell]['ions']
    for irow  in testdict.keys() for icell in testdict[irow].keys()]))
prmlib.ions = unique(append(prmlib.ions,testions))
prmlib.add_zeros(prmlib.ions)

# Calculate activities and difference from test ele
from time import time
go = time()
for i, irow in enumerate(testdict.keys()):
    print('Solving {} of {}...'.format(i, len(testdict.keys())))
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
            lnks = [prmlib.lnk['HSO4'](trc['tempK'], trc['pres'])]
            eqstate_guess = [0.0]
            tots1 = trc['tots']
            fixmols1 = np.array([])
            eles = np.array(['t_HSO4'])
            fixions = np.array([])
            eqstate = pz.equilibrate.solve(eqstate_guess, tots1, fixmols1,
                eles, allions, fixions, allmxs, lnks, ideal=False)['x']
            trc['mols'] = np.vstack(pz.equilibrate.eqstate2mols(eqstate, tots1,
                fixmols1, eles, fixions)[0])
            trc['aw'] = pz.matrix.aw(np.transpose(trc['mols']), allmxs)
#            eqstate = pz.equilibrate.solvequick(eqstate_guess, trc['tots'],
#                [], ['t_HSO4'], trc['ions'], [], allmxs, lnks)['x']
#            trc['mols'] = np.transpose(pz.equilibrate.eqstate2mols(eqstate,
#                trc['tots'], [], ['t_HSO4'], []))
        # Calculate other water activities
        trc['aw'] = pz.model.aw(trc['mols'], trc['ions'], trc['tempK'],
            trc['pres'], prmlib=prmlib, Izero=trc['Istr'] == 0)
for irow in testdict.keys():
    for icell in testdict[irow].keys():
        if icell != testele:
            trc = testdict[irow][icell]
            trc['del_aw'] = trc['aw'] - testdict[irow][testele]['aw']
print(time() - go)

#%% Get arrays for plotting
t_dictrow = vstack([irow for irow  in testdict.keys()
    for icell in testdict[irow].keys() if icell != testele])
t_elemix = array([icell for irow  in testdict.keys()
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
t_eleaw = vstack([testdict[irow][icell]['aw']
    for irow in testdict.keys() for icell in testdict[irow].keys()
    if icell != testele])

#%% Plot results
from matplotlib import pyplot as plt

fig, ax = plt.subplots(1, 1)
#L = np.logical_not(np.logical_or.reduce((
#    t_elemix == 'sucrose',
#    t_elemix == 'glycerol',
#    t_elemix == 'urea',
#    t_elemix == '(trisH)2SO4',
#    t_elemix == 'CsCl',
#    t_elemix == 'CuCl2',
#    t_elemix == 'CuSO4',
#    t_elemix == 'CuSO4-CuCl2',
##    t_elemix == 'H2SO4',
##    t_elemix == 'K2CO3',
##    t_elemix == 'LiCl',
##    t_elemix == 'MgCl2-CsCl',
##    t_elemix == 'Na2SO4-CuCl2',
##    t_elemix == 'Na2SO4-CuSO4',
##    t_elemix == 'NaCl-CuCl2',
##    t_elemix == 'NaCl-CuSO4',
##    t_elemix == 'trisHCl',
#    t_elemix == 'CaCl2',
#)))
#L = np.logical_or.reduce((
#    t_elemix == 'MgCl2',
#    t_elemix == 'NaCl-MgCl2',
#))
L = t_elemix == 'H2SO4'
#L = np.logical_and(
#    np.logical_or.reduce((
#        t_elemix == 'KCl',
#        t_elemix == 'CaCl2',
#        t_elemix == 'MgCl2',
#        t_elemix == 'MgCl2-KCl-NaCl',
#        t_elemix == 'NaCl-KCl',
#    )),
#    t_tempK.ravel() == 298.15,
#)
#ax.scatter(t_testIstr, t_delaw)

sim_mNaCl = np.arange(0.01, 2.5, 0.01)**2
sim_mols = np.array([sim_mNaCl, sim_mNaCl])
sim_ions = np.array(['Na', 'Cl'])
sim_aw = pz.model.aw(sim_mols, sim_ions, np.full_like(sim_mNaCl, 323.15),
    np.full_like(sim_mNaCl, 10.1325), prmlib=prmlib, Izero=False)

ax.plot(sim_mNaCl, sim_aw)
ax.scatter(t_testIstr[L], t_eleaw[L])

#from scipy.io import savemat
#savemat('testfiles/isonew_' + testele + '.mat',
#        {'dictrow': t_dictrow,
#         'elemix' : t_elemix ,
#         'testtot': t_testtot,
#         'T'      : t_T      ,
#         'delaw'  : t_delaw  })

#%%
#            irc['osm'] = pz.model.osm(irc['mols'],irc['ions'],irc['T'],prmlib)
#
#            irc['aw'] = pz.model.osm2aw(irc['mols'],irc['osm'])
#
## Get all water activities
#all_aw = [concatenate(
#    [isodict[irow][icell]['aw'] for icell in isodict[irow].keys()]).ravel() \
#     for irow in isodict.keys()]
#
#aw_mean = array([mean(aw) for aw in all_aw])
#aw_sd   = array([std (aw) for aw in all_aw])
#ncups   = array([size(aw) for aw in all_aw])
#
#def std_unbias(stds,nobs):
#
#    def keven(k2):
#
#        k = k2/2
#
#        return sqrt(2 / (pi * (2*k - 1))) * 2**(2*k - 2) \
#            * factorial(k - 1)**2 / factorial(2*k - 2)
#
#    def kodd(k2p1):
#
#        k = (k2p1 - 1)/2
#
#        return sqrt(pi / k) * factorial(2*k - 1) \
#            / (2**(2*k - 1) * factorial(k - 1)**2)
#
#    c4 = ones_like(nobs, dtype='float64')
#
#    Leven = nobs % 2 == 0
#    Lodd  = logical_not(Leven)
#
#    c4[Leven] = keven(nobs[Leven])
#    c4[Lodd ] = kodd (nobs[Lodd ])
#
#    return stds / c4
#
#aw_sdu = std_unbias(aw_sd,ncups)
#
##%%
#from matplotlib import pyplot as plt
#
#plt.scatter(range(len(aw_sdu)),aw_sdu)
##plt.scatter(aw_mean,aw_sdu)
#
#sdu_max = np_abs(np_max(aw_sdu)) * 1.1
#
#plt.ylim([0,sdu_max])
#
