from copy import deepcopy
import pytzer as pz
import numpy as np

#allmols, allions, tempK, pres, prmlib, Gex_nRT, osm, aw, acfs, eqstates \
#    = pz.blackbox_equilibrate('testfiles/CRP94 Table 8.csv')

filename = 'testfiles/CRP94 Table 8.csv'
# Import test dataset
tots, fixmols, eles, fixions, tempK, pres = pz.io.gettots(filename)
allions = pz.properties.getallions(eles, fixions)
prmlib = deepcopy(pz.libraries.MIAMI)
prmlib.add_zeros(allions) # just in case
prmlib.lnk['HSO4'] = pz.dissociation.HSO4_CRP94
# Solve for equilibria
eqstate_guess = [0.0]
allmols, allions, eqstates = pz.equilibrate.solveloop(eqstate_guess, tots,
    fixmols, eles, fixions, tempK, pres, prmlib=prmlib)

#%%
alpha = np.round(allmols[1]/tots[0], decimals=5)
