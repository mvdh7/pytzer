from copy import deepcopy
import numpy as np
import pytzer as pz

itype = 'bC'
ions = np.array(['Sr', 'Cl'])
tempK = 298.15
pres = 10.1325
prmfuncs = pz.meta.getprmfuncs()
ifuncs = pz.meta.getifuncs(prmfuncs, itype, ions)
ivals = pz.meta.evalifuncs(ifuncs, tempK, pres)
mols = np.array([[3.0], [3.0]])
prmlib = deepcopy(pz.libraries.MIAMI)
prmlib.bC['Na-Cl'] = pz.parameters.bC_Na_Cl_JESS
prmlib.bC['K-Cl'] = pz.parameters.bC_K_Cl_JESS
aw = pz.model.aw(mols, ions, tempK, pres, prmlib=prmlib)
osms = pz.meta.plotifuncs(prmfuncs, itype, ions, tempK, pres, prmlib)
