# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
"""Define module metadata."""
from . import model, parameters
from autograd import numpy as np
from matplotlib import pyplot as plt
version = '0.4.1'

def getprmfuncs():
    """Generate dict containing all interaction parameter functions."""
    functypes = ('bC', 'theta', 'psi', 'lambd', 'zeta', 'mu')
    functypes_ = tuple(['{}_'.format(functype) for functype in functypes])
    allnames = [name for name in dir(parameters)
        if name.startswith(functypes_) and not name.endswith('_none')]
    prmfuncs = {functype: {name: getattr(parameters, name)
            for name in allnames if name.startswith('{}_'.format(functype))}
        for functype in functypes}
    return prmfuncs

def getifuncs(prmfuncs, itype, ions):
    """Extract all interaction functions for a particular interaction."""
    ifuncs = {name: prmfuncs[itype][name] for name in prmfuncs[itype].keys()
        if name.startswith(((1+len(ions))*'{}_').format(itype, *ions))}
    return ifuncs

def evalifuncs(ifuncs, tempK, pres):
    """Evaluate interaction parameters under given conditions."""
    ivals = {name: ifuncs[name](tempK, pres) for name in ifuncs.keys()}
    return ivals

def plotifuncs(prmfuncs, itype, ions, tempK, pres, prmlibBase):
    if itype == 'bC':
        ifuncs = getifuncs(prmfuncs, itype, ions)
        # option 1
#        mols = np.linspace(0.01, 2.5, 100)**2
#        mols = np.array([mols, mols])
#        tempK = np.full_like(mols[0], tempK)
        # option 2
        tempK = np.linspace(273.15, 323.15, 100)
        mols = np.full_like(tempK, 1.0)
        mols = np.array([mols, mols])
        pres = np.full_like(mols[0], pres)
        # option 3
#        pres = np.linspace(0.0, 5000.0, 100)
#        mols = np.full_like(pres, 1.0)
#        mols = np.array([mols, mols])
#        tempK = np.full_like(mols[0], tempK)
        osms = np.full((len(ifuncs), len(mols[0])), np.nan)
        for i, ifunc in enumerate(ifuncs):
            prmlibBase.bC['-'.join(ions)] = ifuncs[ifunc]
            osms[i] = model.osm(mols, ions, tempK, pres, prmlib=prmlibBase)
        fig, ax = plt.subplots()
        for i, ifunc in enumerate(ifuncs):
#            ax.plot(np.sqrt(mols[0]), osms[i], label=ifunc.split('_')[-1])
            ax.plot(tempK, osms[i], label=ifunc.split('_')[-1])
            ax.legend()
    return osms
    
def getirefs(ifuncs):
    """Generate a list of literature sources for the interaction functions."""
    return [name.split('_')[-1] for name in ifuncs.keys()]
