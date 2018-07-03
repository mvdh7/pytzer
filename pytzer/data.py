from  autograd import numpy as np
import pandas as pd
from .constants import Mw
from . import tconv

###############################################################################

def dis(datapath):
    
    disbase = pd.read_excel(datapath+'dis.xlsx', sheet_name='DIS data',
                            header=0, skiprows=2, usecols=5)
    
    disbase = prep(disbase)
    
    return disbase

###############################################################################

def fpd(datapath):

    fpdbase = pd.read_excel(datapath+'fpd.xlsx', sheet_name='FPD data',
                            header=0, skiprows=2, usecols=6)

    # Calculate freezing point temperature from FPD
    fpdbase['t'] = 273.15 - fpdbase.fpd
    fpdbase = prep(fpdbase)

    # FPD to osmotic coefficient
    mols = np.array([(fpdbase.nC*fpdbase.m).values,
                     (fpdbase.nA*fpdbase.m).values]).transpose()
    fpdbase['osm'] = tconv.fpd2osm(mols,fpdbase.fpd.values)

    return fpdbase

###############################################################################

def vpl(datapath):

    vplbase = pd.read_excel(datapath+'vpl.xlsx', sheet_name='VPL data',
                            header=0, skiprows=2, usecols=8)
    vplbase = prep(vplbase)

    # Osmotic coefficient from water activity
    vplbase['osm'] = -np.log(vplbase.aw) / (vplbase.nu * vplbase.m * Mw)

    return vplbase

###############################################################################
def prep(xxxbase):

    # Add extra variables
    xxxbase['sqm'] = np.sqrt(xxxbase.m)
    xxxbase['nu'],xxxbase['zC'],xxxbase['zA'],xxxbase['nC'],xxxbase['nA'] \
        = znu(xxxbase.ele)

    # Sort by electrolyte, then molality, then source
    xxxbase = xxxbase.sort_values(['ele', 'm', 'src'])

    return xxxbase

def znu(ele):

    # Define dicts
    zC = {'NaCl':+1, 'KCl':+1, 'Na2SO4':+1, 'CaCl2':+2, 'H2SO4':+1}
    zA = {'NaCl':-1, 'KCl':-1, 'Na2SO4':-2, 'CaCl2':-1, 'H2SO4':-2}
    nC = {'NaCl': 1, 'KCl': 1, 'Na2SO4': 2, 'CaCl2': 1, 'H2SO4': 2}
    nA = {'NaCl': 1, 'KCl': 1, 'Na2SO4': 1, 'CaCl2': 2, 'H2SO4': 1}

    # Return: nu, zC, zA, nuC, nuA
    return ele.map(nC) + ele.map(nA), \
           ele.map(zC), ele.map(zA), ele.map(nC), ele.map(nA)
           