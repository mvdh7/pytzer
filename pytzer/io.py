# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
"""Import solution composition data, and export the results."""
from autograd.numpy import (array, concatenate, genfromtxt, logical_and,
    nan_to_num, savetxt, shape, transpose, vstack)
from autograd.numpy import sum as np_sum
from .properties import _ele2ionmass, _ion2mass

def getmols(filename, delimiter=',', skip_top=0):
    """Import molality, temperature and pressure data from a CSV file, where
    all ionic concentrations are defined (i.e. no equilibration)."""
    data = genfromtxt(filename, delimiter=delimiter, skip_header=skip_top+1)
    if len(shape(data)) == 1:
        data = array([data,])
    head = genfromtxt(filename, delimiter=delimiter, dtype='U',
        skip_header=skip_top, skip_footer=shape(data)[0])
    nan_to_num(data, copy=False)
    TL = head == 'tempK'
    PL = head == 'pres'
    mols  = transpose(data[:, logical_and(~TL, ~PL)])
    ions  = head[logical_and(~TL, ~PL)]
    tempK = data[:, TL].ravel()
    pres = data[:, PL].ravel()
    return mols, ions, tempK, pres

def gettots(filename, delimiter=',', skip_top=0):
    """Import molality, temperature and pressure data from a CSV file, where
    some total concentrations are defined (i.e. with equilibration)."""
    data = genfromtxt(filename, delimiter=delimiter, skip_header=skip_top+1)
    if len(shape(data)) == 1:
        data = array([data,])
    head = genfromtxt(filename, delimiter=delimiter, dtype='U',
        skip_header=skip_top, skip_footer=shape(data)[0])
    nan_to_num(data, copy=False)
    TL = head == 'tempK'
    PL = head == 'pres'
    tempK = data[:, TL].ravel()
    pres = data[:, PL].ravel()
    data = data[:, logical_and(~TL, ~PL)].transpose()
    head = head[logical_and(~TL, ~PL)]
    eles = array([ele for ele in head if 't_' in ele])
    ions = array([ion for ion in head if 't_' not in ion])
    tots = array([tot for i, tot in enumerate(data)
        if 't_' in head[i]])
    mols = array([mol for i, mol in enumerate(data)
        if 't_' not in head[i]])
    return tots, mols, eles, ions, tempK, pres

def saveall(filename, mols, ions, tempK, pres, osm, aw, acfs):
    """Save molality, temperature, pressure, and calculated activity data
    to a CSV file.
    """
    savetxt(
        filename,
        concatenate((
                vstack(tempK),
                vstack(pres),
                transpose(mols),
                vstack(osm),
                vstack(aw),
                transpose(acfs)
            ), axis=1
        ),
        delimiter=',',
        header=','.join(concatenate((
            ['tempK', 'pres'], ions, ['osm', 'aw'],
            ['g'+ion for ion in ions])
        )),
        comments='')

def _u2v(mols, ions, tots, eles):
    """Get approximate conversion factor for molinity (mol/kg-solution) to
    molality (mol/kg-H2O).
    """
    ionmasses = array([_ion2mass[ion] for ion in ions])*mols.ravel()
    elemasses = (array([_ion2mass[_ele2ionmass[ele]] for ele in eles])*
        tots.ravel())
    totalsalts = (np_sum(ionmasses) + np_sum(elemasses))*1e-3 # kg
    u2v = 1 + totalsalts
    return u2v
    
def solution2solvent(mols, ions, tots, eles):
    """Roughly convert molinity (mol/kg-solution) to molality (mol/kg-H2O)."""
    u2v = _u2v(mols, ions, tots, eles)
    mols = mols*u2v
    tots = tots*u2v
    return mols, tots
    
def solvent2solution(mols, ions, tots, eles):
    """Roughly convert molality (mol/kg-H2O) to molinity (mol/kg-solution)."""
    u2v = _u2v(mols, ions, tots, eles)
    mols = mols/u2v
    tots = tots/u2v
    return mols, tots

def salinity2mols(salinity, MgOH=False):
    """Convert salinity (g/kg-sw) to molality for typical seawater, simplified
    for the WM13 tris buffer model, following MZF93.
    """
    if MgOH:
        mols = (array([0.44516, 0.01077, 0.01058, 0.56912])*
            salinity/43.189)
        ions = array(['Na', 'Ca', 'K', 'Cl'])
        tots = array([0.02926, 0.05518])*salinity/43.189
        eles = array(['t_HSO4', 't_Mg'])
    else:
        mols = (array([0.44516, 0.05518, 0.01077, 0.01058, 0.56912])*
            salinity/43.189)
        ions = array(['Na', 'Mg', 'Ca', 'K', 'Cl'])
        tots = array([0.02926])*salinity/43.189
        eles = array(['t_HSO4'])
    return mols, ions, tots, eles
