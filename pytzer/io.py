# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
"""Import solution composition data, and export the results."""
from autograd.numpy import (array, concatenate, genfromtxt, logical_and,
    nan_to_num, savetxt, shape, transpose, vstack)

def getmols(filename, delimiter=',', skip_top=0):
    """Import molality, temperature and pressure data from a CSV file, where
    all ionic concentrations are defined (i.e. no equilibration)."""
    data = genfromtxt(filename, delimiter=delimiter, skip_header=skip_top+1)
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
