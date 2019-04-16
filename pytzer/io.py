# Pytzer: Pitzer model for chemical activities in aqueous solutions
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)

"""Import data to use with pytzer, and export the results."""

from autograd.numpy import concatenate, genfromtxt, logical_and, nan_to_num, \
    savetxt, shape, transpose, vstack

#==============================================================================
#=================================================== Import molality data =====

def getmols(filename, delimiter=',', skip_top=0):
    """Import molality, temperature and pressure data from a CSV file."""
    data = genfromtxt(filename, delimiter=delimiter, skip_header=skip_top+1)
    head = genfromtxt(filename, delimiter=delimiter, dtype='U',
                      skip_header=skip_top, skip_footer=shape(data)[0])
    nan_to_num(data, copy=False)
    TL = head == 'tempK'
    PL = head == 'pres'
    mols  = transpose(data[:, logical_and(~TL, ~PL)])
    ions  = head[logical_and(~TL, ~PL)]
    tempK = data[:, TL].ravel()
    pres  = data[:, PL].ravel()
    return mols, ions, tempK, pres

# Save results
def saveall(filename, mols, ions, tempK, pres, osm, aw, acfs):
    """Save molality, temperature, pressure, and calculated activity data
    to a CSV file.
    """
    savetxt(filename,
            concatenate((vstack(tempK),
                         vstack(pres),
                         transpose(mols),
                         vstack(osm),
                         vstack(aw),
                         transpose(acfs)),
                        axis=1),
            delimiter=',',
            header=','.join(concatenate((
                ['tempK', 'pres'], ions, ['osm', 'aw'],
                ['g'+ion for ion in ions]))),
            comments='')
