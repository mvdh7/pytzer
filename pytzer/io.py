# pytzer: the Pitzer model for chemical speciation
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)

from autograd.numpy import genfromtxt, nan_to_num, shape

#==============================================================================
#=================================================== Import molality data =====

def getmols(filename, delimiter=','):

    data = genfromtxt(filename, delimiter=delimiter, skip_header=1)
    head = genfromtxt(filename, delimiter=delimiter, dtype='U',
                      skip_footer=shape(data)[0])

    nan_to_num(data, copy=False)

    TL = head == 'temp'

    mols = data[:,~TL]
    ions = head[~TL]
    T = data[:,TL]

    return mols, ions, T
