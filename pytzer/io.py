# pytzer: Pitzer model for chemical activities in aqueous solutions
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)

from autograd.numpy import concatenate, genfromtxt, nan_to_num, savetxt, \
                           shape, transpose, vstack

#==============================================================================
#=================================================== Import molality data =====

def getmols(filename, delimiter=',', skip_top=0):

    data = genfromtxt(filename, delimiter=delimiter, skip_header=skip_top+1)
    head = genfromtxt(filename, delimiter=delimiter, dtype='U',
                      skip_header=skip_top, skip_footer=shape(data)[0])

    nan_to_num(data, copy=False)

    TL = head == 'tempK'

    mols  = transpose(data[:,~TL])
    ions  = head[~TL]
    tempK = data[:,TL].ravel()

    return mols, ions, tempK

# Save results
def saveall(filename, mols, ions, tempK, osm, aw, acfs):
    
    savetxt(filename,
            concatenate((vstack(tempK),
                         transpose(mols),
                         vstack(osm),
                         vstack(aw),
                         transpose(acfs)), 
                        axis=1),
            delimiter=',',
            header=','.join(concatenate((['tempK'], ions, ['osm','aw'],
                                         ['g'+ion for ion in ions]))),
            comments='')
