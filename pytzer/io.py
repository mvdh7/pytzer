from autograd.numpy import array, concatenate, genfromtxt, nan_to_num, \
                           shape, unique, vstack

##### FILE I/O ################################################################

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

# Return list of unique ions from list of electrolytes
def ele2ions(ele):

    # Define ions in each electrolyte given full dissociation
    idict = {'BaCl2'  : array(['Ba', 'Cl' ]),
             'CaCl2'  : array(['Ca', 'Cl' ]),
             'H2SO4'  : array(['H' , 'SO4']),
             'KCl'    : array(['K' , 'Cl' ]),
             'KOH'    : array(['K' , 'OH' ]),
             'MgCl2'  : array(['Mg', 'Cl' ]),
             'Na2SO4' : array(['Na', 'SO4']),
             'NaCl'   : array(['Na', 'Cl' ]),
             'ZnBr2'  : array(['Zn', 'Br' ]),
             'ZnCl2'  : array(['Zn', 'Cl' ])}

    cats = unique(array([idict[ionpair][0] for ionpair in ele]).ravel())
    anis = unique(array([idict[ionpair][1] for ionpair in ele]).ravel())

    ions = concatenate((cats,anis))

    return ions, cats, anis, idict
