from autograd.numpy import array, concatenate, unique, vstack
import pandas as pd

##### FILE I/O ################################################################

def getIons(filename):

    # Import input conditions from .csv
    idf = pd.read_csv(filename, float_precision='round_trip')

    # Replace missing values with zero
    idf = idf.fillna(0)

    # Get temperatures
    T = vstack(idf.temp.values)

    # Get ionic concentrations
    idf_tots = idf[idf.keys()[idf.keys() != 'temp']]
    tots = idf_tots.values

    # Get list of ions
    ions = array(idf_tots.keys())

    return T, tots, ions, idf

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
