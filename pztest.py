import pytzer as pz
#import numpy as np 

filename = 'testfiles/GenerateConcs.csv'

mols,ions,T = pz.io.getmols(filename)

cf = pz.cfdicts.MarChemSpec

# Print out coefficient values to file

cf.print_coeffs(298.15,'print_coeffs/pytzer_MarChemSpec_25.txt')

print(cf.ions)

#cf.add_zeros(np.array(['Ba','Ca']))

## Cut out zero ionic strengths and do calculations
#
#zs = pz.props.charges(ions)[0]
#I = np.vstack(0.5 * (np.sum(mols * zs**2, 1)))
#
#Gex_nRT = np.full_like(T   , np.nan)
#osm     = np.full_like(T   , np.nan)
#acfs    = np.full_like(mols, np.nan)
#
#L = (I > 0).ravel()
#
#nargs = (mols[L,:], ions, T[L], cf)
#
#Gex_nRT[L  ] = pz.model.Gex_nRT(*nargs)
#osm    [L  ] = pz.model.osm    (*nargs)
#acfs   [L,:] = pz.model.acfs   (*nargs)
