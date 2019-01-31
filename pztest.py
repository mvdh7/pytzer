from copy import deepcopy
import pytzer as pz
import numpy as np

# Import test dataset
filename = 'testfiles/GenerateConcs.csv'
mols,ions,T = pz.io.getmols(filename)

# Change temperature
TEMP = np.float_(278.15)
T[:] = TEMP

# Set up CoefficientDictionary
cf = deepcopy(pz.cfdicts.MarChemSpec05)
cf.add_zeros(ions) # just in case

# Print out coefficients at 298.15 K
cf.print_coeffs(TEMP,'print_coeffs/' + cf.name + '.txt')

# Separate out zero ionic strengths
zs = pz.props.charges(ions)[0]
I = np.vstack(0.5 * (np.sum(mols * zs**2, 1)))

Gex_nRT = np.full_like(T   , np.nan)
osm     = np.full_like(T   , np.nan)
acfs    = np.full_like(mols, np.nan)

L = (I > 0).ravel()

nargsL  = (mols[ L,:], ions, T[ L], cf)
nargsLx = (mols[~L,:], ions, T[~L], cf)

# Do calculations
print('Calculating excess Gibbs energies...')
Gex_nRT[ L] = pz.model.Gex_nRT(*nargsL)
Gex_nRT[~L] = pz.model.Gex_nRT(*nargsLx, Izero=True)

print('Calculating osmotic coefficients...')
osm[ L] = pz.model.osm(*nargsL)
osm[~L] = pz.model.osm(*nargsLx, Izero=True)

print('Calculating water activity...')
aw = pz.model.osm2aw(mols,osm)

print('Calculating activity coefficients...')
acfs[ L,:] = pz.model.acfs(*nargsL)
acfs[~L,:] = pz.model.acfs(*nargsLx, Izero=True)

# Save results for plotting in MATLAB
from scipy.io import savemat
savemat('testfiles/threeway/threeway05.mat',
        {'mols': mols,
         'ions': ions,
         'T'   : T   ,
         'acfs': acfs,
         'osm' : osm ,
         'aw'  : aw  ,
         'I'   : I   })
