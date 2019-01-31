from copy import deepcopy
import pytzer as pz
import numpy as np 
from scipy.io import savemat

filename = 'testfiles/GenerateConcs.csv'

mols,ions,T = pz.io.getmols(filename)

cf = deepcopy(pz.cfdicts.MarChemSpec25)

#cf.lambd['tris-trisH'] = pz.coeffs.lambd_none

cf.add_zeros(ions) # just in case

cf.print_coeffs(298.15,'print_coeffs/' + cf.name + '.txt')

# Cut out zero ionic strengths and do calculations
zs = pz.props.charges(ions)[0]
I = np.vstack(0.5 * (np.sum(mols * zs**2, 1)))

Gex_nRT = np.full_like(T   , np.nan)
osm     = np.full_like(T   , np.nan)
acfs    = np.full_like(mols, np.nan)

L = (I > 0).ravel()

nargsL  = (mols[ L,:], ions, T[ L], cf)
nargsLx = (mols[~L,:], ions, T[~L], cf)

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

savemat('testfiles/threeway/threeway.mat',
        {'mols': mols,
         'ions': ions,
         'T'   : T   ,
         'acfs': acfs,
         'osm' : osm ,
         'aw'  : aw  ,
         'I'   : I   })
