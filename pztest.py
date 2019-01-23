import pytzer as pz
import numpy as np

filename = 'testfiles/GenerateConcs.csv'

mols,ions,T = pz.io.getmols(filename)

cf = pz.cfdicts.MarChemSpec

# Cut out zero ionic strengths

zs = pz.props.charges(ions)[0]
I = np.vstack(0.5 * (np.sum(mols * zs**2, 1)))

Gex_nRT = np.full_like(T   , np.nan)
osm     = np.full_like(T   , np.nan)
acfs    = np.full_like(mols, np.nan)

L = (I > 0).ravel()

nargs = (mols[L,:], ions, T[L], cf)

Gex_nRT[L  ] = pz.model.Gex_nRT(*nargs)
osm    [L  ] = pz.model.osm    (*nargs)
acfs   [L,:] = pz.model.acfs   (*nargs)
