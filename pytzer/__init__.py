# pytzer: Pitzer model for chemical activities in aqueous solutions
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ['cfdicts', 'coeffs', 'constants', 'io', 'jfuncs', 'meta', 'model',
           'props', 'tables']

from . import cfdicts, coeffs, constants, io, jfuncs, meta, model, \
              props, tables

__version__ = meta.version

# Black box function
from copy import deepcopy
from numpy import concatenate, full_like, nan, savetxt, vstack
from numpy import any as np_any
from numpy import sum as np_sum

def blackbox(filename, cfdict=cfdicts.MarChemSpec, savefile=True):

    # Import test dataset
    mols,ions,T = io.getmols(filename)

    cf = deepcopy(cfdict)
    cf.add_zeros(ions) # just in case

    # Separate out zero ionic strengths
    zs = props.charges(ions)[0]
    I = vstack(0.5 * (np_sum(mols * zs**2, 1)))

    Gex_nRT = full_like(T   , nan)
    osm     = full_like(T   , nan)
    acfs    = full_like(mols, nan)

    L = (I > 0).ravel()

    nargsL  = (mols[ L,:], ions, T[ L], cf)
    nargsLx = (mols[~L,:], ions, T[~L], cf)

    # Do calculations
    print('Calculating excess Gibbs energies...')
    Gex_nRT[ L] = model.Gex_nRT(*nargsL)
    if np_any(~L):
        Gex_nRT[~L] = model.Gex_nRT(*nargsLx, Izero=True)

    print('Calculating osmotic coefficients...')
    osm[ L] = model.osm(*nargsL)
    if np_any(~L):
        osm[~L] = model.osm(*nargsLx, Izero=True)

    print('Calculating water activity...')
    aw = model.osm2aw(mols,osm)

    print('Calculating activity coefficients...')
    acfs[ L,:] = model.acfs(*nargsL)
    if np_any(~L):
        acfs[~L,:] = model.acfs(*nargsLx, Izero=True)

    # Save results unless requested not to
    if savefile:
        filestem = filename.replace('.csv','')
        savetxt(filestem + '_py.csv',
                concatenate((T,mols,osm,aw,acfs), axis=1),
                delimiter=',',
                header=','.join(concatenate((['temp'],ions,['osm','aw'],
                                             ['g'+ion for ion in ions]))),
                comments='')

    return mols,ions,T,cf,Gex_nRT,osm,aw,acfs
