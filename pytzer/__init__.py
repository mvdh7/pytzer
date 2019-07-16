# Pytzer: Pitzer model for chemical activities in aqueous solutions.
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
"""Pitzer model for chemical activities in aqueous solutions."""
from . import (cflibs, coefficients, constants, debyehueckel, dissociation, io,
    jfuncs, matrix, meta, model, properties, tables, teos10)

__all__ = [
    'cflibs',
    'coefficients',
    'constants',
    'debyehueckel',
    'dissociation',
    'io',
    'jfuncs',
    'matrix',
    'meta',
    'model',
    'properties',
    'tables',
    'teos10',
]
__version__ = meta.version
__author__ = 'Matthew P. Humphreys'

from copy import deepcopy
from numpy import full_like, nan
from numpy import any as np_any

def blackbox(filename, cflib=cflibs.Seawater, savefile=True):
    """Import a CSV file with molality data, calculate all activity
    coefficients, and save results to a new CSV file.
    """
    # Import test dataset
    mols, ions, tempK, pres = io.getmols(filename)
    cflib = deepcopy(cflib)
    cflib.add_zeros(ions) # just in case
    # Separate out zero ionic strengths
    zs = properties.charges(ions)[0]
    I = model.Istr(mols, zs)
    Gex_nRT = full_like(tempK, nan)
    osm = full_like(tempK, nan)
    aw = full_like(tempK, nan)
    acfs = full_like(mols, nan)
    L = I > 0
    nargsL  = (mols[:,  L], ions, tempK[ L], pres[ L], cflib)
    nargsLx = (mols[:, ~L], ions, tempK[~L], pres[~L], cflib)
    # Do calculations
    print('Calculating excess Gibbs energies...')
    Gex_nRT[L] = model.Gex_nRT(*nargsL)
    if np_any(~L):
        Gex_nRT[~L] = model.Gex_nRT(*nargsLx, Izero=True)
    print('Calculating osmotic coefficients...')
    osm[L] = model.osm(*nargsL)
    if np_any(~L):
        osm[~L] = model.osm(*nargsLx, Izero=True)
    print('Calculating water activity...')
    aw[L] = model.aw(*nargsL)
    if np_any(~L):
        aw[~L] = model.aw(*nargsLx, Izero=True)
    print('Calculating activity coefficients...')
    acfs[:,L] = model.acfs(*nargsL)
    if np_any(~L):
        acfs[:, ~L] = model.acfs(*nargsLx, Izero=True)
    # Save results, unless requested not to
    if savefile:
        filestem = filename.replace('.csv','')
        io.saveall(filestem + '_py.csv',
            mols, ions, tempK, pres, osm, aw, acfs)
    print('Finished!')
    return mols, ions, tempK, pres, cflib, Gex_nRT, osm, aw, acfs
