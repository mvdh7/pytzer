# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2020  Matthew Paul Humphreys  (GNU GPLv3)
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
from . import (
    constants,
    convert,
    debyehueckel,
    dissociation,
    # equilibrate,
    # io,
    libraries,
    # measure,
    meta,
    model,
    parameters,
    # plot,
    # potentials,
    # properties,
    # teos10,
    unsymmetrical,
)
from .convert import (
    activity_to_osmotic,
    log_activities_to_mean,
    osmotic_to_activity,
)
from .libraries import ParameterLibrary
from .meta import update_func_J
from .model import (
    activity_coefficients,
    activity_water,
    Gibbs_nRT,
    log_activity_coefficients,
    log_activity_water,
    osmotic_coefficient,
    split_molalities_charges,
)

__version__ = meta.version
__author__ = "Matthew P. Humphreys"


def hello():
    print("This is Pytzer v{}.".format(__version__))


say_hello = hello


# from copy import deepcopy
# from numpy import full_like, nan
# from numpy import any as np_any

# def blackbox(filename, prmlib=libraries.MIAMI, savefile=True):
#     """Import a CSV file with molality data, calculate all activity
#     coefficients, and save results to a new CSV file.
#     """
#     # Import test dataset
#     mols, ions, tempK, pres = io.getmols(filename)
#     prmlib = deepcopy(prmlib)
#     prmlib.add_zeros(ions) # just in case
#     # Separate out zero ionic strengths
#     zs = properties.charges(ions)[0]
#     I = model.Istr(mols, zs)
#     Gex_nRT = full_like(tempK, nan)
#     osm = full_like(tempK, nan)
#     aw = full_like(tempK, nan)
#     acfs = full_like(mols, nan)
#     L = I > 0
#     nargsL  = (mols[:,  L], ions, tempK[ L], pres[ L], prmlib)
#     nargsLx = (mols[:, ~L], ions, tempK[~L], pres[~L], prmlib)
#     # Do calculations
#     print('Calculating excess Gibbs energies...')
#     Gex_nRT[L] = model.Gex_nRT(*nargsL)
#     if np_any(~L):
#         Gex_nRT[~L] = model.Gex_nRT(*nargsLx, Izero=True)
#     print('Calculating osmotic coefficients...')
#     osm[L] = model.osm(*nargsL)
#     if np_any(~L):
#         osm[~L] = model.osm(*nargsLx, Izero=True)
#     print('Calculating water activity...')
#     aw[L] = model.aw(*nargsL)
#     if np_any(~L):
#         aw[~L] = model.aw(*nargsLx, Izero=True)
#     print('Calculating activity coefficients...')
#     acfs[:,L] = model.acfs(*nargsL)
#     if np_any(~L):
#         acfs[:, ~L] = model.acfs(*nargsLx, Izero=True)
#     # Save results, unless requested not to
#     if savefile:
#         filestem = filename.replace('.csv','')
#         io.saveall(filestem + '_py.csv',
#             mols, ions, tempK, pres, osm, aw, acfs)
#     print('Finished!')
#     return mols, ions, tempK, pres, prmlib, Gex_nRT, osm, aw, acfs

# def blackbox_equilibrate(filename, prmlib=libraries.MIAMI, savefile=True):
#     """Import a CSV file with molality data, solve for equilibrium speciation,
#     calculate all activity coefficients, and save results to a new CSV file.
#     """
#     # Import test dataset
#     tots, fixmols, eles, fixions, tempK, pres = io.gettots(filename)
#     allions = properties.getallions(eles, fixions)
#     prmlib = deepcopy(prmlib)
#     prmlib.add_zeros(allions) # just in case
#     # Solve for equilibria
#     q = 0
#     for ele in eles:
#         q += len(properties._eq2ions[ele]) - 1
#     eqstate_guess = [0.0 for _ in range(q)]
#     if q == 0:
#         eqstate_guess = [30.0,]
#     else:
#         eqstate_guess.append(30.0)
#     allmols, allions, eqstates = equilibrate.solveloop(eqstate_guess, tots,
#         fixmols, eles, fixions, tempK, pres, prmlib=prmlib)
#     # Separate out zero ionic strengths
#     zs = properties.charges(allions)[0]
#     I = model.Istr(allmols, zs)
#     Gex_nRT = full_like(tempK, nan)
#     osm = full_like(tempK, nan)
#     aw = full_like(tempK, nan)
#     acfs = full_like(allmols, nan)
#     L = I > 0
#     nargsL  = (allmols[:,  L], allions, tempK[ L], pres[ L], prmlib)
#     nargsLx = (allmols[:, ~L], allions, tempK[~L], pres[~L], prmlib)
#     # Do calculations
#     print('Calculating excess Gibbs energies...')
#     Gex_nRT[L] = model.Gex_nRT(*nargsL)
#     if np_any(~L):
#         Gex_nRT[~L] = model.Gex_nRT(*nargsLx, Izero=True)
#     print('Calculating osmotic coefficients...')
#     osm[L] = model.osm(*nargsL)
#     if np_any(~L):
#         osm[~L] = model.osm(*nargsLx, Izero=True)
#     print('Calculating water activity...')
#     aw[L] = model.aw(*nargsL)
#     if np_any(~L):
#         aw[~L] = model.aw(*nargsLx, Izero=True)
#     print('Calculating activity coefficients...')
#     acfs[:,L] = model.acfs(*nargsL)
#     if np_any(~L):
#         acfs[:, ~L] = model.acfs(*nargsLx, Izero=True)
#     # Save results, unless requested not to
#     if savefile:
#         filestem = filename.replace('.csv','')
#         io.saveall(filestem + '_py.csv',
#             allmols, allions, tempK, pres, osm, aw, acfs)
#     print('Finished!')
#     return (allmols, allions, tempK, pres, prmlib, Gex_nRT, osm, aw, acfs,
#         eqstates)
