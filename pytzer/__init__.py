# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2024  M.P. Humphreys  (GNU GPLv3)
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
"""
Pytzer
======
Pytzer is an implementation of the Pitzer model for chemical activities in aqueous
solutions in Python, including an equilibrium solver.
"""

from jax import numpy as np

from . import (
    constants,
    convert,
    debyehueckel,
    dissociation,
    equilibrate,
    get,
    libraries,
    meta,
    model,
    parameters,
    prepare,
    properties,
    teos10,
    unsymmetrical,
)
from .convert import (
    activity_to_osmotic,
    log_activities_to_mean,
    osmotic_to_activity,
)
from .equilibrate.solver import ks_to_thermo, solve, solve_stoich
from .get import solve_df
from .libraries import Library
from .meta import hello, set_library
from .model import (
    Gibbs_nRT,
    activity_coefficients,
    activity_water,
    log_activity_coefficients,
    log_activity_water,
    osmotic_coefficient,
)

# Assign shortcuts
libraries_all = list(libraries.libraries_all.keys())
library = model.library
get_solutes = library.get_solutes
get_totals = library.get_totals
totals_to_solutes = library.totals_to_solutes

# General package info
__version__ = meta.version
__author__ = meta.author
