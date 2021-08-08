# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
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
    equilibrate,
    io,
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
from .equilibrate import solve, solve_manual
from .equilibrate.components import find_solutes
from .equilibrate.stoichiometric import solve as solve_stoichiometric
from .equilibrate.thermodynamic import solve as solve_thermodynamic
from .io import solve_df
from .libraries import ParameterLibrary
from .meta import hello, update_func_J
from .model import (
    activity_coefficients,
    activity_water,
    Gibbs_nRT,
    log_activity_coefficients,
    log_activity_water,
    osmotic_coefficient,
)
from collections import OrderedDict as odict

__version__ = meta.version
__author__ = meta.author

say_hello = hello
