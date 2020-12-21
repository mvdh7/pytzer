# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2020  Matthew Paul Humphreys  (GNU GPLv3)
"""Convenient wrappers for the main Pitzer model functions."""
from . import model, prepare
from .libraries import Seawater


def Gibbs_nRT(
    solute_molalities,
    temperature=298.15,
    pressure=10.10325,
    library=Seawater,
    verbose=True,
):
    """Calculate the excess Gibbs energy of a solution divided by n*R*T."""
    args, ss = prepare.get_pytzer_args(solute_molalities)
    params = library.get_parameters(
        **ss, temperature=temperature, pressure=pressure, verbose=verbose
    )
    return model.Gibbs_nRT(*args, **params)


def activity_coefficients(
    solute_molalities,
    temperature=298.15,
    pressure=10.10325,
    library=Seawater,
    verbose=True,
):
    """Calculate the activity coefficient of all solutes."""
    args, ss = prepare.get_pytzer_args(solute_molalities)
    params = library.get_parameters(
        **ss, temperature=temperature, pressure=pressure, verbose=verbose
    )
    return model.activity_coefficients(*args, **params)


def activity_water(
    solute_molalities,
    temperature=298.15,
    pressure=10.10325,
    library=Seawater,
    verbose=True,
):
    """Calculate the water activity."""
    args, ss = prepare.get_pytzer_args(solute_molalities)
    params = library.get_parameters(
        **ss, temperature=temperature, pressure=pressure, verbose=verbose
    )
    return model.activity_water(*args, **params)


def log_activity_coefficients(
    solute_molalities,
    temperature=298.15,
    pressure=10.10325,
    library=Seawater,
    verbose=True,
):
    """Calculate the natural log of the activity coefficient of all solutes."""
    args, ss = prepare.get_pytzer_args(solute_molalities)
    params = library.get_parameters(
        **ss, temperature=temperature, pressure=pressure, verbose=verbose
    )
    return model.log_activity_coefficients(*args, **params)


def log_activity_water(
    solute_molalities,
    temperature=298.15,
    pressure=10.10325,
    library=Seawater,
    verbose=True,
):
    """Calculate the natural log of the water activity."""
    args, ss = prepare.get_pytzer_args(solute_molalities)
    params = library.get_parameters(
        **ss, temperature=temperature, pressure=pressure, verbose=verbose
    )
    return model.log_activity_water(*args, **params)


def osmotic_coefficient(
    solute_molalities,
    temperature=298.15,
    pressure=10.10325,
    library=Seawater,
    verbose=True,
):
    """Calculate the osmotic coefficient of the solution."""
    args, ss = prepare.get_pytzer_args(solute_molalities)
    params = library.get_parameters(
        **ss, temperature=temperature, pressure=pressure, verbose=verbose
    )
    return model.osmotic_coefficient(*args, **params)
