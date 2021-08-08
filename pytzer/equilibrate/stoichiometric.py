# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
"""Solve for stoichiometric equilibrium."""

from collections import OrderedDict
import jax
from jax import lax, numpy as np
from . import components


@jax.jit
def guess_ptargets(ptargets, totals):
    """Update ptargets with vaguely sensible first guesses for the stoichiometric
    solver.
    """
    assert isinstance(ptargets, OrderedDict)
    for s in ptargets:
        if s == "H":
            ptargets["H"] = 8.0
        elif s == "F":
            assert "F" in totals
            ptargets["F"] = -np.log10(totals["F"] / 2)
        elif s == "CO3":
            assert "CO2" in totals
            ptargets["CO3"] = -np.log10(totals["CO2"] / 10)
        elif s == "PO4":
            assert "PO4" in totals
            ptargets["PO4"] = -np.log10(totals["PO4"] / 2)
    return ptargets


@jax.jit
def create_ptargets(totals, ks_constants):
    """Generate ptargets with first-guess solver values."""
    ptargets = OrderedDict()
    ptargets["H"] = 0.0
    if "F" in totals:
        if "MgF" in ks_constants or "CaF" in ks_constants:
            ptargets["F"] = 0.0
    if "CO2" in totals:
        if (
            "MgCO3" in ks_constants
            or "CaCO3" in ks_constants
            or "SrCO3" in ks_constants
        ):
            ptargets["CO3"] = 0.0
    if "PO4" in totals:
        if (
            "MgH2PO4" in ks_constants
            or "MgHPO4" in ks_constants
            or "MgPO4" in ks_constants
            or "CaH2PO4" in ks_constants
            or "CaHPO4" in ks_constants
            or "CaPO4" in ks_constants
        ):
            ptargets["PO4"] = 0.0
    ptargets = guess_ptargets(ptargets, totals)
    return ptargets


def get_alkalinity(solutes):
    """Calculate 'Dickson' alkalinity."""

    def add_if_in(key):
        if key in solutes:
            return solutes[key]
        else:
            return 0

    return (
        add_if_in("OH")
        - add_if_in("H")
        + add_if_in("MgOH")
        - add_if_in("HF")
        + add_if_in("HCO3")
        + add_if_in("CO3") * 2
        + add_if_in("HPO4")
        + add_if_in("PO4") * 2
        - add_if_in("H3PO4")
        + add_if_in("MgCO3") * 2
        + add_if_in("CaCO3") * 2
        + add_if_in("SrCO3") * 2
        + add_if_in("MgHPO4")
        + add_if_in("MgPO4") * 2
        + add_if_in("CaHPO4")
        + add_if_in("CaPO4") * 2
        - add_if_in("HSO4")
        + add_if_in("HS")
        + add_if_in("BOH4")
        + add_if_in("NH3")
        + add_if_in("H3SiO4")
        - add_if_in("HNO2")
    )


def get_explicit_alkalinity(totals):
    """Calculate explicit total alkalinity."""

    def add_if_in(key):
        if key in totals:
            return totals[key]
        else:
            return 0

    return (
        add_if_in("Na")
        + add_if_in("K")
        - add_if_in("Cl")
        - add_if_in("Br")
        + add_if_in("Mg") * 2
        + add_if_in("Ca") * 2
        + add_if_in("Sr") * 2
        - add_if_in("F")
        - add_if_in("PO4")
        - add_if_in("SO4") * 2
        + add_if_in("NH3")
        - add_if_in("NO2")
    )


def get_total_F(solutes):
    def add_if_in(key):
        if key in solutes:
            return solutes[key]
        else:
            return 0

    return add_if_in("F") + add_if_in("HF") + add_if_in("MgF") + add_if_in("CaF")


def get_total_CO2(solutes):
    def add_if_in(key):
        if key in solutes:
            return solutes[key]
        else:
            return 0

    return (
        add_if_in("CO2")
        + add_if_in("HCO3")
        + add_if_in("CO3")
        + add_if_in("CaCO3")
        + add_if_in("MgCO3")
        + add_if_in("SrCO3")
    )


def get_total_PO4(solutes):
    def add_if_in(key):
        if key in solutes:
            return solutes[key]
        else:
            return 0

    return (
        add_if_in("PO4")
        + add_if_in("HPO4")
        + add_if_in("H2PO4")
        + add_if_in("H3PO4")
        + add_if_in("MgPO4")
        + add_if_in("MgHPO4")
        + add_if_in("MgH2PO4")
        + add_if_in("CaPO4")
        + add_if_in("CaHPO4")
        + add_if_in("CaH2PO4")
    )


all_total_targets = {
    "H": lambda totals: get_explicit_alkalinity(totals),
    "F": lambda totals: totals["F"],
    "CO3": lambda totals: totals["CO2"],
    "PO4": lambda totals: totals["PO4"],
}


def get_total_targets(totals, ptargets):
    return OrderedDict((pf, all_total_targets[pf](totals)) for pf in ptargets)


all_solute_targets = {
    "H": get_alkalinity,
    "F": get_total_F,
    "CO3": get_total_CO2,
    "PO4": get_total_PO4,
}


def get_solute_targets(solutes, ptargets):
    return OrderedDict((pf, all_solute_targets[pf](solutes)) for pf in ptargets)


def solver_func(ptargets_values, ptargets, totals, ks_constants):
    ptargets = OrderedDict(
        (k, ptargets_values[i]) for i, k in enumerate(ptargets.keys())
    )
    total_targets = get_total_targets(totals, ptargets)
    solutes = components.get_solutes(totals, ks_constants, ptargets)
    solute_targets = get_solute_targets(solutes, ptargets)
    targets = np.array(
        [total_targets[pf] - solute_targets[pf] for pf in ptargets.keys()]
    )
    return targets


solver_jac = jax.jit(jax.jacfwd(solver_func))


@jax.jit
def solve(totals, ks_constants, ptargets=None):
    def cond(ptargets_values):
        target = solver_func(ptargets_values, ptargets, totals, ks_constants)
        return np.any(np.abs(target) > 1e-9)

    def body(ptargets_values):
        target = -solver_func(ptargets_values, ptargets, totals, ks_constants)
        jac = solver_jac(ptargets_values, ptargets, totals, ks_constants)
        p_diff = np.linalg.solve(jac, target)
        p_diff = np.where(p_diff > 1, 1, p_diff)
        p_diff = np.where(p_diff < -1, -1, p_diff)
        return ptargets_values + p_diff

    if ptargets is None:
        ptargets = create_ptargets(totals, ks_constants)
    ptargets_values = np.array([v for v in ptargets.values()])
    ptargets_values = lax.while_loop(cond, body, ptargets_values)
    ptargets_final = OrderedDict(
        (k, ptargets_values[i]) for i, k in enumerate(ptargets.keys())
    )
    return ptargets_final
