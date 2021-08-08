import numpy as np
import pytzer as pz
from pytzer import equilibrate as eq


# Set tolerances for np.isclose()
tol = dict(atol=0, rtol=1e-4)


def test_pure_water():
    """Can we solve pure water?"""
    totals = pz.odict()
    ks_constants = {"H2O": 10 ** -14}
    ptargets = eq.stoichiometric.solve(totals, ks_constants)
    solutes = eq.components.get_solutes(totals, ks_constants, ptargets)
    assert len(ptargets) == 1
    assert len(solutes) == 2
    assert np.isclose(solutes["H"], solutes["OH"], **tol)
    assert np.isclose(solutes["H"] * solutes["OH"], ks_constants["H2O"], **tol)


def test_NaCl():
    """Can we solve NaCl?"""
    totals = pz.odict(Na=1.5, Cl=1.5)
    ks_constants = {"H2O": 10 ** -14}
    ptargets = eq.stoichiometric.solve(totals, ks_constants)
    solutes = eq.components.get_solutes(totals, ks_constants, ptargets)
    assert len(ptargets) == 1
    assert len(solutes) == 4
    assert np.isclose(solutes["H"], solutes["OH"], **tol)
    assert np.isclose(solutes["H"] * solutes["OH"], ks_constants["H2O"], **tol)
    assert np.isclose(solutes["Na"], totals["Na"], **tol)
    assert np.isclose(solutes["Cl"], totals["Cl"], **tol)


def test_NaCl_HCl():
    """Can we solve NaCl + HCl?"""
    totals = pz.odict(Na=1.5, Cl=3.5)
    ks_constants = {"H2O": 10 ** -14}
    ptargets = eq.stoichiometric.solve(totals, ks_constants)
    solutes = eq.components.get_solutes(totals, ks_constants, ptargets)
    assert len(ptargets) == 1
    assert len(solutes) == 4
    assert np.isclose(
        solutes["H"] + solutes["Na"], solutes["OH"] + solutes["Cl"], **tol
    )
    assert np.isclose(solutes["H"] * solutes["OH"], ks_constants["H2O"], **tol)
    assert np.isclose(solutes["Na"], totals["Na"], **tol)
    assert np.isclose(solutes["Cl"], totals["Cl"], **tol)


def test_NaCl_SO4_only():
    """Can we solve NaCl + SO4 without the HSO4 equilibrium?"""
    totals = pz.odict(Na=1.5, Cl=1.5, SO4=1.0)
    ks_constants = {"H2O": 10 ** -14}
    ptargets = eq.stoichiometric.solve(totals, ks_constants)
    solutes = eq.components.get_solutes(totals, ks_constants, ptargets)
    assert len(ptargets) == 1
    assert len(solutes) == 5
    assert np.isclose(
        solutes["H"] + solutes["Na"],
        solutes["OH"] + solutes["Cl"] + 2 * solutes["SO4"],
        **tol
    )
    assert np.isclose(solutes["H"] * solutes["OH"], ks_constants["H2O"], **tol)
    assert np.isclose(solutes["Na"], totals["Na"], **tol)
    assert np.isclose(solutes["Cl"], totals["Cl"], **tol)
    assert np.isclose(solutes["SO4"], totals["SO4"], **tol)


def test_NaCl_H2SO4():
    """Can we solve NaCl + SO4 with the HSO4 equilibrium?"""
    totals = pz.odict(Na=1.5, Cl=1.5, SO4=1.0)
    ks_constants = {"H2O": 10 ** -14, "HSO4": 10 ** -1}
    ptargets = eq.stoichiometric.solve(totals, ks_constants)
    solutes = eq.components.get_solutes(totals, ks_constants, ptargets)
    assert len(ptargets) == 1
    assert len(solutes) == 6
    assert np.isclose(
        solutes["H"] + solutes["Na"],
        solutes["OH"] + solutes["Cl"] + 2 * solutes["SO4"] + solutes["HSO4"],
        **tol
    )
    assert np.isclose(solutes["H"] * solutes["OH"], ks_constants["H2O"], **tol)
    assert np.isclose(
        solutes["SO4"] * solutes["H"] / solutes["HSO4"], ks_constants["HSO4"], **tol
    )
    assert np.isclose(solutes["Na"], totals["Na"], **tol)
    assert np.isclose(solutes["Cl"], totals["Cl"], **tol)
    assert np.isclose(solutes["SO4"] + solutes["HSO4"], totals["SO4"], **tol)


def test_NaCl_H2CO3():
    """Can we solve NaCl + CO2 with the H2CO3 equilibria?"""
    totals = pz.odict(Na=1.5, Cl=1.5, CO2=0.1)
    ks_constants = {
        "H2O": 10 ** -14,
        "H2CO3": 10 ** -5,
        "HCO3": 10 ** -9,
    }
    ptargets = eq.stoichiometric.solve(totals, ks_constants)
    solutes = eq.components.get_solutes(totals, ks_constants, ptargets)
    assert len(ptargets) == 1
    assert len(solutes) == 7
    assert np.isclose(
        solutes["H"] + solutes["Na"],
        solutes["OH"] + solutes["Cl"] + 2 * solutes["CO3"] + solutes["HCO3"],
        **tol
    )
    assert np.isclose(solutes["H"] * solutes["OH"], ks_constants["H2O"], **tol)
    assert np.isclose(
        solutes["CO3"] * solutes["H"] / solutes["HCO3"], ks_constants["HCO3"], **tol
    )
    assert np.isclose(
        solutes["HCO3"] * solutes["H"] / solutes["CO2"], ks_constants["H2CO3"], **tol
    )
    assert np.isclose(solutes["Na"], totals["Na"], **tol)
    assert np.isclose(solutes["Cl"], totals["Cl"], **tol)
    assert np.isclose(
        solutes["CO2"] + solutes["HCO3"] + solutes["CO3"], totals["CO2"], **tol
    )


def test_CaCl_H2CO3():
    """Can we solve CaCl + CO2 with the H2CO3 equilibria but no CaCO3?"""
    totals = pz.odict(Ca=0.75, Cl=1.5, CO2=0.1)
    ks_constants = {
        "H2O": 10 ** -14,
        "H2CO3": 10 ** -5,
        "HCO3": 10 ** -9,
    }
    ptargets = eq.stoichiometric.solve(totals, ks_constants)
    solutes = eq.components.get_solutes(totals, ks_constants, ptargets)
    assert len(ptargets) == 1
    assert len(solutes) == 7
    assert np.isclose(
        solutes["H"] + 2 * solutes["Ca"],
        solutes["OH"] + solutes["Cl"] + 2 * solutes["CO3"] + solutes["HCO3"],
        **tol
    )
    assert np.isclose(solutes["H"] * solutes["OH"], ks_constants["H2O"], **tol)
    assert np.isclose(
        solutes["CO3"] * solutes["H"] / solutes["HCO3"], ks_constants["HCO3"], **tol
    )
    assert np.isclose(
        solutes["HCO3"] * solutes["H"] / solutes["CO2"], ks_constants["H2CO3"], **tol
    )
    assert np.isclose(solutes["Ca"], totals["Ca"], **tol)
    assert np.isclose(solutes["Cl"], totals["Cl"], **tol)
    assert np.isclose(
        solutes["CO2"] + solutes["HCO3"] + solutes["CO3"], totals["CO2"], **tol
    )


def test_CaCl_H2CO3_CaCO3():
    """Can we solve CaCl + CO2 with the H2CO3 and CaCO3 equilibria?"""
    totals = pz.odict(Ca=0.75, Cl=1.5, CO2=0.1)
    ks_constants = {
        "H2O": 10 ** -14,
        "H2CO3": 10 ** -5,
        "HCO3": 10 ** -9,
        "CaCO3": 10 ** -4,
    }
    ptargets = eq.stoichiometric.solve(totals, ks_constants)
    solutes = eq.components.get_solutes(totals, ks_constants, ptargets)
    assert len(ptargets) == 2
    assert len(solutes) == 8
    assert np.isclose(
        solutes["H"] + 2 * solutes["Ca"],
        solutes["OH"] + solutes["Cl"] + 2 * solutes["CO3"] + solutes["HCO3"],
        **tol
    )
    assert np.isclose(solutes["H"] * solutes["OH"], ks_constants["H2O"], **tol)
    assert np.isclose(
        solutes["CO3"] * solutes["H"] / solutes["HCO3"], ks_constants["HCO3"], **tol
    )
    assert np.isclose(
        solutes["HCO3"] * solutes["H"] / solutes["CO2"], ks_constants["H2CO3"], **tol
    )
    assert np.isclose(
        solutes["CaCO3"] / (solutes["Ca"] * solutes["CO3"]),
        ks_constants["CaCO3"],
        **tol
    )
    assert np.isclose(solutes["Ca"] + solutes["CaCO3"], totals["Ca"], **tol)
    assert np.isclose(solutes["Cl"], totals["Cl"], **tol)
    assert np.isclose(
        solutes["CO2"] + solutes["HCO3"] + solutes["CO3"] + solutes["CaCO3"],
        totals["CO2"],
        **tol
    )


def test_all_ptargets():
    """Can we solve with all ptargets active?"""
    totals = pz.odict(Ca=2.0, Cl=4.0, CO2=0.1, PO4=0.5, F=1.0)
    ks_constants = {
        "H2O": 10 ** -14,
        "H2CO3": 10 ** -5,
        "HCO3": 10 ** -9,
        "HF": 10 ** -2,
        "H3PO4": 10 ** -11,
        "H2PO4": 10 ** -6,
        "HPO4": 10 ** -3,
        "CaCO3": 10 ** -4,
        "CaF": 10 ** 1,
        "CaH2PO4": 10 ** -7,
    }
    ptargets = eq.stoichiometric.solve(totals, ks_constants)
    solutes = eq.components.get_solutes(totals, ks_constants, ptargets)
    assert len(ptargets) == 4
    assert len(solutes) == 16
    assert np.isclose(solutes["H"] * solutes["OH"], ks_constants["H2O"], **tol)


# test_pure_water()
# test_NaCl()
# test_NaCl_HCl()
# test_NaCl_SO4_only()
# test_NaCl_H2SO4()
# test_NaCl_H2CO3()
# test_CaCl_H2CO3()
# test_CaCl_H2CO3_CaCO3()
# test_all_ptargets()
