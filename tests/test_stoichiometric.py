from jax import numpy as np

import pytzer as pz

# Select parameter library
pz.set_library(pz, "CWTD23")
# Set tolerances for np.isclose()
tol = dict(atol=0, rtol=1e-4)
temperature = 298.15
pressure = 10.1325


def test_pure_water():
    """Can we solve pure water?"""
    totals = {}
    ks_constants = {"H2O": 1e-14}
    thermo = pz.ks_to_thermo(ks_constants)
    ssr = pz.equilibrate.solver.solve_stoich(totals, temperature, pressure, thermo)
    solutes = pz.totals_to_solutes(totals, ssr.stoich, thermo)
    assert np.isclose(solutes["H"], solutes["OH"], **tol)
    assert np.isclose(solutes["H"] * solutes["OH"], ks_constants["H2O"], **tol)
    for solute, molality in solutes.items():
        if solute not in ["H", "OH"]:
            assert molality == 0.0
        else:
            assert molality > 0


def test_NaCl():
    """Can we solve NaCl?"""
    totals = dict(Na=1.5, Cl=1.5)
    ks_constants = {"H2O": 1e-14}
    thermo = pz.ks_to_thermo(ks_constants)
    ssr = pz.equilibrate.solver.solve_stoich(totals, temperature, pressure, thermo)
    solutes = pz.totals_to_solutes(totals, ssr.stoich, thermo)
    assert np.isclose(solutes["H"], solutes["OH"], **tol)
    assert np.isclose(solutes["H"] * solutes["OH"], ks_constants["H2O"], **tol)
    assert np.isclose(solutes["Na"], totals["Na"], **tol)
    assert np.isclose(solutes["Cl"], totals["Cl"], **tol)
    for solute, molality in solutes.items():
        if solute not in ["H", "OH", "Na", "Cl"]:
            assert molality == 0.0
        else:
            assert molality > 0


def test_NaCl_HCl():
    """Can we solve NaCl + HCl?"""
    totals = dict(Na=1.5, Cl=3.5)
    ks_constants = {"H2O": 1e-14}
    thermo = pz.ks_to_thermo(ks_constants)
    ssr = pz.equilibrate.solver.solve_stoich(
        totals, temperature, pressure, thermo, iter_stoich=20
    )
    solutes = pz.totals_to_solutes(totals, ssr.stoich, thermo)
    assert np.isclose(
        solutes["H"] + solutes["Na"], solutes["OH"] + solutes["Cl"], **tol
    )
    assert np.isclose(solutes["H"] * solutes["OH"], ks_constants["H2O"], **tol)
    assert np.isclose(solutes["Na"], totals["Na"], **tol)
    assert np.isclose(solutes["Cl"], totals["Cl"], **tol)
    for solute, molality in solutes.items():
        if solute not in ["H", "OH", "Na", "Cl"]:
            assert molality == 0.0
        else:
            assert molality > 0


def test_NaCl_HSO4():
    """Can we solve NaCl + SO4 with the HSO4 equilibrium?"""
    totals = dict(Na=1.5, Cl=1.5, SO4=1.0)
    ks_constants = {"H2O": 1e-14, "HSO4": 1e-1}
    thermo = pz.ks_to_thermo(ks_constants)
    ssr = pz.equilibrate.solver.solve_stoich(
        totals, temperature, pressure, thermo, iter_stoich=20
    )
    solutes = pz.totals_to_solutes(totals, ssr.stoich, thermo)
    assert np.isclose(
        solutes["H"] + solutes["Na"],
        solutes["OH"] + solutes["Cl"] + 2 * solutes["SO4"] + solutes["HSO4"],
        **tol,
    )
    assert np.isclose(solutes["H"] * solutes["OH"], ks_constants["H2O"], **tol)
    assert np.isclose(
        solutes["SO4"] * solutes["H"] / solutes["HSO4"], ks_constants["HSO4"], **tol
    )
    assert np.isclose(solutes["Na"], totals["Na"], **tol)
    assert np.isclose(solutes["Cl"], totals["Cl"], **tol)
    assert np.isclose(solutes["SO4"] + solutes["HSO4"], totals["SO4"], **tol)
    for solute, molality in solutes.items():
        if solute not in ["H", "OH", "Na", "Cl", "HSO4", "SO4"]:
            assert molality == 0.0
        else:
            assert molality > 0


def test_NaCl_H2CO3():
    """Can we solve NaCl + CO2 with the H2CO3 equilibria?"""
    pz.set_library(pz, "CWTD23")
    totals = dict(Na=1.5, Cl=1.5, CO2=0.1)
    ks_constants = {
        "H2O": 10**-14,
        "H2CO3": 10**-5,
        "HCO3": 10**-9,
    }
    thermo = pz.ks_to_thermo(ks_constants)
    ssr = pz.equilibrate.solver.solve_stoich(
        totals, temperature, pressure, thermo, iter_stoich=20
    )
    solutes = pz.totals_to_solutes(totals, ssr.stoich, thermo)
    assert np.isclose(
        solutes["H"] + solutes["Na"],
        solutes["OH"] + solutes["Cl"] + 2 * solutes["CO3"] + solutes["HCO3"],
        **tol,
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


def test_CaCl_H2CO3_CaCO3():
    """Can we solve CaCl + CO2 with the H2CO3 and CaCO3 equilibria?"""
    pz.set_library(pz, "CWTD23")
    totals = dict(Ca=0.75, Cl=1.5, CO2=0.1)
    ks_constants = {
        "H2O": 10**-14,
        "H2CO3": 10**-5,
        "HCO3": 10**-9,
        "CaCO3": 10**-4,
    }
    thermo = pz.ks_to_thermo(ks_constants)
    ssr = pz.equilibrate.solver.solve_stoich(
        totals, temperature, pressure, thermo, iter_stoich=20
    )
    solutes = pz.totals_to_solutes(totals, ssr.stoich, thermo)
    assert np.isclose(
        solutes["H"] + 2 * solutes["Ca"],
        solutes["OH"] + solutes["Cl"] + 2 * solutes["CO3"] + solutes["HCO3"],
        **tol,
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
        **tol,
    )
    assert np.isclose(solutes["Ca"] + solutes["CaCO3"], totals["Ca"], **tol)
    assert np.isclose(solutes["Cl"], totals["Cl"], **tol)
    assert np.isclose(
        solutes["CO2"] + solutes["HCO3"] + solutes["CO3"] + solutes["CaCO3"],
        totals["CO2"],
        **tol,
    )


def test_all_ptargets():
    """Can we solve with all ptargets active?"""
    totals = dict(Ca=2.0, Cl=4.0, CO2=0.1, F=1.0)
    ks_constants = {
        "H2O": 10**-14,
        "H2CO3": 10**-5,
        "HCO3": 10**-9,
        "HF": 10**-2,
        "CaCO3": 10**-4,
        "CaF": 10**1,
    }
    thermo = pz.ks_to_thermo(ks_constants)
    ssr = pz.equilibrate.solver.solve_stoich(
        totals, temperature, pressure, thermo, iter_stoich=30
    )
    solutes = pz.totals_to_solutes(totals, ssr.stoich, thermo)
    assert np.isclose(solutes["H"] * solutes["OH"], ks_constants["H2O"], **tol)


# test_pure_water()
# test_NaCl()
# test_NaCl_HCl()
# test_NaCl_HSO4()
# test_NaCl_H2CO3()
# test_CaCl_H2CO3_CaCO3()
# test_all_ptargets()
