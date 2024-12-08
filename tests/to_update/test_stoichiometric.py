# %%
import numpy as onp
from jax import numpy as np

import pytzer as pz

# Set tolerances for np.isclose()
tol = dict(atol=0, rtol=1e-4)

totals = dict()
temperature = 298.15
pressure = 10.1325
thermo = onp.ones(len(pz.library.equilibria_all))
thermo[pz.library.equilibria_all.index("H2O")] = 1e-14
thermo = np.log(thermo)

pz.equilibrate.solver.solve_stoich(
    totals,
    temperature,
    pressure,
    thermo,
    stoich=None,
    iter_stoich=10,
    verbose=False,
    warn_cutoff=1e-8,
)

# %%
stoich = None
iter_stoich = 10
verbose = True
warn_cutoff = 1e-8

totals.update({t: 0.0 for t in pz.library.totals_all if t not in totals})
stoich_targets = pz.library.get_stoich_targets(totals)
if stoich is None:
    stoich = pz.library.stoich_init(totals)
stoich = np.array([8.0, np.inf, np.inf])
# Solve!
if verbose:
    print("STOICH", 0)
    print(stoich)
for _s in range(iter_stoich):
    stoich_adjust = pz.equilibrate.solver.get_stoich_adjust(
        stoich, totals, thermo, stoich_targets
    )
    if verbose:
        print("STOICH", _s + 1)
        print(stoich)
        print(stoich_adjust)
    stoich = stoich + stoich_adjust
# if np.any(np.abs(stoich_adjust) > warn_cutoff):
#     warnings.warn(
#         "Solver did not converge below `warn_cutoff` - "
#         + "try increasing `iter_stoich`."
#     )


# %%
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
    totals = pz.odict(Na=1.5, Cl=1.5, CO2=0.1)
    ks_constants = {
        "H2O": 10**-14,
        "H2CO3": 10**-5,
        "HCO3": 10**-9,
    }
    ptargets = eq.stoichiometric.solve(totals, ks_constants)
    solutes = eq.components.get_solutes(totals, ks_constants, ptargets)
    assert len(ptargets) == 1
    assert len(solutes) == 7
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


def test_CaCl_H2CO3():
    """Can we solve CaCl + CO2 with the H2CO3 equilibria but no CaCO3?"""
    totals = pz.odict(Ca=0.75, Cl=1.5, CO2=0.1)
    ks_constants = {
        "H2O": 10**-14,
        "H2CO3": 10**-5,
        "HCO3": 10**-9,
    }
    ptargets = eq.stoichiometric.solve(totals, ks_constants)
    solutes = eq.components.get_solutes(totals, ks_constants, ptargets)
    assert len(ptargets) == 1
    assert len(solutes) == 7
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
    assert np.isclose(solutes["Ca"], totals["Ca"], **tol)
    assert np.isclose(solutes["Cl"], totals["Cl"], **tol)
    assert np.isclose(
        solutes["CO2"] + solutes["HCO3"] + solutes["CO3"], totals["CO2"], **tol
    )


def test_CaCl_H2CO3_CaCO3():
    """Can we solve CaCl + CO2 with the H2CO3 and CaCO3 equilibria?"""
    totals = pz.odict(Ca=0.75, Cl=1.5, CO2=0.1)
    ks_constants = {
        "H2O": 10**-14,
        "H2CO3": 10**-5,
        "HCO3": 10**-9,
        "CaCO3": 10**-4,
    }
    ptargets = eq.stoichiometric.solve(totals, ks_constants)
    solutes = eq.components.get_solutes(totals, ks_constants, ptargets)
    assert len(ptargets) == 2
    assert len(solutes) == 8
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
    totals = pz.odict(Ca=2.0, Cl=4.0, CO2=0.1, PO4=0.5, F=1.0)
    ks_constants = {
        "H2O": 10**-14,
        "H2CO3": 10**-5,
        "HCO3": 10**-9,
        "HF": 10**-2,
        "H3PO4": 10**-11,
        "H2PO4": 10**-6,
        "HPO4": 10**-3,
        "CaCO3": 10**-4,
        "CaF": 10**1,
        "CaH2PO4": 10**-7,
    }
    ptargets = eq.stoichiometric.solve(totals, ks_constants)
    solutes = eq.components.get_solutes(totals, ks_constants, ptargets)
    assert len(ptargets) == 4
    assert len(solutes) == 16
    assert np.isclose(solutes["H"] * solutes["OH"], ks_constants["H2O"], **tol)


def test_get_constants():
    """Does the get_constants function agree with the full thermodynamic solver for
    non-zero total molalities?
    """
    totals = pz.odict(Na=4.0, Cl=4.0, CO2=0.1, PO4=0.5, F=1.0)
    solutes, pks_constants = pz.solve(totals)
    which_constants = ["H2CO3", "HCO3", "HF"]
    f_ks_constants = eq.stoichiometric.get_constants(
        solutes, which_constants=which_constants
    )
    f_pks_constants = {k: -np.log10(v) for k, v in f_ks_constants.items()}
    for c in which_constants:
        assert np.isclose(pks_constants[c], f_pks_constants[c], rtol=0, atol=1e-8)


def test_get_constants_zero():
    """Can we retrieve the carbonic acid  and HF stoichiometric equilibrium constants
    when the total dissolved inorganic carbon and fluoride are zero?
    """
    # Solve without DIC and HF
    totals = pz.odict(Na=1.5, Cl=3.5)
    ks_constants = {"H2O": 10**-14}
    ptargets = eq.stoichiometric.solve(totals, ks_constants)
    solutes = eq.components.get_solutes(totals, ks_constants, ptargets)
    # Get DIC and HF equilibria
    which_constants = ["H2CO3", "HCO3", "HF"]
    dic_eq = eq.stoichiometric.get_constants(solutes, which_constants=which_constants)
    assert len(dic_eq) == len(which_constants)
    for c in which_constants:
        assert c in dic_eq
        assert isinstance(dic_eq[c].item(), float)


# test_pure_water()
# test_NaCl()
# test_NaCl_HCl()
# test_NaCl_SO4_only()
# test_NaCl_HSO4()
# test_NaCl_H2CO3()
# test_CaCl_H2CO3()
# test_CaCl_H2CO3_CaCO3()
# test_all_ptargets()
# test_get_constants()
# test_get_constants_zero()
