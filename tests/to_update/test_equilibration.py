# %%
from datetime import datetime

import pytzer as pz

# Select parameter library
pz.set_library(pz, "CWTD23")

totals = pz.get_totals()
totals.update(
    {
        "CO2": 0.002,
        "Na": 0.5023,
        "K": 0.081,
        "Cl": 0.6,
        "BOH3": 0.0004,
        "SO4": 0.02,
        "F": 0.001,
        "Sr": 0.02,
        "Mg": 0.01,
        "Ca": 0.05,
        "Br": 0.1,
    }
)
temperature = 298.15
pressure = 10.1325

start = datetime.now()
scr = pz.solve(totals, 298.15, 10.1325)
print("pz.solve", datetime.now() - start)
solutes = pz.totals_to_solutes(totals, scr.stoich, scr.thermo)

# start = datetime.now()
# scr_scan = pz.solve_scan(totals, 298.15, 10.1325)
# print(datetime.now() - start)

import warnings

import jax
from jax import numpy as np

from pytzer import library
from pytzer.equilibrate.solver import SolveResult, get_stoich_adjust, get_thermo_adjust

stoich = None
thermo = None
iter_stoich_per_thermo = 3
iter_thermo = 6

# Solver targets---known from the start
totals = totals.copy()
totals.update({t: 0.0 for t in library.totals_all if t not in totals})
stoich_targets = library.get_stoich_targets(totals)
thermo_targets = np.array(
    [library.equilibria[eq](temperature) for eq in library.equilibria_all]
)  # these are ln(k)
if stoich is None:
    stoich = library.stoich_init(totals)
if thermo is None:
    thermo = thermo_targets.copy()


def scanner_stoich(carry, x):
    stoich, stoich_adjust, thermo = carry
    stoich_adjust = get_stoich_adjust(stoich, totals, thermo, stoich_targets)
    stoich = stoich + stoich_adjust
    return (stoich, stoich_adjust, thermo), x


def scanner_thermo(carry, x):
    stoich, stoich_adjust, thermo, thermo_adjust = carry
    for _s in range(iter_stoich_per_thermo):
        stoich_adjust = get_stoich_adjust(stoich, totals, thermo, stoich_targets)
        stoich = stoich + stoich_adjust
    thermo_adjust = get_thermo_adjust(
        thermo, totals, temperature, pressure, stoich, thermo_targets
    )
    thermo = thermo + thermo_adjust
    return (stoich, stoich_adjust, thermo, thermo_adjust), x


def scanner_full(carry, x):
    stoich, stoich_adjust, thermo, thermo_adjust = carry
    scanner_s = jax.lax.scan(
        scanner_stoich,
        (stoich, np.zeros_like(stoich), thermo),
        length=iter_stoich_per_thermo,
    )[0]
    stoich, stoich_adjust, thermo = scanner_s
    thermo_adjust = get_thermo_adjust(
        thermo, totals, temperature, pressure, stoich, thermo_targets
    )
    thermo = thermo + thermo_adjust
    return (stoich, stoich_adjust, thermo, thermo_adjust), x


# %%
print(stoich)
# (stoich, stoich_adjust, thermo), x = scanner_stoich(
#     (stoich, np.zeros_like(stoich), thermo), None
# )

start = datetime.now()
scanner_s = jax.lax.scan(
    scanner_stoich,
    (stoich, np.zeros_like(stoich), thermo),
    length=iter_stoich_per_thermo,
)[0]
stoich, stoich_adjust, thermo = scanner_s
print("scanner_stoich", datetime.now() - start)

# %%
print(thermo)
(stoich, stoich_adjust, thermo, thermo_adjust), x = scanner_thermo(
    (stoich, np.zeros_like(stoich), thermo, np.zeros_like(thermo)), None
)

# %%
start = datetime.now()
scanner_t = jax.lax.scan(
    scanner_thermo,
    (stoich, np.zeros_like(stoich), thermo, np.zeros_like(thermo)),
    length=iter_thermo,
)
print("scanner_thermo", datetime.now() - start)

start = datetime.now()
scanner_f = jax.lax.scan(
    scanner_full,
    (stoich, np.zeros_like(stoich), thermo, np.zeros_like(thermo)),
    length=iter_thermo,
)
print("scanner_full  ", datetime.now() - start)

# The scan functions do not compile much faster than the original solver function


# %%
# THIS IS NOT FASTER THAN THE ORIGINAL SOLVER FUNCTION WITH LOOPS - STICK WITH THAT!
def solve_scan(
    totals,
    temperature,
    pressure,
    stoich=None,
    thermo=None,
    iter_thermo=6,
    iter_stoich_per_thermo=3,
    verbose=False,
    warn_cutoff=1e-8,
):
    totals = totals.copy()
    totals.update({t: 0.0 for t in library.totals_all if t not in totals})
    stoich_targets = library.get_stoich_targets(totals)
    thermo_targets = np.array(
        [library.equilibria[eq](temperature) for eq in library.equilibria_all]
    )  # these are ln(k)
    if stoich is None:
        stoich = library.stoich_init(totals)
    if thermo is None:
        thermo = thermo_targets.copy()
    # Solve!

    def scanner_stoich(carry, x):
        stoich, stoich_adjust, thermo = carry
        stoich_adjust = get_stoich_adjust(stoich, totals, thermo, stoich_targets)
        stoich = stoich + stoich_adjust
        return (stoich, stoich_adjust, thermo), x

    def scanner_full(carry, x):
        stoich, stoich_adjust, thermo, thermo_adjust = carry
        scanner_s = jax.lax.scan(
            scanner_stoich,
            (stoich, np.zeros_like(stoich), thermo),
            length=iter_stoich_per_thermo,
        )[0]
        stoich, stoich_adjust, thermo = scanner_s
        thermo_adjust = get_thermo_adjust(
            thermo, totals, temperature, pressure, stoich, thermo_targets
        )
        thermo = thermo + thermo_adjust
        return (stoich, stoich_adjust, thermo, thermo_adjust), x

    scanner_f = jax.lax.scan(
        scanner_full,
        (stoich, np.zeros_like(stoich), thermo, np.zeros_like(thermo)),
        length=iter_thermo,
    )
    stoich, stoich_adjust, thermo, thermo_adjust = scanner_f[0]
    if np.any(np.abs(stoich_adjust) > warn_cutoff):
        warnings.warn(
            "Solver did not converge below `warn_cutoff` - "
            + "try increasing `iter_stoich_per_thermo`."
        )
    if np.any(np.abs(thermo_adjust) > warn_cutoff):
        warnings.warn(
            "Solver did not converge below `warn_cutoff` - "
            + "try increasing `iter_thermo`."
        )
    return SolveResult(stoich, thermo, stoich_adjust, thermo_adjust)


start = datetime.now()
tt = solve_scan(totals, temperature, pressure)
print("scanner_full f", datetime.now() - start)
