from collections import OrderedDict
import jax
from . import components, stoichiometric, thermodynamic


@jax.jit
def _solve(pfixed, totals, ks_constants):
    total_targets = stoichiometric.get_total_targets(totals, pfixed)
    pfixed = stoichiometric.solve(totals, ks_constants, pfixed=pfixed)
    solutes = components.get_solutes(totals, ks_constants, pfixed)
    return solutes, ks_constants


def solve(equilibria_to_solve, totals, ks_constants, params, pfixed=None):
    """Solve for thermodynamic equilibrium to return solute molalities and stoichiometric
    equilibrium constants.
    """
    if pfixed is None:
        pfixed = stoichiometric.create_pfixed(totals=totals)
    optresult_thermodynamic = thermodynamic.solve(
        equilibria_to_solve, totals, ks_constants, params, pfixed=pfixed
    )
    ks_constants = thermodynamic.update_ks_constants(
        ks_constants, optresult_thermodynamic
    )
    solutes, ks_constants = _solve(pfixed, totals, ks_constants)
    return solutes, ks_constants
