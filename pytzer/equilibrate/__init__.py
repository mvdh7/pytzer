from collections import OrderedDict
import jax
from . import components, stoichiometric, thermodynamic


@jax.jit
def _solve(pfixed_initial, totals, ks_constants):
    total_targets = stoichiometric.get_total_targets(totals, pfixed_initial)
    pfixed = stoichiometric.solve(pfixed_initial, totals, ks_constants)
    fixed = OrderedDict((k, 10.0 ** -v) for k, v in pfixed.items())
    solutes = components.get_solutes(fixed, totals, ks_constants)
    return solutes, ks_constants


def solve(equilibria_to_solve, pfixed, totals, ks_constants, params):
    """Solve for thermodynamic equilibrium."""
    if isinstance(pfixed, list):
        pfixed = stoichiometric.guess_pfixed(totals, pfixed)
    assert isinstance(pfixed, OrderedDict)
    optresult_thermodynamic = thermodynamic.solve(
        equilibria_to_solve, pfixed, totals, ks_constants, params
    )
    ks_constants = thermodynamic.update_ks_constants(
        ks_constants, optresult_thermodynamic
    )
    solutes, ks_constants = _solve(pfixed, totals, ks_constants)
    return solutes, ks_constants
