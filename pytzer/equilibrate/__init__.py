from collections import OrderedDict
import jax
from . import components, stoichiometric, thermodynamic


def solve(equilibria_to_solve, pm_initial, totals, ks_constants, params, which_pms):
    optresult_thermodynamic = thermodynamic.solve(
        equilibria_to_solve, pm_initial, totals, ks_constants, params, which_pms,
    )
    ks_constants = thermodynamic.update_ks_constants(
        ks_constants, optresult_thermodynamic
    )
    total_targets = stoichiometric.get_total_targets(totals, which_pms)
    pm_final = stoichiometric.solve(pm_initial, totals, ks_constants, total_targets)
    m_final = 10.0 ** -pm_final
    solutes_final = components.get_all(*m_final, totals, ks_constants)
    return solutes_final, ks_constants


@jax.jit
def _solve_v2(pfixed_initial, totals, ks_constants):
    total_targets = stoichiometric.get_total_targets(totals, pfixed_initial)
    pfixed = stoichiometric.solve_v2(pfixed_initial, totals, ks_constants)
    fixed = OrderedDict((k, 10.0 ** -v) for k, v in pfixed.items())
    solutes = components.get_all_v2(fixed, totals, ks_constants)
    return solutes, ks_constants


def solve_v2(equilibria_to_solve, pfixed_initial, totals, ks_constants, params):
    optresult_thermodynamic = thermodynamic.solve_v2(
        equilibria_to_solve, pfixed_initial, totals, ks_constants, params
    )
    ks_constants = thermodynamic.update_ks_constants(
        ks_constants, optresult_thermodynamic
    )
    return _solve_v2(pfixed_initial, totals, ks_constants)
