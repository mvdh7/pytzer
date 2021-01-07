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
