import jax
from jax import lax, numpy as np
from . import components


def guess_pm_initial(totals, which_pms):
    m_initial = np.array([])
    for pm in which_pms:
        if pm == "H":
            m_initial = np.append(m_initial, 1e-8)
        elif pm == "F":
            assert totals["F"] > 0
            m_initial = np.append(m_initial, totals["F"] / 2)
        elif pm == "CO2":
            assert totals["CO2"] > 0
            m_initial = np.append(m_initial, totals["CO2"] / 10)
        elif pm == "PO4":
            assert totals["PO4"] > 0
            m_initial = np.append(m_initial, totals["PO4"] / 2)
    return -np.log10(m_initial)


def get_alkalinity(solutes):
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
    "CO2": lambda totals: totals["CO2"],
    "PO4": lambda totals: totals["PO4"],
}


def get_total_targets(totals, which_pms):
    return {pm: all_total_targets[pm](totals) for pm in which_pms}


all_solute_targets = {
    "H": lambda solutes: get_alkalinity(solutes),
    "F": lambda solutes: get_total_F(solutes),
    "CO2": lambda solutes: get_total_CO2(solutes),
    "PO4": lambda solutes: get_total_PO4(solutes),
}


def get_solute_targets(solutes, which_pms):
    return {pm: all_solute_targets[pm](solutes) for pm in which_pms}


def solver_func(
    p_molalities, totals, ks_constants, total_targets,
):
    molalities = 10 ** -p_molalities
    h, f, co3, po4 = molalities
    solutes = components.get_all(h, f, co3, po4, totals, ks_constants)
    solute_targets = get_solute_targets(solutes, total_targets.keys())
    targets = np.array(
        [
            total_target - solute_target
            for total_target, solute_target in zip(
                total_targets.values(), solute_targets.values()
            )
        ]
    )
    return targets


solver_jac = jax.jit(jax.jacfwd(solver_func))


@jax.jit
def solve(
    p_molalities, totals, ks_constants, total_targets,
):

    tol_alkalinity = 1e-9
    tol_total_F = 1e-9
    tol_total_CO2 = 1e-9
    tol_total_PO4 = 1e-9
    tols = np.array([tol_alkalinity, tol_total_F, tol_total_CO2, tol_total_PO4])

    def cond(p_molalities):
        target = solver_func(p_molalities, totals, ks_constants, total_targets,)
        return np.any(np.abs(target) > tols)

    def body(p_molalities):
        target = -solver_func(p_molalities, totals, ks_constants, total_targets,)
        jac = solver_jac(p_molalities, totals, ks_constants, total_targets,)
        p_diff = np.linalg.solve(jac, target)
        p_diff = np.where(p_diff > 1, 1, p_diff)
        p_diff = np.where(p_diff < -1, -1, p_diff)
        return p_molalities + p_diff

    return lax.while_loop(cond, body, p_molalities)
