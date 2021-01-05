import jax
from jax import lax, numpy as np
from . import components


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


def get_total_targets(totals):
    return (
        get_explicit_alkalinity(totals),
        totals["F"],
        totals["CO2"],
        totals["PO4"],
    )


def get_solute_targets(solutes):
    return (
        get_alkalinity(solutes),
        get_total_F(solutes),
        get_total_CO2(solutes),
        get_total_PO4(solutes),
    )


def solver_func(
    p_molalities,
    totals,
    ks_constants,
    target_alkalinity=None,
    target_total_F=None,
    target_total_CO2=None,
    target_total_PO4=None,
):
    h, f, co3, po4 = 10 ** -p_molalities
    solutes = components.get_all(h, f, co3, po4, totals, ks_constants)
    targets = np.array([])
    if target_alkalinity is not None:
        targets = np.append(targets, target_alkalinity - get_alkalinity(solutes))
    if target_total_F is not None:
        targets = np.append(targets, target_total_F - get_total_F(solutes))
    if target_total_CO2 is not None:
        targets = np.append(targets, target_total_CO2 - get_total_CO2(solutes))
    if target_total_PO4 is not None:
        targets = np.append(targets, target_total_PO4 - get_total_PO4(solutes))
    return targets


solver_jac = jax.jit(jax.jacfwd(solver_func))


@jax.jit
def solve(
    p_molalities,
    totals,
    ks_constants,
    target_alkalinity=None,
    target_total_F=None,
    target_total_CO2=None,
    target_total_PO4=None,
):

    tol_alkalinity = 1e-9
    tol_total_F = 1e-9
    tol_total_CO2 = 1e-9
    tol_total_PO4 = 1e-9
    tols = np.array([tol_alkalinity, tol_total_F, tol_total_CO2, tol_total_PO4])

    def cond(p_molalities):
        target = solver_func(
            p_molalities,
            totals,
            ks_constants,
            target_alkalinity=target_alkalinity,
            target_total_F=target_total_F,
            target_total_CO2=target_total_CO2,
            target_total_PO4=target_total_PO4,
        )
        return np.any(np.abs(target) > tols)

    def body(p_molalities):
        target = -solver_func(
            p_molalities,
            totals,
            ks_constants,
            target_alkalinity=target_alkalinity,
            target_total_F=target_total_F,
            target_total_CO2=target_total_CO2,
            target_total_PO4=target_total_PO4,
        )
        jac = solver_jac(
            p_molalities,
            totals,
            ks_constants,
            target_alkalinity=target_alkalinity,
            target_total_F=target_total_F,
            target_total_CO2=target_total_CO2,
            target_total_PO4=target_total_PO4,
        )
        p_diff = np.linalg.solve(jac, target)
        p_diff = np.where(p_diff > 1, 1, p_diff)
        p_diff = np.where(p_diff < -1, -1, p_diff)
        return p_molalities + p_diff

    return lax.while_loop(cond, body, p_molalities)
