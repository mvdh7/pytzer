from collections import OrderedDict
import jax
from jax import lax, numpy as np
from . import components


def guess_pm_initial(totals, pfixed):
    m_initial = np.array([])
    for pm in pfixed:
        if pm == "H":
            m_initial = np.append(m_initial, 1e-8)
        elif pm == "F":
            assert totals["F"] > 0
            m_initial = np.append(m_initial, totals["F"] / 2)
        elif pm == "CO3":
            assert totals["CO2"] > 0
            m_initial = np.append(m_initial, totals["CO2"] / 10)
        elif pm == "PO4":
            assert totals["PO4"] > 0
            m_initial = np.append(m_initial, totals["PO4"] / 2)
    return -np.log10(m_initial)


def guess_pfixed(totals, fixed_solutes):
    pfixed = OrderedDict()
    for fs in fixed_solutes:
        if fs == "H":
            pfixed["H"] = 8.0
        elif fs == "F":
            assert totals["F"] > 0
            pfixed["F"] = -np.log10(totals["F"] / 2)
        elif fs == "CO3":
            assert totals["CO2"] > 0
            pfixed["CO3"] = -np.log10(totals["CO2"] / 10)
        elif fs == "PO4":
            assert totals["PO4"] > 0
            pfixed["PO4"] = -np.log10(totals["PO4"] / 2)
    return pfixed


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
    "CO3": lambda totals: totals["CO2"],
    "PO4": lambda totals: totals["PO4"],
}


def get_total_targets(totals, pfixed):
    return OrderedDict((pf, all_total_targets[pf](totals)) for pf in pfixed)


all_solute_targets = {
    "H": lambda solutes: get_alkalinity(solutes),
    "F": lambda solutes: get_total_F(solutes),
    "CO3": lambda solutes: get_total_CO2(solutes),
    "PO4": lambda solutes: get_total_PO4(solutes),
}


def get_solute_targets(solutes, pfixed):
    return OrderedDict((pf, all_solute_targets[pf](solutes)) for pf in pfixed)


def solver_func(pfixed_values, pfixed, totals, ks_constants):
    fixed = OrderedDict(
        (k, 10.0 ** -pfixed_values[i]) for i, k in enumerate(pfixed.keys())
    )
    total_targets = get_total_targets(totals, pfixed)
    solutes = components.get_solutes(fixed, totals, ks_constants)
    solute_targets = get_solute_targets(solutes, pfixed)
    targets = np.array([total_targets[pf] - solute_targets[pf] for pf in pfixed.keys()])
    return targets


solver_jac = jax.jit(jax.jacfwd(solver_func))


@jax.jit
def solve(pfixed, totals, ks_constants):
    def cond(pfixed_values):
        target = solver_func(pfixed_values, pfixed, totals, ks_constants)
        return np.any(np.abs(target) > 1e-9)

    def body(pfixed_values):
        target = -solver_func(pfixed_values, pfixed, totals, ks_constants)
        jac = solver_jac(pfixed_values, pfixed, totals, ks_constants)
        p_diff = np.linalg.solve(jac, target)
        p_diff = np.where(p_diff > 1, 1, p_diff)
        p_diff = np.where(p_diff < -1, -1, p_diff)
        return pfixed_values + p_diff

    pfixed_values = np.array([v for v in pfixed.values()])
    pfixed_values = lax.while_loop(cond, body, pfixed_values)
    pfixed_final = OrderedDict(
        (k, pfixed_values[i]) for i, k in enumerate(pfixed.keys())
    )
    return pfixed_final
