import copy, jax, pytzer as pz, time
from jax import numpy as np, lax
from scipy import optimize


def get_alkalinity_ec(molality_args, alkalinity_args):
    return np.sum(
        np.array(
            [
                np.sum(m * a)
                for m, a in zip(molality_args, alkalinity_args)
                if len(m) > 0
            ]
        )
    )


def alkalinity_pH(pH, pkstars, m_tots):
    H = 10.0 ** -pH
    k1, k2, kw = 10.0 ** -pkstars
    (t_CO2,) = m_tots
    return t_CO2 * (2 * k1 * k2 + k1 * H) / (H ** 2 + k1 * H + k1 * k2) + kw / H - H


@jax.jit
def grad_alkalinity_pH(pH, pkstars, m_tots):
    return jax.jit(jax.grad(alkalinity_pH))(pH, pkstars, m_tots)


@jax.jit
def get_delta_pH(pH, alkalinity, pkstars, m_tots):
    grad = grad_alkalinity_pH(pH, pkstars, m_tots)
    return np.where(
        grad == 0, 0.0, (alkalinity - alkalinity_pH(pH, pkstars, m_tots)) / grad
    )


@jax.jit
def solve_pH(alkalinity, pkstars, m_tots):
    pH = 7.0
    tol = 1e-6

    def cond(pH):
        return np.abs(get_delta_pH(pH, alkalinity, pkstars, m_tots)) >= tol

    def body(pH):
        pH = pH + get_delta_pH(pH, alkalinity, pkstars, m_tots)
        return pH

    pH = lax.while_loop(cond, body, pH)
    return pH


@jax.jit
def pH_to_molalities(pH, pkstars, m_tots, m_cats_f, m_anis_f, m_neus_f):
    H = 10.0 ** -pH
    k1, k2, kw = 10.0 ** -pkstars
    (t_CO2,) = m_tots
    CO2 = t_CO2 * H ** 2 / (H ** 2 + k1 * H + k1 * k2)
    HCO3 = t_CO2 * k1 * H / (H ** 2 + k1 * H + k1 * k2)
    CO3 = t_CO2 * k1 * k2 / (H ** 2 + k1 * H + k1 * k2)
    OH = kw / H
    return (
        np.array([*m_cats_f, H]),
        np.array([*m_anis_f, HCO3, CO3, OH]),
        np.array([*m_neus_f, CO2]),
    )


@jax.jit
def get_Gibbs_equilibria(
    pkstars, lnks, alkalinity, m_cats_f, m_anis_f, m_neus_f, z_cats, z_anis, params
):
    # Solve for pH
    pH = solve_pH(alkalinity, pkstars, m_tots)
    m_cats, m_anis, m_neus = pH_to_molalities(
        pH, pkstars, m_tots, m_cats_f, m_anis_f, m_neus_f
    )
    ln_aw = pz.model.log_activity_water_map(
        m_cats, m_anis, m_neus, z_cats, z_anis, **params
    )
    ln_acfs = pz.model.log_activity_coefficients_map(
        m_cats, m_anis, m_neus, z_cats, z_anis, **params
    )
    # Extract outputs - BRITTLE!
    H = m_cats[-1]  # BRITTLE!
    ln_acf_H = ln_acfs[0][-1]  # BRITTLE!
    _, _, HCO3, CO3, OH = m_anis  # BRITTLE!
    _, _, ln_acf_HCO3, ln_acf_CO3, ln_acf_OH = ln_acfs[1]  # BRITTLE!
    (CO2,) = m_neus  # BRITTLE!
    (ln_acf_CO2,) = ln_acfs[2]  # BRITTLE!
    # Get equilibria
    lnk1, lnk2, lnkw = lnks
    gH2O = pz.equilibrate.Gibbs_H2O(ln_aw, H, ln_acf_H, OH, ln_acf_OH, lnkw)
    gH2CO3 = pz.equilibrate.Gibbs_H2CO3(
        ln_aw, H, ln_acf_H, HCO3, ln_acf_HCO3, CO2, ln_acf_CO2, lnk1
    )
    gHCO3 = pz.equilibrate.Gibbs_HCO3(
        H, ln_acf_H, HCO3, ln_acf_HCO3, CO3, ln_acf_CO3, lnk2
    )
    g_total = np.array([gH2O, gH2CO3, gHCO3])
    return g_total


# def get_Gibbs_equilibria(
#     pkstars, lnks, alkalinity, m_cats_f, m_anis_f, m_neus_f, z_cats, z_anis, params
# ):
#     return _get_Gibbs_equilibria(
#         pkstars, lnks, alkalinity, m_cats_f, m_anis_f, m_neus_f, z_cats, z_anis, params
#     ).item()


# @jax.jit
# def grad_Gibbs_equilibria(
#     pkstars, lnks, alkalinity, m_cats_f, m_anis_f, m_neus_f, z_cats, z_anis, params
# ):
#     f = lambda pkstars: get_Gibbs_equilibria(
#         pkstars, lnks, alkalinity, m_cats_f, m_anis_f, m_neus_f, z_cats, z_anis, params
#     )
#     v = np.ones_like(pkstars)
#     return jax.jvp(f, (pkstars,), (v,))


# @jax.jit
# def get_delta_pkstars(
#     pkstars, lnks, alkalinity, m_cats_f, m_anis_f, m_neus_f, z_cats, z_anis, params
# ):
#     g, gr = grad_Gibbs_equilibria(
#         pkstars, lnks, alkalinity, m_cats_f, m_anis_f, m_neus_f, z_cats, z_anis, params
#     )
#     return np.where(gr == 0, 0.0, -g / gr)


# def solve_pkstars(
#     lnks, alkalinity, m_cats_f, m_anis_f, m_neus_f, z_cats, z_anis, params
# ):

#     pkstars = copy.deepcopy(lnks)
#     tol = 1e-6

#     def cond(pkstars):
#         return np.any(
#             np.abs(
#                 get_delta_pkstars(
#                     pkstars,
#                     lnks,
#                     alkalinity,
#                     m_cats_f,
#                     m_anis_f,
#                     m_neus_f,
#                     z_cats,
#                     z_anis,
#                     params,
#                 )
#             )
#             >= tol
#         )

#     def body(pkstars):
#         pkstars = pkstars + get_delta_pkstars(
#             pkstars,
#             lnks,
#             alkalinity,
#             m_cats_f,
#             m_anis_f,
#             m_neus_f,
#             z_cats,
#             z_anis,
#             params,
#         )
#         return pkstars

#     delta = get_delta_pkstars(
#         pkstars,
#         lnks,
#         alkalinity,
#         m_cats_f,
#         m_anis_f,
#         m_neus_f,
#         z_cats,
#         z_anis,
#         params,
#     )
#     print(delta)

#     while cond(pkstars):
#         print(pkstars)
#         pkstars = body(pkstars)

#     # pkstars = lax.while_loop(cond, body, pkstars)
#     return pkstars


#%% Test inputs
cations_f = ["Na", "Mg", "Ca", "K", "Sr"]
m_cats_f = np.array([1.0 + 2250e-6, 0.5, 0.5, 0.1, 0.2])
z_cats_f = np.array([+1, +2, +2, +1, +2])
a_cats_f = np.array([+1, +2, +2, +1, +2])
cations = [*cations_f, "H"]
z_cats = np.array([*z_cats_f, +1])
anions_f = ["Cl", "Br"]
m_anis_f = np.array([3.0, 0.5])
z_anis_f = np.array([-1, -1])
a_anis_f = np.array([-1, -1])
anions = [*anions_f, "HCO3", "CO3", "OH"]
z_anis = np.array([*z_anis_f, -1, -2, -1])
neutrals_f = []
m_neus_f = np.array([])
a_neus_f = np.array([])
neutrals = [*neutrals_f, "CO2"]
totals = ["t_CO2"]
m_tots = np.array([2000e-6])
a_tots = np.array([0])
reactions = ["k1", "k2", "kw"]
pkstars = np.array([6.35, 10.33, 14.0])
lnks = np.log(10.0 ** -pkstars)

# Workflow/testing
alkalinity_ec = get_alkalinity_ec(
    (m_cats_f, m_anis_f, m_neus_f, m_tots), (a_cats_f, a_anis_f, a_neus_f, a_tots)
).item()
pH_solved = solve_pH(alkalinity_ec, pkstars, m_tots).item()  # initial condition
alkalinity_solved = alkalinity_pH(pH_solved, pkstars, m_tots)  # just to check
molalities = pH_to_molalities(pH_solved, pkstars, m_tots, m_cats_f, m_anis_f, m_neus_f)
params = pz.libraries.Seawater.get_parameters(cations, anions, neutrals, verbose=False)

print("here we go")
go = time.time()
get_Gibbs_eq = get_Gibbs_equilibria(
    pkstars, lnks, alkalinity_ec, m_cats_f, m_anis_f, m_neus_f, z_cats, z_anis, params
)
print(time.time() - go)

# Solving
print("here we go")
go = time.time()
args = (lnks, alkalinity_ec, m_cats_f, m_anis_f, m_neus_f, z_cats, z_anis, params)
x0 = copy.deepcopy(pkstars)
pkstars_optresult = optimize.root(get_Gibbs_equilibria, x0, args=args, method="hybr")
print(time.time() - go)
pkstars_solved = pkstars_optresult["x"]
sps_Gibbs = get_Gibbs_equilibria(pkstars_solved, *args)
pH_final = solve_pH(alkalinity_ec, pkstars_solved, m_tots).item()
molalities_final = pH_to_molalities(
    pH_final, pkstars_solved, m_tots, m_cats_f, m_anis_f, m_neus_f
)
