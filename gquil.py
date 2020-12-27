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
    k1, k2, kw, kB, kSO4, kHF = 10.0 ** -pkstars
    t_CO2, t_BOH3, t_SO4, t_HF = m_tots
    alk_CO2 = t_CO2 * (2 * k1 * k2 + k1 * H) / (H ** 2 + k1 * H + k1 * k2)
    alk_w = kw / H - H
    alk_BOH3 = t_BOH3 * kB / (kB + H)
    alk_HSO4 = -t_SO4 / (1 + kSO4 / H)
    alk_HF = -t_HF / (1 + kHF / H)
    return alk_CO2 + alk_BOH3 + alk_w + alk_HSO4 + alk_HF


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
    k1, k2, kw, kB, kSO4, kHF = 10.0 ** -pkstars
    t_CO2, t_BOH3, t_SO4, t_HF = m_tots
    CO2 = t_CO2 * H ** 2 / (H ** 2 + k1 * H + k1 * k2)
    HCO3 = t_CO2 * k1 * H / (H ** 2 + k1 * H + k1 * k2)
    CO3 = t_CO2 * k1 * k2 / (H ** 2 + k1 * H + k1 * k2)
    BOH4 = t_BOH3 * kB / (kB + H)
    BOH3 = t_BOH3 - BOH4  # TODO switch to get this from H instead
    OH = kw / H
    HSO4 = t_SO4 / (1 + kSO4 / H)
    SO4 = t_SO4 - HSO4  # TODO switch to get this from H instead
    HF = t_HF / (1 + kHF / H)
    F = t_HF - HF  # TODO switch to get this from H instead
    return (
        np.array([*m_cats_f, H]),
        np.array([*m_anis_f, OH, HCO3, CO3, BOH4, HSO4, SO4, F]),
        np.array([*m_neus_f, CO2, BOH3, HF]),
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
    H = m_cats[-1]
    ln_acf_H = ln_acfs[0][-1]
    OH, HCO3, CO3, BOH4, HSO4, SO4, F = m_anis[-7:]
    (
        ln_acf_OH,
        ln_acf_HCO3,
        ln_acf_CO3,
        ln_acf_BOH4,
        ln_acf_HSO4,
        ln_acf_SO4,
        ln_acf_F,
    ) = ln_acfs[1][-7:]
    CO2, BOH3, HF = m_neus
    ln_acf_CO2, ln_acf_BOH3, ln_acf_HF = ln_acfs[2]
    # Get equilibria
    lnk1, lnk2, lnkw, lnkBOH3, lnkHSO4, lnkHF = lnks
    gH2O = pz.equilibrate.Gibbs_H2O(ln_aw, H, ln_acf_H, OH, ln_acf_OH, lnkw)
    gH2CO3 = pz.equilibrate.Gibbs_H2CO3(
        ln_aw, H, ln_acf_H, HCO3, ln_acf_HCO3, CO2, ln_acf_CO2, lnk1
    )
    gHCO3 = pz.equilibrate.Gibbs_HCO3(
        H, ln_acf_H, HCO3, ln_acf_HCO3, CO3, ln_acf_CO3, lnk2
    )
    gBOH3 = pz.equilibrate.Gibbs_BOH3(
        ln_aw, ln_acf_BOH4, BOH4, ln_acf_BOH3, BOH3, ln_acf_H, H, lnkBOH3
    )
    gSO4 = pz.equilibrate.Gibbs_HSO4(
        H, ln_acf_H, SO4, ln_acf_SO4, HSO4, ln_acf_HSO4, lnkHSO4
    )
    gHF = pz.equilibrate.Gibbs_HF(H, ln_acf_H, F, ln_acf_F, HF, ln_acf_HF, lnkHF)
    g_total = np.array([gH2O, gH2CO3, gHCO3, gBOH3, gSO4, gHF])
    return g_total


#%% Test inputs
cations_f = ["Na", "Mg", "Ca", "K", "Sr"]
m_cats_f = np.array([0.5201, 0.06, 0.01, 0.01, 0.0001])
z_cats_f = np.array([+1, +2, +2, +1, +2])
a_cats_f = np.array([+1, +2, +2, +1, +2])
cations = [*cations_f, "H"]
z_cats = np.array([*z_cats_f, +1])
anions_f = ["Cl", "Br"]
m_anis_f = np.array([0.60695, 0.001])
z_anis_f = np.array([-1, -1])
a_anis_f = np.array([-1, -1])
anions = [*anions_f, "OH", "HCO3", "CO3", "BOH4", "HSO4", "SO4", "F"]
z_anis = np.array([*z_anis_f, -1, -1, -2, -1, -1, -2, -1])
neutrals_f = []
m_neus_f = np.array([])
a_neus_f = np.array([])
neutrals = [*neutrals_f, "CO2", "BOH3", "HF"]
totals = ["t_CO2", "t_BOH3", "t_SO4", "t_HF"]
m_tots = np.array([2000e-6, 400e-6, 0.03, 0.0001])
a_tots = np.array([0, 0, -2, -1])
reactions = ["k1", "k2", "kw", "kBOH3", "kSO4", "kHF"]
lnks = np.array(
    [
        pz.dissociation.H2CO3_MP98(),
        pz.dissociation.HCO3_MP98(),
        pz.dissociation.H2O_MF(),
        pz.dissociation.BOH3_M79(),
        pz.dissociation.HSO4_CRP94(),
        pz.dissociation.HF_MP98(),
    ]
)
pkstars = -np.log10(np.exp(lnks))

# Workflow/testing
alkalinity_ec = get_alkalinity_ec(
    (m_cats_f, m_anis_f, m_neus_f, m_tots), (a_cats_f, a_anis_f, a_neus_f, a_tots)
).item()
pH_solved = solve_pH(alkalinity_ec, pkstars, m_tots).item()  # initial condition
alkalinity_solved = alkalinity_pH(pH_solved, pkstars, m_tots)  # just to check
molalities = pH_to_molalities(pH_solved, pkstars, m_tots, m_cats_f, m_anis_f, m_neus_f)
params = pz.libraries.Seawater.get_parameters(cations, anions, neutrals, verbose=False)

print("here we go - warm-up")
go = time.time()
get_Gibbs_eq = get_Gibbs_equilibria(
    pkstars, lnks, alkalinity_ec, m_cats_f, m_anis_f, m_neus_f, z_cats, z_anis, params
)
print(time.time() - go)

# Solving
print("here we go - solver")
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
