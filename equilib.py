import jax
from jax import numpy as np, lax
import pytzer as pz


@jax.jit
def get_alkalinity(pH, kstar_H2O):
    h = 10.0 ** -pH
    return kstar_H2O / h - h


@jax.jit
def grad_alkalinity(pH, kstar_H2O):
    return jax.grad(get_alkalinity)(pH, kstar_H2O)


@jax.jit
def get_delta_pH(pH, alkalinity, kstar_H2O):
    grad = grad_alkalinity(pH, kstar_H2O)
    return np.where(grad == 0, 0.0, (alkalinity - get_alkalinity(pH, kstar_H2O)) / grad)


@jax.jit
def solve_pH(alkalinity, kstar_H2O):

    pH = 3.0
    pH_tol = 1e-6

    def cond(pH):
        return np.abs(get_delta_pH(pH, alkalinity, kstar_H2O)) >= pH_tol

    def body(pH):
        pH = pH + get_delta_pH(pH, alkalinity, kstar_H2O)
        return pH

    pH = lax.while_loop(cond, body, pH)
    return pH


@jax.jit
def pH_to_solute_molalities(pH, kstar_H2O):
    h = 10.0 ** -pH
    return {
        "OH": kstar_H2O / h,
        "H": h,
    }


@jax.jit
def pH_to_molalities(pH, kstar_H2O):
    H = 10.0 ** -pH
    OH = kstar_H2O / H
    Cl = H - OH + 1.0
    Na = 1.0
    return np.array([H, Na]), np.array([OH, Cl]), np.array([])


pH = 3.1
pk_H2O = 14.0
kstar_H2O_i = 10.0 ** -pk_H2O
ln_k_H2O = np.log(kstar_H2O_i)
alkalinity = get_alkalinity(pH, kstar_H2O_i)  # should be from EC here

#%% Solve

pH_solved = solve_pH(alkalinity, kstar_H2O_i)
sm = pH_to_solute_molalities(pH_solved, kstar_H2O_i)
sm["Cl"] = -alkalinity + 1.0
sm["Na"] = 1.0

(m_cats, m_anis, m_neus, z_cats, z_anis), ss = pz.get_pytzer_args(sm)
params = pz.libraries.Seawater.get_parameters(**ss, verbose=False)


@jax.jit
def get_Gibbs_H2O(pkstar_H2O, ln_k_H2O, alkalinity, z_cats, z_anis, params):
    kstar_H2O = 10 ** -pkstar_H2O
    pH_solved = solve_pH(alkalinity, kstar_H2O)
    m_cats, m_anis, m_neus = pH_to_molalities(pH_solved, kstar_H2O)
    ln_aw = pz.model.log_activity_water(
        m_cats, m_anis, m_neus, z_cats, z_anis, **params
    )
    ln_acfs = pz.model.log_activity_coefficients(
        m_cats, m_anis, m_neus, z_cats, z_anis, **params
    )
    ln_acf_H = ln_acfs[0][0]
    ln_acf_OH = ln_acfs[1][0]
    m_H = m_cats[0]
    m_OH = m_anis[0]

    gH2O = pz.equilibrate.Gibbs_H2O(ln_aw, m_H, ln_acf_H, m_OH, ln_acf_OH, ln_k_H2O)

    return gH2O ** 2


#%%
@jax.jit
def grad_Gibbs_H2O(pkstar_H2O, ln_k_H2O, alkalinity, z_cats, z_anis, params):
    f = lambda pkstar_H2O: get_Gibbs_H2O(
        pkstar_H2O, ln_k_H2O, alkalinity, z_cats, z_anis, params
    )
    v = np.ones_like(pkstar_H2O)
    return jax.jvp(f, (pkstar_H2O,), (v,))


@jax.jit
def get_delta_pkstar_H2O(pkstar_H2O, ln_k_H2O, alkalinity, z_cats, z_anis, params):
    g, gr = grad_Gibbs_H2O(pkstar_H2O, ln_k_H2O, alkalinity, z_cats, z_anis, params)
    return np.where(gr == 0, 0.0, -g / gr)


#%%
gH2O = get_Gibbs_H2O(pk_H2O, ln_k_H2O, alkalinity, z_cats, z_anis, params)
grH2O = grad_Gibbs_H2O(pk_H2O, ln_k_H2O, alkalinity, z_cats, z_anis, params)
dH2O = get_delta_pkstar_H2O(pk_H2O, ln_k_H2O, alkalinity, z_cats, z_anis, params)

#%%
# @jax.jit
def solve_pkstar_H2O(ln_k_H2O, alkalinity, z_cats, z_anis, params):

    pkstar_H2O = 14.0
    tol = 1e-6

    def cond(pkstar_H2O):
        return (
            np.abs(
                get_delta_pkstar_H2O(
                    pkstar_H2O, ln_k_H2O, alkalinity, z_cats, z_anis, params
                )
            )
            >= tol
        )

    def body(pkstar_H2O):
        pkstar_H2O = pkstar_H2O + get_delta_pkstar_H2O(
            pkstar_H2O, ln_k_H2O, alkalinity, z_cats, z_anis, params
        )
        return pkstar_H2O

    while cond(pkstar_H2O):
        pkstar_H2O = body(pkstar_H2O)

    # pkstar_H2O = lax.while_loop(cond, body, pkstar_H2O)
    return pkstar_H2O


#%%
pkstar_solved = solve_pkstar_H2O(ln_k_H2O, alkalinity, z_cats, z_anis, params)
