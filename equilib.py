import jax
from jax import numpy as np, lax
import pytzer as pz


@jax.jit
def get_alkalinity(pH, kH2O):
    h = 10.0 ** -pH
    return kH2O / h - h


@jax.jit
def grad_alkalinity(pH, kH2O):
    return jax.grad(get_alkalinity)(pH, kH2O)


@jax.jit
def get_delta_pH(pH, alkalinity, kH2O):
    grad = grad_alkalinity(pH, kH2O)
    return np.where(grad == 0, 0.0, (alkalinity - get_alkalinity(pH, kH2O)) / grad,)


@jax.jit
def solve_pH(alkalinity, kH2O):

    pH = 3.0
    pH_tol = 1e-6

    def cond(pH):
        return np.abs(get_delta_pH(pH, alkalinity, kH2O)) >= pH_tol

    def body(pH):
        pH = pH + get_delta_pH(pH, alkalinity, kH2O)
        return pH

    pH = lax.while_loop(cond, body, pH)
    return pH


@jax.jit
def pH_to_solute_molalities(pH, kH2O):
    h = 10.0 ** -pH
    return {
        "OH": kH2O / h,
        "H": h,
    }


@jax.jit
def pH_to_molalities(pH, kH2O):
    H = 10.0 ** -pH
    OH = kH2O / H
    Cl = H - OH
    return np.array([H]), np.array([OH, Cl]), np.array([])


pH = 3.1
ln_kH2O = 14.0
kH2O = 10 ** -ln_kH2O
alkalinity = get_alkalinity(pH, kH2O)

#%% Solve

pH_solved = solve_pH(alkalinity, kH2O)
sm = pH_to_solute_molalities(pH_solved, kH2O)
sm["Cl"] = -alkalinity

(m_cats, m_anis, m_neus, z_cats, z_anis), ss = pz.get_pytzer_args(sm)
params = pz.libraries.Seawater.get_parameters(**ss, verbose=False)


@jax.jit
def get_Gibbs_H2O(kH2O, alkalinity, z_cats, z_anis, params):
    pH_solved = solve_pH(alkalinity, kH2O)
    m_cats, m_anis, m_neus = pH_to_molalities(pH_solved, kH2O)
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

    gH2O = pz.equilibrate.Gibbs_H2O(ln_aw, m_H, ln_acf_H, m_OH, ln_acf_OH, ln_kH2O)

    return gH2O


def grad_Gibbs_H2O(kH2O, alkalinity, z_cats, z_anis, params):
    return jax.grad(get_Gibbs_H2O)(kH2O, alkalinity, z_cats, z_anis, params)


gH2O = get_Gibbs_H2O(kH2O, alkalinity, z_cats, z_anis, params)
grH2O = grad_Gibbs_H2O(kH2O, alkalinity, z_cats, z_anis, params)
