import jax
from jax import numpy as np


@jax.jit
def get_alkalinity(pH, kH2O):
    h = 10.0 ** -pH
    return kH2O / h + h


@jax.jit
def grad_alkalinity(pH, kH2O):
    return jax.grad(get_alkalinity)(pH, kH2O)


@jax.jit
def get_delta_pH(pH, alkalinity, kH2O):
    grad = grad_alkalinity(pH, kH2O)
    return np.where(grad == 0,
        0.0,
        (alkalinity - get_alkalinity(pH, kH2O)) / grad,
    )



def cond(d_pH):
    pH_tol = 1e-12
    return np.abs(d_pH[0]) >= pH_tol

alkalinity = 1e-6
kH2O = 1e-14


def body(d_pH):
    print(d_pH)
    d_pH[0] = get_delta_pH(d_pH[1], alkalinity, kH2O)
    d_pH[1] = d_pH[0] + d_pH[1]
    return d_pH


init = [7.0, 1e-14]


jlwhile = jax.lax.while_loop(cond, body, init)


# @jax.jit
def solve_pH(alkalinity, kH2O):
    pH = 6.0
    pH_tol = 1e-12
    delta_pH = 1.0 + pH_tol   
    while np.abs(delta_pH) >= pH_tol:
        delta_pH = get_delta_pH(pH, alkalinity, kH2O)  # the pH jump
        # # To keep the jump from being too big:
        # abs_deltapH = np.abs(deltapH)
        # np.sign_deltapH = np.sign(deltapH)
        # # Jump by 1 instead if `deltapH` > 5
        # deltapH = np.where(abs_deltapH > 5.0, np.sign_deltapH, deltapH)
        # # Jump by 0.5 instead if 1 < `deltapH` < 5
        # deltapH = np.where(
        #     (abs_deltapH > 0.5) & (abs_deltapH <= 5.0), 0.5 * np.sign_deltapH, deltapH,
        # )  # assumes that once we're within 1 of the correct pH, we will converge
        pH = pH + delta_pH
    return pH


pH = 5.1
kH2O = 1e-14
alk = get_alkalinity(pH, kH2O).item() * 1e6
alkg = grad_alkalinity(pH, kH2O).item() * 1e6
dpH = get_delta_pH(pH, alk*1e-6, kH2O).item()
pH_solved = solve_pH(alk*1e-6, kH2O).item()


@jax.jit
def pKstars_to_molalities(pKstars):
    if "H2O" in pKstars:
        k = 10.0 ** -pKstars["H2O"]
    return k


pKstars = {"H2O": 14.0}
molalities = pKstars_to_molalities(pKstars).item()


