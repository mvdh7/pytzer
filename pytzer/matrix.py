import jax
from jax import numpy as np
from . import convert, model


def ionic_strength(molalities, charges):
    """Calculate the ionic strength.

    Parameters
    ----------
    molalities : array_like
        Molality of each solute in mol/kg.
    charges : array_like
        Charge on each solute.

    Returns
    -------
    float
        Ionic strength of the solution in mol/kg.
    """
    return molalities @ np.transpose(charges**2) / 2


def ionic_z(molalities, charges):
    """Calculate the Z function.

    Parameters
    ----------
    molalities : array_like
        Molality of each solute in mol/kg.
    charges : array_like
        Charge on each solute.

    Returns
    -------
    float
        Z function of the solution in mol/kg.
    """
    return molalities @ np.transpose(np.abs(charges))


def sum_B_CT(I, Z, beta0, beta1, beta2, alpha1, alpha2, c0, c1, omega):
    """Calculate the sum of the B and CT functions following CRP94 equations (AI7) and
    (AI10) respectively.

    Parameters
    ----------
    I : float
        Ionic strength of the solution in mol/kg.
    Z : float
        Z function of the solution in mol/kg.
    beta0 : array_like
        Beta-0 Pitzer model coefficients.
    beta1 : array_like
        Beta-1 Pitzer model coefficients.
    beta2 : array_like
        Beta-2 Pitzer model coefficients.
    alpha1 : array_like
        Alpha-1 Pitzer model coefficients.
    alpha2 : array_like
        Alpha-2 Pitzer model cofficients.
    c0 : array_like
        C-0 Pitzer model coefficients.
    c1 : array_like
        C-1 Pitzer model coefficients.
    omega : array_like
        Omega Pitzer model coefficients.

    Returns
    -------
    array_like
        Sum of the B and CT functions.
    """
    return (
        beta0
        + beta1 * model.g(alpha1 * np.sqrt(I))
        + beta2 * model.g(alpha2 * np.sqrt(I))
        + (c0 + 4 * c1 * model.h(omega * np.sqrt(I))) * Z / 2
    )


def xij(Aphi, I, charges):
    """xij function for unsymmetrical mixing."""
    return (np.transpose(charges) @ charges) * 6 * Aphi * np.sqrt(I)


def xi(Aphi, I, charges):
    """xi function for unsymmetrical mixing."""
    return charges**2 * 6 * Aphi * np.sqrt(I)


def xj(Aphi, I, zs):
    """xj function for unsymmetrical mixing."""
    return np.transpose(zs**2) * 6 * Aphi * np.sqrt(I)


def etheta(Aphi, I, charges):
    """E-theta function for unsymmetrical mixing."""
    x01 = xij(Aphi, I, charges)
    x00 = xi(Aphi, I, charges)
    x11 = xj(Aphi, I, charges)
    func_J = jax.vmap(jax.vmap(model.func_J))
    return (
        (np.transpose(charges) @ charges)
        * (func_J(x01) - (func_J(x00) + func_J(x11)) / 2)
        / (4 * I)
    )


@jax.jit
def Gibbs_nRT(
    solutes,
    n_cats_triu,
    n_anis_triu,
    Aphi=None,
    beta0_ca=None,
    beta1_ca=None,
    beta2_ca=None,
    alpha1_ca=None,
    alpha2_ca=None,
    c0_ca=None,
    c1_ca=None,
    omega_ca=None,
    theta_xx=None,
    psi_cca=None,
    psi_caa=None,
    lambda_nx=None,
    zeta_nca=None,
    mu_nnn=None,
    **parameters_extra
):
    """Excess Gibbs energy of a solution."""
    # Get the molalities of cations, anions and neutrals in separate arrays
    molalities = np.array([[m for m in solutes.values()]])
    m_cats = np.array([[m for s, m in solutes.items() if s in convert.all_cations]])
    m_anis = np.array([[m for s, m in solutes.items() if s in convert.all_anions]])
    m_neus = np.array([[m for s, m in solutes.items() if s in convert.all_neutrals]])
    # Get the charges of cations and anions in separate arrays
    s2c = convert.solute_to_charge
    charges = np.array([[s2c[s] for s in solutes]])
    z_cats = np.array([[s2c[s] for s in solutes if s in convert.all_cations]])
    z_anis = np.array([[s2c[s] for s in solutes if s in convert.all_anions]])
    I = ionic_strength(molalities, charges)
    # if I == 0:
    #     Gibbs_nRT = (
    #         molalities @ lambda_nx @ np.transpose(molalities)
    #         + m_neus @ zeta_nca @ np.transpose(m_cats_anis)
    #         + m_neus**3 @ mu_nnn
    #     )
    # else:
    #     # Get the products of cation and anion molalities
    m_cats_anis = np.array([(np.transpose(m_cats) @ m_anis).ravel()])
    Z = ionic_z(molalities, charges)
    m_cats_cats = np.array([(np.transpose(m_cats) @ m_cats)[n_cats_triu]])
    m_anis_anis = np.array([(np.transpose(m_anis) @ m_anis)[n_anis_triu]])
    Gibbs_nRT = model.Gibbs_DH(Aphi, I) + molalities @ (
        sum_B_CT(
            I,
            Z,
            beta0_ca,
            beta1_ca,
            beta2_ca,
            alpha1_ca,
            alpha2_ca,
            c0_ca,
            c1_ca,
            omega_ca,
        )
        + theta_xx
        + lambda_nx
    ) @ np.transpose(molalities)
    +m_cats @ etheta(Aphi, I, z_cats) @ np.transpose(m_cats)
    +m_anis @ etheta(Aphi, I, z_anis) @ np.transpose(m_anis)
    if psi_cca.size > 0:
        Gibbs_nRT = Gibbs_nRT + m_cats_cats @ psi_cca @ np.transpose(m_anis)
    if psi_caa.size > 0:
        Gibbs_nRT = Gibbs_nRT + m_anis_anis @ psi_caa @ np.transpose(m_cats)
    if zeta_nca.size > 0:
        Gibbs_nRT = Gibbs_nRT + m_neus @ zeta_nca @ np.transpose(m_cats_anis)
    if mu_nnn.size > 0:
        Gibbs_nRT = Gibbs_nRT + m_neus**3 @ mu_nnn
    return Gibbs_nRT[0][0]
