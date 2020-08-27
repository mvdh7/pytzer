import jax
from jax import numpy as jnp


@jax.jit
def ionic_strength(molalities, charges):
    """Ionic strength."""
    return 0.5 * jnp.sum(molalities * charges ** 2)


@jax.jit
def ionic_z(molalities, charges):
    """Z function."""
    return jnp.sum(molalities * jnp.abs(charges))


@jax.jit
def Gibbs_DH(I, parameters):  # from CRP94 Eq. (AI1)
    """Calculate the Debye-Hueckel component of the excess Gibbs energy."""
    return (
        -4
        * parameters["Aphi"]
        * I
        * jnp.log(1 + parameters["b"] * jnp.sqrt(I))
        / parameters["b"]
    )


@jax.jit
def g(x):
    """g function following CRP94 Eq. (AI13)."""
    return 2 * (1 - (1 + x) * jnp.exp(-x)) / x ** 2


@jax.jit
def h(x):
    """h function following CRP94 Eq. (AI15)."""
    return (6 - (6 + x * (6 + 3 * x + x ** 2)) * jnp.exp(-x)) / x ** 4


@jax.jit
def B(sqrt_I, params_B):
    """B function following CRP94 Eq. (AI7)."""
    B = params_B["b0"]
    if "b1" in params_B:
        B = B + params_B["b1"] * g(params_B["alph1"] * sqrt_I)
    if "b2" in params_B:
        B = B + params_B["b2"] * g(params_B["alph2"] * sqrt_I)
    return B


@jax.jit
def CT(sqrt_I, params_CT):
    """CT function following CRP94 Eq. (AI10)."""
    CT = params_CT["C0"]
    if "C1" in params_CT:
        CT = CT + 4 * params_CT["C1"] * h(params_CT["omega"] * sqrt_I)
    return CT


@jax.jit
def xij(sqrt_I, z0, z1, parameters):
    """xij function for unsymmetrical mixing."""
    return 6 * z0 * z1 * parameters["Aphi"] * sqrt_I


@jax.jit
def etheta(I, z0, z1, parameters):
    """etheta function for unsymmetrical mixing."""
    sqrt_I = jnp.sqrt(I)
    x00 = xij(sqrt_I, z0, z0, parameters)
    x01 = xij(sqrt_I, z0, z1, parameters)
    x11 = xij(sqrt_I, z1, z1, parameters)
    return (
        z0
        * z1
        * (
            parameters.jfunc(x01)
            - 0.5 * (parameters.jfunc(x00) + parameters.jfunc(x11))
        )
        / (4 * I)
    )


# class Functions:
#     def __init__(self):
#         self.ca = []


# class Parameters:
#     def __init__(self):
#         self.functions = Functions()


# Next steps [2020-08-27]:
# - Set up the parameters input
# --- .ca is a numpy array where each element is a function of sqrt_I at T, P
# --- Other fields are arrays of parameters already evaluated at T, P
# --- The order of entries in these arrays needs to match their order in molalities
# - Both charges and parameters inputs should be automatically generated from a list of
#   ions along with T, P conditions (e.g. with a Parameters method)

@jax.jit
def Gex_nRT(molalities, charges, parameters):
    """Calculate the excess Gibbs energy of a solution."""
    # Note that oceanographers record ocean pressure as only due to the water,
    # so at the sea surface pressure = 0 dbar, but the atmospheric pressure
    # should also be taken into account for this model
    # Ionic strength etc.
    I = ionic_strength(molalities, charges)
    sqrt_I = jnp.sqrt(I)
    m_cats = molalities[charges > 0]
    m_anis = molalities[charges < 0]
    m_neus = molalities[charges == 0]
    # Split up charges
    z_cats = charges[charges > 0]
    z_anis = charges[charges < 0]
    # Begin with Debye-Hueckel component
    Gex_nRT = Gibbs_DH(I, parameters)
    # Loop through cations
    for CX, m_cat_x in enumerate(m_cats):
        # Add c-a interactions
        for A, m_ani in enumerate(m_anis):
            Gex_nRT = Gex_nRT + m_cat_x * m_ani * parameters.ca[CX, A](sqrt_I)
        # Add c-c' interactions
        for _CY, m_cat_y in enumerate(m_cats[CX + 1 :]):
            CY = _CY + CX + 1
            Gex_nRT = Gex_nRT + m_cat_x * m_cat_y * 2 * parameters.cc[CX, CY]
            # Unsymmetrical mixing terms
            if z_cats[CX] != z_cats[CY]:
                Gex_nRT = Gex_nRT + m_cat_x * m_cat_y * 2 * etheta(
                    I, z_cats[CX], z_cats[CY], parameters
                )
            # Add c-c'-a interactions
            for A, m_ani in enumerate(m_anis):
                Gex_nRT = (
                    Gex_nRT + m_cat_x * m_cat_y * m_ani * parameters.cca[CX, CY, A]
                )
    # Loop through anions
    for AX, m_ani_x in enumerate(m_anis):
        # Add a-a' interactions
        for _AY, m_ani_y in enumerate(m_anis[AX + 1 :]):
            AY = _AY + AX + 1
            Gex_nRT = Gex_nRT + m_ani_x * m_ani_y * 2 * parameters.aa[AX, AY]
            # Unsymmetrical mixing terms
            if z_anis[AX] != z_anis[AY]:
                Gex_nRT = Gex_nRT + m_ani_x * m_ani_y * 2 * etheta(
                    I, z_anis[AX], z_anis[AY], parameters
                )
            # Add c-a-a' interactions
            for C, m_cat in enumerate(m_cats):
                Gex_nRT = (
                    Gex_nRT + m_ani_x * m_ani_y * m_cat * parameters.caa[C, AX, AY]
                )
    # Add neutral interactions
    for NX, m_neu_x in enumerate(m_neus):
        # Add n-c interactions
        for C, m_cat in enumerate(m_cats):
            Gex_nRT = Gex_nRT + m_neu_x * m_cat * 2 * parameters.nc[NX, C]
            # Add n-c-a interactions
            for A, m_ani in enumerate(m_anis):
                Gex_nRT = Gex_nRT + m_neu_x * m_cat * m_ani * parameters.nca[NX, C, A]
        # Add n-a interactions
        for A, m_ani in enumerate(m_anis):
            Gex_nRT = Gex_nRT + m_neu_x * m_ani * 2 * parameters.na[NX, A]
        # n-n' excluding n-n
        for _NY, m_neu_y in enumerate(m_neus[NX + 1 :]):
            NY = _NY + NX + 1
            Gex_nRT = Gex_nRT + m_neu_x * m_neu_y * 2 * parameters.nn[NX, NY]
        # n-n
        Gex_nRT = Gex_nRT + m_neu_x ** 2 * parameters.nn[NX, NX]
        # n-n-n
        Gex_nRT = Gex_nRT + m_neu_x ** 3 * parameters.nnn[NX]
    return Gex_nRT


@jax.jit
def activity_coefficients(molalities, charges, parameters):
    return jax.grad(Gex_nRT)(molalities, charges, parameters)
