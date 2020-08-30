import jax
from jax import numpy as jnp
from . import properties


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


class Functions:
    def __init__(self):
        self.ca = {}
        self.cc = {}
        self.aa = {}
        self.cca = {}
        self.caa = {}
        self.nc = {}
        self.na = {}
        self.nn = {}
        self.nca = {}
        self.nnn = {}

    def add_ca(self, cation, anion, func=BC_NONE):
        if cation not in self.ca:
            self.ca[cation] = {}
        self.ca[cation][anion] = func

    def add_cc(self, cation_x, cation_y, func=THETA_NONE):
        if cation_x not in self.cc:
            self.cc[cation_x] = {}
        if cation_y not in self.cc:
            self.cc[cation_y] = {}
        self.cc[cation_x][cation_y] = self.cc[cation_y][cation_x] = func

    def add_aa(self, anion_x, anion_y, func=THETA_NONE):
        if anion_x not in self.cc:
            self.cc[anion_x] = {}
        if anion_y not in self.cc:
            self.cc[anion_y] = {}
        self.cc[anion_x][anion_y] = self.cc[anion_y][anion_x] = func

    def add_cca(self, cation_x, cation_y, anion, func=PSI_NONE):
        if cation_x not in self.cca:
            self.cca[cation_x] = {}
        if cation_y not in self.cca[cation_x]:
            self.cca[cation_x][cation_y] = {}
        if cation_y not in self.cca:
            self.cca[cation_y] = {}
        if cation_x not in self.cca[cation_y]:
            self.cca[cation_y][cation_x] = {}
        self.cca[cation_x][cation_y][anion] = self.cca[cation_y][cation_x][anion] = func

    def add_caa(self, cation, anion_x, anion_y, func=PSI_NONE):
        if cation not in self.caa:
            self.caa[cation] = {}
        if anion_x not in self.caa[cation]:
            self.caa[cation][anion_x] = {}
        if anion_y not in self.caa[cation]:
            self.caa[cation][anion_y] = {}
        self.caa[cation][anion_x][anion_y] = self.cca[cation][anion_y][anion_x] = func


class Parameters:
    def __init__(self, temperature=None, pressure=None):
        self.temperature = temperature
        self.pressure = pressure
        self.ca = {}
        self.cc = {}
        self.aa = {}
        self.cca = {}
        self.caa = {}
        self.nc = {}
        self.na = {}
        self.nn = {}
        self.nca = {}
        self.nnn = {}


def get_Ifunc(func_TP):
    ftp = func_TP

    def Ifunc(sqrt_I, Z):
        return 2 * (
            ftp["b0"]
            + ftp["b1"] * g(ftp["alph1"] * sqrt_I)
            + ftp["b2"] * g(ftp["alph2"] * sqrt_I)
        ) + Z * (ftp["C0"] + 4 * ftp["C1"] * h(ftp["omega"] * sqrt_I))

    return Ifunc


def ions2charges(ions):
    """Find the charge on each of a list of ions."""
    return jnp.array([properties._ion2charge[ion] for ion in ions])


class ParameterLibrary:
    def __init__(self):
        self.functions = Functions()
        self.parameters = Parameters()

    def set_parameters(self, ions, temperature=298.15, pressure=10.1325):
        self.parameters.temperature, self.parameters.pressure = T, P = (
            temperature,
            pressure,
        )
        charges = ions2charges(ions)
        cations = ions[charges > 0]
        anions = ions[charges < 0]
        neutrals = ions[charges == 0]
        for CX, cation_x in enumerate(cations):
            for A, anion in enumerate(anions):
                self.parameters.ca[CX][A] = jax.jit(
                    get_Ifunc(self.functions.ca[cation_x][anion](T, P))
                )
            for _CY, cation_y in enumerate(cations[CX + 1 :]):
                CY = _CY + CX + 1
                self.parameters.cc[CX][CY] = self.functions.cc[cation_x][cation_y](T, P)
                for A, anion in enumerate(anions):
                    self.parameters.cca[CX][CY][A] = self.functions.cca[cation_x][
                        cation_y
                    ][anion](T, P)
        for AX, anion_x in enumerate(anions):
            for _AY, anion_y in enumerate(anions[AX + 1 :]):
                AY = _AY + AX + 1
                self.parameters.aa[AX][AY] = self.functions.aa[anion_x][anion_y](T, P)
                for C, cation in enumerate(cations):
                    self.parameters.caa[C][AX][AY] = self.functions.caa[cation][
                        anion_x
                    ][anion_y](T, P)
        for NX, neutral_x in enumerate(neutrals):
            for C, cation in enumerate(cations):
                self.parameters.nc[NX][C] = self.functions.nc[neutral_x][cation](T, P)
                for A, anion in enumerate(anions):
                    self.parameters.nca[NX][C][A] = self.functions.nca[neutral_x][
                        cation
                    ][anion](T, P)
            for A, anion in enumerate(anions):
                self.parameters.na[NX][A] = self.functions.na[neutral_x][anion](T, P)
            for _NY, neutral_y in enumerate(neutrals[NX + 1 :]):
                NY = _NY + NX + 1
                self.parameters.nn[NX][NY] = self.functions.nn[neutral_x][neutral_y](
                    T, P
                )
            self.parameters.nn[NX][NX] = self.functions.nn[neutral_x][neutral_x](T, P)
            self.parameters.nnn[NX] = self.functions.nnn[neutral_x](T, P)


@jax.jit
def Gex_nRT(molalities, charges, parameters):
    """Calculate the excess Gibbs energy of a solution."""
    # Note that oceanographers record ocean pressure as only due to the water,
    # so at the sea surface pressure = 0 dbar, but the atmospheric pressure
    # should also be taken into account for this model
    # Ionic strength etc.
    I = ionic_strength(molalities, charges)
    Z = ionic_z(molalities, charges)
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
            Gex_nRT = Gex_nRT + m_cat_x * m_ani * (
                2 * B(sqrt_I, parameters.ca[CX][A])
                + Z * CT(sqrt_I, parameters.ca[CX][A])
            )
        # Add c-c' interactions
        for _CY, m_cat_y in enumerate(m_cats[CX + 1 :]):
            CY = _CY + CX + 1
            Gex_nRT = Gex_nRT + m_cat_x * m_cat_y * 2 * parameters.cc[CX][CY]
            # Unsymmetrical mixing terms
            if z_cats[CX] != z_cats[CY]:
                Gex_nRT = Gex_nRT + m_cat_x * m_cat_y * 2 * etheta(
                    I, z_cats[CX], z_cats[CY], parameters
                )
            # Add c-c'-a interactions
            for A, m_ani in enumerate(m_anis):
                Gex_nRT = (
                    Gex_nRT + m_cat_x * m_cat_y * m_ani * parameters.cca[CX][CY][A]
                )
    # Loop through anions
    for AX, m_ani_x in enumerate(m_anis):
        # Add a-a' interactions
        for _AY, m_ani_y in enumerate(m_anis[AX + 1 :]):
            AY = _AY + AX + 1
            Gex_nRT = Gex_nRT + m_ani_x * m_ani_y * 2 * parameters.aa[AX][AY]
            # Unsymmetrical mixing terms
            if z_anis[AX] != z_anis[AY]:
                Gex_nRT = Gex_nRT + m_ani_x * m_ani_y * 2 * etheta(
                    I, z_anis[AX], z_anis[AY], parameters
                )
            # Add c-a-a' interactions
            for C, m_cat in enumerate(m_cats):
                Gex_nRT = (
                    Gex_nRT + m_ani_x * m_ani_y * m_cat * parameters.caa[C][AX][AY]
                )
    # Add neutral interactions
    for NX, m_neu_x in enumerate(m_neus):
        # Add n-c interactions
        for C, m_cat in enumerate(m_cats):
            Gex_nRT = Gex_nRT + m_neu_x * m_cat * 2 * parameters.nc[NX][C]
            # Add n-c-a interactions
            for A, m_ani in enumerate(m_anis):
                Gex_nRT = Gex_nRT + m_neu_x * m_cat * m_ani * parameters.nca[NX][C][A]
        # Add n-a interactions
        for A, m_ani in enumerate(m_anis):
            Gex_nRT = Gex_nRT + m_neu_x * m_ani * 2 * parameters.na[NX][A]
        # n-n' excluding n-n
        for _NY, m_neu_y in enumerate(m_neus[NX + 1 :]):
            NY = _NY + NX + 1
            Gex_nRT = Gex_nRT + m_neu_x * m_neu_y * 2 * parameters.nn[NX][NY]
        # n-n
        Gex_nRT = Gex_nRT + m_neu_x ** 2 * parameters.nn[NX][NX]
        # n-n-n
        Gex_nRT = Gex_nRT + m_neu_x ** 3 * parameters.nnn[NX]
    return Gex_nRT


@jax.jit
def activity_coefficients(molalities, charges, parameters):
    return jax.grad(Gex_nRT)(molalities, charges, parameters)
