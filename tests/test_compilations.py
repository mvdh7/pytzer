import itertools
import jax
from jax import numpy as np
from datetime import datetime as dt
from pytzer import parameters as p, debyehueckel, unsymmetrical
from pytzer.constants import b
from pytzer.convert import solute_to_charge
import pytzer as pz


def Gibbs_DH(Aphi, I):
    """The Debye-Hueckel component of the excess Gibbs energy following CRP94 eq. (AI1).

    Parameters
    ----------
    Aphi : float
        Debye-Hueckel limiting slope for the osmotic coefficient.
    I : float
        Ionic strength of the solution in mol/kg.

    Returns
    -------
    float
        Debye-Hueckel component of the excess Gibbs energy.
    """
    return -4 * Aphi * I * np.log(1 + b * np.sqrt(I)) / b


def g(x):
    """g function, following CRP94 Eq. (AI13)."""
    return 2 * (1 - (1 + x) * np.exp(-x)) / x**2


def h(x):
    """h function, following CRP94 Eq. (AI15)."""
    return (6 - (6 + x * (6 + 3 * x + x**2)) * np.exp(-x)) / x**4


def B(sqrt_I, b0, b1, b2, alph1, alph2):
    """B function, following CRP94 Eq. (AI7)."""
    return b0 + b1 * g(alph1 * sqrt_I) + b2 * g(alph2 * sqrt_I)


def CT(sqrt_I, C0, C1, omega):
    """CT function, following CRP94 Eq. (AI10)."""
    return C0 + 4 * C1 * h(omega * sqrt_I)


def xij(Aphi, I, z0, z1):
    """xij function for unsymmetrical mixing."""
    return 6 * z0 * z1 * Aphi * np.sqrt(I)


def etheta(Aphi, I, z0, z1):
    """etheta function for unsymmetrical mixing."""
    x00 = xij(Aphi, I, z0, z0)
    x01 = xij(Aphi, I, z0, z1)
    x11 = xij(Aphi, I, z1, z1)
    return (
        z0
        * z1
        * (library.func_J(x01) - 0.5 * (library.func_J(x00) + library.func_J(x11)))
        / (4 * I)
    )


library = pz.Library()
library.update_Aphi(debyehueckel.Aosm_M88)
library.update_func_J(unsymmetrical.P75_eq47)
# Tables S14-S18 (beta and C coefficients)
library.update_ca("Ca", "Br", p.bC_Ca_Br_SP78)
library.update_ca("Ca", "BOH4", p.bC_Ca_BOH4_SRM87)
library.update_ca("Ca", "Cl", p.bC_Ca_Cl_GM89)  # CWTD23 cite M88 but uses GM89
library.update_ca(
    "Ca", "HCO3", p.bC_Ca_HCO3_CWTD23
)  # CWTD23 cite HM93 but it's not - it's POS85 with a digit missing
library.update_ca("Ca", "HSO4", p.bC_Ca_HSO4_HMW84)
library.update_ca("Ca", "OH", p.bC_Ca_OH_HMW84)
library.update_ca("Ca", "SO4", p.bC_Ca_SO4_HEW82)
library.update_ca("CaF", "Cl", p.bC_CaF_Cl_PM16)
library.update_ca("H", "Br", p.bC_H_Br_MP98)
library.update_ca("H", "Cl", p.bC_H_Cl_CMR93)
library.update_ca("H", "HSO4", p.bC_H_HSO4_CRP94)
library.update_ca("H", "SO4", p.bC_H_SO4_CRP94)
library.update_ca("K", "Br", p.bC_K_Br_CWTD23)
library.update_ca("K", "BOH4", p.bC_K_BOH4_CWTD23)  # CWTD23 cite SRRJ87 but it's not
library.update_ca("K", "Cl", p.bC_K_Cl_GM89)
library.update_ca("K", "CO3", p.bC_K_CO3_CWTD23)  # CWTD23 cite SRG87 but it's not
library.update_ca("K", "F", p.bC_K_F_CWTD23)  # CWTD cite PM73 + SP78
library.update_ca("K", "HCO3", p.bC_K_HCO3_RGW84)
library.update_ca("K", "HSO4", p.bC_K_HSO4_WM13)
library.update_ca("K", "OH", p.bC_K_OH_MP98)
library.update_ca("K", "SO4", p.bC_K_SO4_GM89)
library.update_ca("Mg", "Br", p.bC_Mg_Br_SP78)
library.update_ca("Mg", "BOH4", p.bC_Mg_BOH4_SRM87)  # CWTD23 cite 88SR, numbers agree
library.update_ca("Mg", "Cl", p.bC_Mg_Cl_PP87i)
library.update_ca("Mg", "HCO3", p.bC_Mg_HCO3_CWTD23)  # CWTD23 cite POS85 but it's not
library.update_ca("Mg", "HSO4", p.bC_Mg_HSO4_HMW84)
library.update_ca("Mg", "SO4", p.bC_Mg_SO4_PP86ii)
library.update_ca("MgF", "Cl", p.bC_MgF_Cl_PM16)
library.update_ca("MgOH", "Cl", p.bC_MgOH_Cl_HMW84)
library.update_ca("Na", "Br", p.bC_Na_Br_CWTD23)  # CWTD23 cite 73PM
library.update_ca("Na", "BOH4", p.bC_Na_BOH4_CWTD23)  # TODO check vs MP98 function
library.update_ca("Na", "Cl", p.bC_Na_Cl_M88)
library.update_ca(
    "Na", "CO3", p.bC_Na_CO3_CWTD23b
)  # TODO check code vs table (see functions)
library.update_ca("Na", "F", p.bC_Na_F_CWTD23)
library.update_ca(
    "Na", "HCO3", p.bC_Na_HCO3_CWTD23b
)  # TODO check code vs table (see functions)
library.update_ca("Na", "HSO4", p.bC_Na_HSO4_CWTD23)
library.update_ca("Na", "OH", p.bC_Na_OH_PP87i)
library.update_ca("Na", "SO4", p.bC_Na_SO4_M88)
library.update_ca("Sr", "Br", p.bC_Sr_Br_SP78)
library.update_ca("Sr", "BOH4", p.bC_Ca_BOH4_SRM87)  # CWTD23 use Ca function
library.update_ca("Sr", "Cl", p.bC_Sr_Cl_CWTD23)
library.update_ca("Sr", "HCO3", p.bC_Ca_HCO3_CWTD23)  # CWTD23 use Ca function
library.update_ca("Sr", "HSO4", p.bC_Ca_HSO4_HMW84)  # CWTD23 use Ca function
library.update_ca("Sr", "OH", p.bC_Ca_OH_HMW84)  # CWTD23 use Ca function
library.update_ca("Sr", "SO4", p.bC_Ca_SO4_HEW82)  # CWTD23 use Ca function

# Table S19 (cc theta and psi coefficients)
library.update_cc("Ca", "H", p.theta_Ca_H_RGO81)
library.update_cca("Ca", "H", "Cl", p.psi_Ca_H_Cl_RGO81)
library.update_cc("Ca", "K", p.theta_Ca_K_GM89)
library.update_cca("Ca", "K", "Cl", p.psi_Ca_K_Cl_GM89)
library.update_cc("Ca", "Mg", p.theta_Ca_Mg_HMW84)  # CWTD23 cite HW80
library.update_cca("Ca", "Mg", "Cl", p.psi_Ca_Mg_Cl_HMW84)  # CWTD23 cite HW80
library.update_cca("Ca", "Mg", "SO4", p.psi_Ca_Mg_SO4_HMW84)  # CWTD23 cite HEW82
library.update_cc("Ca", "Na", p.theta_Ca_Na_M88)
library.update_cca("Ca", "Na", "Cl", p.psi_Ca_Na_Cl_M88)
library.update_cca("Ca", "Na", "SO4", p.psi_Ca_Na_SO4_HMW84)
library.update_cc("H", "K", p.theta_H_K_HWT22)  # CWTD23 cite CMR93
library.update_cca("H", "K", "Br", p.psi_H_K_Br_PK74)
library.update_cca("H", "K", "Cl", p.psi_H_K_Cl_HMW84)
library.update_cca("H", "K", "SO4", p.psi_H_K_SO4_HMW84)
library.update_cca("H", "K", "HSO4", p.psi_H_K_HSO4_HMW84)
library.update_cc("H", "Mg", p.theta_H_Mg_RGB80)
library.update_cca("H", "Mg", "Cl", p.psi_H_Mg_Cl_RGB80)
library.update_cca("H", "Mg", "HSO4", p.psi_H_Mg_HSO4_HMW84)  # not cited, but in code
library.update_cc("H", "Na", p.theta_H_Na_HWT22)
library.update_cca("H", "Na", "Br", p.psi_H_Na_Br_PK74)
library.update_cca("H", "Na", "Cl", p.psi_H_Na_Cl_PK74)
library.update_cc("H", "Sr", p.theta_H_Sr_RGRG86)
library.update_cca("H", "Sr", "Cl", p.psi_H_Sr_Cl_RGRG86)
library.update_cca("K", "Mg", "Cl", p.psi_K_Mg_Cl_PP87ii)
library.update_cca("K", "Mg", "SO4", p.psi_K_Mg_SO4_HMW84)  # CWTD23 cite HW80
library.update_cca("K", "Mg", "HSO4", p.psi_K_Mg_HSO4_HMW84)
library.update_cc("K", "Na", p.theta_K_Na_GM89)
library.update_cca("K", "Na", "Br", p.psi_K_Na_Br_PK74)
library.update_cca("K", "Na", "Cl", p.psi_K_Na_Cl_GM89)
library.update_cca("K", "Na", "SO4", p.psi_K_Na_SO4_GM89)
library.update_cc("K", "Sr", p.theta_Na_Sr_MP98)  # CWTD23 use Na-Sr function
library.update_cca("K", "Sr", "Cl", p.psi_Na_Sr_Cl_MP98)  # CWTD23 use Na-Sr function
library.update_cca("Mg", "MgOH", "Cl", p.psi_Mg_MgOH_Cl_HMW84)
library.update_cc("Mg", "Na", p.theta_Mg_Na_HMW84)  # CWTD23 cite P75
library.update_cca("Mg", "Na", "Cl", p.psi_Mg_Na_Cl_PP87ii)
library.update_cca("Mg", "Na", "SO4", p.psi_Mg_Na_SO4_HMW84)  # CWTD23 cite HW80
library.update_cc("Na", "Sr", p.theta_Na_Sr_MP98)
library.update_cca("Na", "Sr", "Cl", p.psi_Na_Sr_Cl_MP98)

# Table S20 (aa theta and psi coefficients)
library.update_aa("BOH4", "Cl", p.theta_BOH4_Cl_CWTD23)
library.update_caa("Ca", "BOH4", "Cl", p.psi_Ca_BOH4_Cl_MP98)  # CWTD23 cite 02P
library.update_caa("Mg", "BOH4", "Cl", p.psi_Mg_BOH4_Cl_MP98)  # CWTD23 cite 02P
library.update_caa("Na", "BOH4", "Cl", p.psi_Na_BOH4_Cl_MP98)  # CWTD23 cite 02P
library.update_aa("BOH4", "SO4", p.theta_BOH4_SO4_FW86)
library.update_aa("Br", "OH", p.theta_Br_OH_PK74)
library.update_caa("K", "Br", "OH", p.psi_K_Br_OH_PK74)
library.update_caa("Na", "Br", "OH", p.psi_Na_Br_OH_PK74)
library.update_aa("CO3", "Cl", p.theta_CO3_Cl_PP82)
library.update_caa("Na", "CO3", "Cl", p.psi_Na_CO3_Cl_TM82)
library.update_aa("Cl", "F", p.theta_Cl_F_MP98)  # CWTD23 cite 88CB
library.update_caa("Na", "Cl", "F", p.psi_Na_Cl_F_CWTD23)  # CWTD23 cite 88CB
library.update_aa("Cl", "HCO3", p.theta_Cl_HCO3_PP82)
library.update_caa("Mg", "Cl", "HCO3", p.psi_Mg_Cl_HCO3_HMW84)
library.update_caa("Na", "Cl", "HCO3", p.psi_Na_Cl_HCO3_PP82)
library.update_aa("Cl", "HSO4", p.theta_Cl_HSO4_HMW84)
library.update_caa("H", "Cl", "HSO4", p.psi_H_Cl_HSO4_HMW84)
library.update_caa("Na", "Cl", "HSO4", p.psi_Na_Cl_HSO4_HMW84)
library.update_aa("Cl", "OH", p.theta_Cl_OH_CWTD23)  # CWTD23 cite 02P
library.update_caa("Ca", "Cl", "OH", p.psi_Ca_Cl_OH_HMW84)
library.update_caa("K", "Cl", "OH", p.psi_K_Cl_OH_HMW84)
library.update_caa("Na", "Cl", "OH", p.psi_Na_Cl_OH_PK74)
library.update_aa("Cl", "SO4", p.theta_Cl_SO4_M88)
library.update_caa("Ca", "Cl", "SO4", p.psi_Ca_Cl_SO4_HMW84)  # CWTD23 cite 82HE/80HW
library.update_caa("K", "Cl", "SO4", p.psi_K_Cl_SO4_GM89)  # CWTD23 cite M88
library.update_caa("Mg", "Cl", "SO4", p.psi_Mg_Cl_SO4_HMW84)  # CWTD23 cite 82HE/80HW
library.update_caa("Na", "Cl", "SO4", p.psi_Na_Cl_SO4_M88)
library.update_aa("CO3", "SO4", p.theta_CO3_SO4_HMW84)
library.update_caa("K", "CO3", "SO4", p.psi_K_CO3_SO4_HMW84)
library.update_caa("Na", "CO3", "SO4", p.psi_Na_CO3_SO4_HMW84)
library.update_aa("HCO3", "SO4", p.theta_HCO3_SO4_HMW84)
library.update_caa("Mg", "HCO3", "SO4", p.psi_Mg_HCO3_SO4_HMW84)
library.update_caa("Na", "HCO3", "SO4", p.psi_Na_HCO3_SO4_HMW84)
library.update_caa("K", "HSO4", "SO4", p.psi_K_HSO4_SO4_HMW84)
library.update_aa("OH", "SO4", p.theta_OH_SO4_HMW84)
library.update_caa("K", "OH", "SO4", p.psi_K_OH_SO4_HMW84)
library.update_caa("Na", "OH", "SO4", p.psi_Na_OH_SO4_HMW84)

# Table S21 (lambda and zeta coefficients)
library.update_na("BOH3", "Cl", p.lambd_BOH3_Cl_FW86)
library.update_nc("BOH3", "K", p.lambd_BOH3_K_FW86)
library.update_nc("BOH3", "Na", p.lambd_BOH3_Na_FW86)
library.update_nca("BOH3", "Na", "SO4", p.zeta_BOH3_Na_SO4_FW86)
library.update_na("BOH3", "SO4", p.lambd_BOH3_SO4_FW86)
library.update_nc("CO2", "Ca", p.lambd_CO2_Ca_HM93)
library.update_nca("CO2", "Ca", "Cl", p.zeta_CO2_Ca_Cl_HM93)
library.update_na("CO2", "Cl", p.lambd_CO2_Cl_HM93)
library.update_nca("CO2", "H", "Cl", p.zeta_CO2_H_Cl_HM93)
library.update_nc("CO2", "K", p.lambd_CO2_K_HM93)
library.update_nca("CO2", "K", "Cl", p.zeta_CO2_K_Cl_HM93)
library.update_nca("CO2", "K", "SO4", p.zeta_CO2_K_SO4_HM93)
library.update_nc("CO2", "Mg", p.lambd_CO2_Mg_HM93)
library.update_nca("CO2", "Mg", "Cl", p.zeta_CO2_Mg_Cl_HM93)
library.update_nca("CO2", "Mg", "SO4", p.zeta_CO2_Mg_SO4_HM93)
library.update_nc("CO2", "Na", p.lambd_CO2_Na_HM93)
library.update_nca("CO2", "Na", "Cl", p.zeta_CO2_Na_Cl_HM93)
library.update_nca("CO2", "Na", "SO4", p.zeta_CO2_Na_SO4_HM93)
library.update_na("CO2", "SO4", p.lambd_CO2_SO4_HM93)
library.update_nc("HF", "Na", p.lambd_HF_Na_MP98)  # CWTD23 cite 88CB
library.update_nc("MgCO3", "Na", p.lambd_MgCO3_Na_CWTD23)  # CWTD23 cite 83MT


@jax.jit
def Gibbs_nRT(solutes, temperature, pressure):
    def add_ca(combo):
        c, a = combo
        return (
            m_cats[c]
            * m_anis[a]
            * (
                2 * B(sqrt_I, *ca[c][a][:3], *ca[c][a][5:7])
                + Z * CT(sqrt_I, *ca[c][a][3:5], ca[c][a][7])
            )
        )

    def add_cc(combo):
        c0, c1 = combo
        return (
            2
            * m_cats[c0]
            * m_cats[c1]
            * (
                cc[c0][c1]
                + etheta(Aphi, I, library.charges_cat[c0], library.charges_cat[c1])
            )
        )

    def add_cca(combo):
        c0, c1, a = combo
        return m_cats[c0] * m_cats[c1] * m_anis[a] * cca[c0][c1][a]

    def add_aa(combo):
        a0, a1 = combo
        return (
            2
            * m_anis[a0]
            * m_anis[a1]
            * (
                aa[a0][a1]
                + etheta(Aphi, I, library.charges_ani[a0], library.charges_ani[a1])
            )
        )

    def add_caa(combo):
        c, a0, a1 = combo
        return m_cats[c] * m_anis[a0] * m_anis[a1] * caa[c][a0][a1]

    def add_nnn(n):
        return m_neus[n] ** 3 * nnn[n]

    def add_nn(combo):
        n0, n1 = combo
        return 2 * m_neus[n0] * m_neus[n1] * nn[n0][n1]

    def add_nc(combo):
        n, c = combo
        return 2 * m_neus[n] * m_cats[c] * nc[n][c]

    def add_na(combo):
        n, a = combo
        return 2 * m_neus[n] * m_anis[a] * na[n][a]

    def add_nca(combo):
        n, c, a = combo
        return m_neus[n] * m_cats[c] * m_anis[a] * nca[n][c][a]

    m_cats = np.array([solutes[c] for c in library.cations])
    m_anis = np.array([solutes[a] for a in library.anions])
    m_neus = np.array([solutes[n] for n in library.neutrals])

    I = 0.5 * (
        np.sum(m_cats * library.charges_cat**2)
        + np.sum(m_anis * library.charges_ani**2)
    )
    Z = np.sum(m_cats * library.charges_cat) - np.sum(m_anis * library.charges_ani)
    sqrt_I = np.sqrt(I)
    tp = (temperature, pressure)
    Aphi = library.Aphi(*tp)[0]
    gibbs = Gibbs_DH(Aphi, I)

    if len(library.ca_combos) > 0:
        ca = library.get_ca_values(*tp)
        gibbs = gibbs + np.sum(jax.lax.map(add_ca, library.ca_combos))
    if len(library.cc_combos) > 0:
        cc = library.get_cc_values(*tp)
        gibbs = gibbs + np.sum(jax.lax.map(add_cc, library.cc_combos))
    if len(library.cca_combos) > 0:
        cca = library.get_cca_values(*tp)
        gibbs = gibbs + np.sum(jax.lax.map(add_cca, library.cca_combos))
    if len(library.aa_combos) > 0:
        aa = library.get_aa_values(*tp)
        gibbs = gibbs + np.sum(jax.lax.map(add_aa, library.aa_combos))
    if len(library.caa_combos) > 0:
        caa = library.get_caa_values(*tp)
        gibbs = gibbs + np.sum(jax.lax.map(add_caa, library.caa_combos))
    if len(library.nnn_combos) > 0:
        nnn = library.get_nnn_values(*tp)
        gibbs = gibbs + np.sum(jax.lax.map(add_nnn, library.nnn_combos))
    if len(library.nn_combos) > 0:
        nn = library.get_nn_values(*tp)
        gibbs = gibbs + np.sum(jax.lax.map(add_nn, library.nn_combos))
    if len(library.nc_combos) > 0:
        nc = library.get_nc_values(*tp)
        gibbs = gibbs + np.sum(jax.lax.map(add_nc, library.nc_combos))
    if len(library.na_combos) > 0:
        na = library.get_na_values(*tp)
        gibbs = gibbs + np.sum(jax.lax.map(add_na, library.na_combos))
    if len(library.nca_combos) > 0:
        nca = library.get_nca_values(*tp)
        gibbs = gibbs + np.sum(jax.lax.map(add_nca, library.nca_combos))

    return gibbs


mylist = np.array([1, 2, 3])


@jax.jit
def get_pos(i):
    return mylist[np.array(i)]


solutes = {
    "Na": 0.5,
    "K": 0.5,
    "Cl": 0.5,
    "Ca": 0.5,
    "Mg": 0.5,
    "OH": 0.5,
    "SO4": 0.5,
    "CO3": 0.5,
    "CaF": 0.5,
    "CO2": 0.5,
    "BOH3": 0.5,
    "BOH4": 0.5,
}
library.expand_solutes(solutes)
solutes = {k: 0.5 for k in solutes}

temperature, pressure = 298.15, 10.1325
gibbs = Gibbs_nRT(solutes, temperature, pressure)
gibbs_old = pz.model.Gibbs_nRT(solutes, temperature, pressure)
print(gibbs)
print(gibbs_old)

# %%
# solutes = {
#     "Na": 1.1,
#     "Cl": 1.2,
#     "OH": 1.3,
#     "SO4": 1.4,
#     "Br": 1.5,
#     "Mg": 1.6,
#     "Ca": 1.7,
#     "K": 1.8,
#     "Sr": 1.9,
# }
# charges = {
#     "Na": 1,
#     "Cl": -1,
#     "OH": -1,
#     "SO4": -2,
#     "Br": -1,
#     "Mg": 2,
#     "Ca": 2,
#     "K": 1,
#     "Sr": 2,
# }
# solute_to_code = {k: i for i, k in enumerate(solutes)}
# code_to_solute = {i: k for k, i in solute_to_code.items()}
# temperature = 25.0


# def loop_1(solutes, temperature):
#     cations = [s for s in solutes if charges[s] > 0]
#     anions = [s for s in solutes if charges[s] < 1]
#     gibbs = 0.0
#     for cation in cations:
#         for anion in anions:
#             try:
#                 gibbs = gibbs + solutes[cation] * solutes[anion] * library.ca[cation][
#                     anion
#                 ](temperature)
#             except KeyError:
#                 pass
#     return gibbs


# def loop_2(solutes, temperature):
#     gibbs = 0.0
#     for cation in library.ca:
#         for anion in library.ca[cation]:
#             gibbs = gibbs + solutes[cation] * solutes[anion] * library.ca[cation][
#                 anion
#             ](temperature)
#     return gibbs


# def map_1(solutes, temperature):
#     cations = [s for s in solutes if charges[s] > 0]
#     anions = [s for s in solutes if charges[s] < 1]
#     gibbs = 0.0
#     ca = itertools.product(cations, anions)
#     gibbs = gibbs + sum(
#         map(
#             lambda ca: solutes[ca[0]]
#             * solutes[ca[1]]
#             * library.ca[ca[0]][ca[1]](temperature),
#             ca,
#         )
#     )
#     return gibbs


# testscan = np.arange(10)


# def scan_1(solutes, temperature):

#     def scan_func(carry, x):
#         y = testscan[x]
#         return carry + y, y

#     x = np.arange(10)
#     gibbs = jax.lax.scan(scan_func, 0.0, x)
#     return gibbs


# def map_2(solutes, temperature):
#     solutes_coded = np.array(
#         list(
#             map(
#                 lambda k: solutes[k],
#                 solutes.keys(),
#             )
#         )
#     )
#     c_codes = [solute_to_code[s] for s in solutes if charges[s] > 0]
#     a_codes = [solute_to_code[s] for s in solutes if charges[s] < 1]
#     gibbs = 0.0
#     ca = np.array(list(itertools.product(c_codes, a_codes)))
#     gibbs = gibbs + np.sum(
#         jax.lax.map(
#             lambda ca: solutes_coded[ca[0]]
#             * solutes_coded[ca[1]]
#             * library.ca_coded[ca[0]][ca[1]](temperature),
#             ca,
#         )
#     )
#     return gibbs


# #
# gibbs_loop_1 = loop_1(solutes, temperature)
# gibbs_loop_2 = loop_2(solutes, temperature)
# gibbs_map_1 = map_1(solutes, temperature)
# gibbs_scan_1 = scan_1(solutes, temperature)
# # gibbs_map_2 = map_2(solutes, temperature)
# print("loop_1", gibbs_loop_1)
# print("loop_2", gibbs_loop_2)
# print(" map_1", gibbs_map_1)
# # print("map_2 ", gibbs_map_2)

# # %% Test compilation times
# start = dt.now()
# gibbs_loop_1 = jax.jit(loop_1)(solutes, temperature)
# print("Compile loop_1", dt.now() - start)
# start = dt.now()
# gibbs_loop_2 = jax.jit(loop_2)(solutes, temperature)
# print("Compile loop_2", dt.now() - start)
# start = dt.now()
# gibbs_map_1 = jax.jit(map_1)(solutes, temperature)
# print("Compile map_1", dt.now() - start)

# # Test grad times
# start = dt.now()
# grad_loop_1 = jax.grad(loop_1)(solutes, temperature)
# print("Grad loop_1", dt.now() - start)
# start = dt.now()
# grad_loop_2 = jax.grad(loop_2)(solutes, temperature)
# print("Grad loop_2", dt.now() - start)
# start = dt.now()
# grad_map_1 = jax.grad(map_1)(solutes, temperature)
# print("Grad map_1", dt.now() - start)

# # Test compile grad times
# start = dt.now()
# cgrad_loop_1 = jax.jit(jax.grad(loop_1))(solutes, temperature)
# print("Compile grad loop_1", dt.now() - start)
# start = dt.now()
# cgrad_loop_2 = jax.jit(jax.grad(loop_2))(solutes, temperature)
# print("Compile grad loop_2", dt.now() - start)
# start = dt.now()
# cgrad_map_1 = jax.jit(jax.grad(map_1))(solutes, temperature)
# print("Compile grad map_1", dt.now() - start)
