# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2023  Matthew P. Humphreys  (GNU GPLv3)
"""HM93: He & Morse (1993) Geochim. Cosmochim. Acta 57(15), 3533-3554.
https://doi.org/10.1016/0016-7037(93)90137-L
"""

from jax import numpy as np
from ..convert import solute_to_charge as i2c
from .spencer1990 import psi_Mg_Cl_SO4_SMW90
from .pabalanPitzer1987 import psi_Mg_Cl_SO4_PP87ii


def HM93_eq(T, A, B, C, D, E):
    """HM93 parameter equation from p. 3548."""
    return A + B * T + C * T**2 + D / T + E * np.log(T)


def bC_Na_HCO3_HM93(T, P):
    """c-a: sodium bicarbonate [HM93]."""
    b0 = HM93_eq(T, -37.2624193, -0.01445932, 0, 682.885977, 6.8995857)
    b1 = HM93_eq(T, -61.4635193, -0.02446734, 0, 1129.389146, 11.4108589)
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 363.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_HCO3_HM93(T, P):
    """c-a: potassium bicarbonate [HM93]."""
    b0 = HM93_eq(T, -0.3088232, 0.001, 0, -0.00069869, -4.701488e-6)
    b1 = HM93_eq(T, -0.2802, 0.00109999, 0, 0.000936932, 6.15660566e-6)
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 363.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mg_HCO3_HM93(T, P):
    """c-a: magnesium bicarbonate [HM93]."""
    b0 = HM93_eq(T, 13697.10, 8.250840, -0.00434, -273406.1716, -2607.115202)
    b1 = HM93_eq(T, -157839.8351, -92.7779354, 0.0477642, 3203209.6948, 29927.151503)
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 363.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ca_HCO3_HM93(T, P):
    """c-a: calcium bicarbonate [HM93]."""
    b0 = HM93_eq(T, 29576.53405, 18.447305, -0.009989, -576520.5185, -5661.1237)
    b1 = HM93_eq(T, -1028.8510522, -0.3725876718, 8.9691e-5, 26492.240303, 183.13155672)
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 363.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_CO3_HM93(T, P):
    """c-a: sodium carbonate [HM93]."""
    b0 = HM93_eq(T, -60.5387702, -0.023301655, 0, 1108.3760518, 11.19855531)
    b1 = HM93_eq(T, -237.5156616, -0.09989121, 0, 4412.511973, 44.5820703)
    b2 = 0
    Cphi = 0.0052
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["CO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 363.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_CO3_HM93(T, P):
    """c-a: potassium carbonate [HM93]."""
    b0 = HM93_eq(T, -0.1991649, 0.00110, 0, 1.8063362e-5, 0)
    b1 = HM93_eq(T, 0.1330579, 0.00436, 0, 0.0011899, 0)
    b2 = 0
    Cphi = 0.0005
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["CO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 363.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mg_CO3_HM93(T, P):
    """c-a: magnesium carbonate [HM93]."""
    b0 = 0
    b1 = 0
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = -9
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 363.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ca_CO3_HM93(T, P):
    """c-a: calcium carbonate [HM93]."""
    b0 = 0
    b1 = 0
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = -9
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 363.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def psi_Mg_Cl_SO4_HM93(T, P):
    """c-a-a': magnesium chloride sulfate [HM93]."""
    psi = np.where(
        T < 298.15, psi_Mg_Cl_SO4_SMW90(T, P)[0], psi_Mg_Cl_SO4_PP87ii(T, P)[0]
    )
    valid = (T > 273.15 - 54) & (T < 523.25)
    return psi, valid


def lambd_CO2_H_HM93(T, P):
    """n-c: carbon-dioxide hydrogen [HM93]."""
    lambd = 0
    valid = (T >= 273.15) & (T <= 363.15)
    return lambd, valid


def lambd_CO2_Na_HM93(T, P):
    """n-c: carbon-dioxide sodium [HM93]."""
    lambd = HM93_eq(
        T,
        -5496.38465,
        -3.326566,
        0.0017532,
        109399.341,
        1047.021567,
    )
    valid = (T >= 273.15) & (T <= 363.15)
    return lambd, valid


def lambd_CO2_K_HM93(T, P):
    """n-c: carbon-dioxide potassium [HM93]."""
    lambd = HM93_eq(
        T,
        2856.528099,
        1.7670079,
        -0.0009487,
        -55954.1929,
        -546.074467,
    )
    valid = (T >= 273.15) & (T <= 363.15)
    return lambd, valid


def lambd_CO2_Ca_HM93(T, P):
    """n-c: carbon-dioxide calcium [HM93]."""
    lambd = HM93_eq(
        T,
        -12774.6472,
        -8.101555,
        0.00442472,
        245541.5435,
        2452.509720,
    )
    valid = (T >= 273.15) & (T <= 363.15)
    return lambd, valid


def lambd_CO2_Mg_HM93(T, P):
    """n-c: carbon-dioxide magnesium [HM93]."""
    lambd = HM93_eq(
        T,
        -479.362533,
        -0.541843,
        0.00038812,
        3589.474052,
        104.3452732,
    )
    valid = (T >= 273.15) & (T <= 363.15)
    return lambd, valid


def lambd_CO2_Cl_HM93(T, P):
    """n-a: carbon-dioxide chloride [HM93]."""
    lambd = HM93_eq(T, 1659.944942, 0.9964326, -0.00052122, -33159.6177, -315.827883)
    valid = (T >= 273.15) & (T <= 363.15)
    return lambd, valid


def lambd_CO2_SO4_HM93(T, P):
    """n-a: carbon-dioxide sulfate [HM93]."""
    lambd = HM93_eq(
        T,
        2274.656591,
        1.8270948,
        -0.00114272,
        -33927.7625,
        -457.015738,
    )
    valid = (T >= 273.15) & (T <= 363.15)
    return lambd, valid


def zeta_CO2_H_Cl_HM93(T, P):
    """n-c-a: carbon-dioxide hydrogen chloride [HM93]."""
    zeta = HM93_eq(T, -804.121738, -0.470474, 0.000240526, 16334.38917, 152.3838752)
    valid = (T >= 273.15) & (T <= 363.15)
    return zeta, valid


def zeta_CO2_Na_Cl_HM93(T, P):
    """n-c-a: carbon-dioxide sodium chloride [HM93]."""
    zeta = HM93_eq(T, -379.459185, -0.258005, 0.000147823, 6879.030871, 73.74511574)
    valid = (T >= 273.15) & (T <= 363.15)
    return zeta, valid


def zeta_CO2_K_Cl_HM93(T, P):
    """n-c-a: carbon-dioxide potassium chloride [HM93]."""
    zeta = HM93_eq(T, -379.686097, -0.257891, 0.000147333, 6853.264129, 73.79977116)
    valid = (T >= 273.15) & (T <= 363.15)
    return zeta, valid


def zeta_CO2_Ca_Cl_HM93(T, P):
    """n-c-a: carbon-dioxide calcium chloride [HM93]."""
    zeta = HM93_eq(T, -166.065290, -0.018002, -2.47349e-5, 5256.844332, 27.377452415)
    valid = (T >= 273.15) & (T <= 363.15)
    return zeta, valid


def zeta_CO2_Mg_Cl_HM93(T, P):
    """n-c-a: carbon-dioxide magnesium chloride [HM93]."""
    zeta = HM93_eq(T, -1342.60256, -0.772286, 0.000391603, 27726.80974, 253.62319406)
    valid = (T >= 273.15) & (T <= 363.15)
    return zeta, valid


def zeta_CO2_H_SO4_HM93(T, P):
    """n-c-a: carbon-dioxide hydrogen sulfate [HM93]."""
    zeta = 0
    valid = (T >= 273.15) & (T <= 363.15)
    return zeta, valid


def zeta_CO2_Na_SO4_HM93(T, P):
    """n-c-a: carbon-dioxide sodium sulfate [HM93]."""
    zeta = HM93_eq(T, 67030.02482, 37.930519, -0.01894730, -1399082.37, -12630.27457)
    valid = (T >= 273.15) & (T <= 363.15)
    return zeta, valid


def zeta_CO2_K_SO4_HM93(T, P):
    """n-c-a: carbon-dioxide potassium sulfate [HM93]."""
    zeta = HM93_eq(T, -2907.03326, -2.860763, 0.001951086, 30756.86749, 611.37560512)
    valid = (T >= 273.15) & (T <= 363.15)
    return zeta, valid


def zeta_CO2_Mg_SO4_HM93(T, P):
    """n-c-a: carbon-dioxide magnesium sulfate [HM93]."""
    zeta = HM93_eq(T, -7374.24392, -4.608331, 0.002489207, 143162.6076, 1412.302898)
    valid = (T >= 273.15) & (T <= 363.15)
    return zeta, valid
