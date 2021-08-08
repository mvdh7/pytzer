# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
"""Evaluate Pitzer model interaction parameters."""
from jax import numpy as np
from .constants import Tzero
from .convert import solute_to_charge as i2c

# Tolerances for np.isclose() assessment of temperature/pressure validity
temperature_tol = dict(atol=1e-8, rtol=0)  # K
pressure_tol = dict(atol=1e-8, rtol=0)  # dbar


# Note that variable T in this module is equivalent to tempK elsewhere (in K),
# and P is equivalent to pres (in dbar), for convenience
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Zero functions ~~~~~
def bC_none(T, P):
    """c-a: no interaction effect."""
    b0 = 0
    b1 = 0
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = -9
    alph2 = -9
    omega = -9
    valid = T > 0
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def theta_none(T, P):
    """i-i': no interaction effect."""
    theta = 0
    valid = T > 0
    return theta, valid


def psi_none(T, P):
    """i-i'-j: no interaction effect."""
    psi = 0
    valid = T > 0
    return psi, valid


def lambd_none(T, P):
    """n-s: no interaction effect."""
    lambd = 0
    valid = T > 0
    return lambd, valid


def zeta_none(T, P):
    """n-c-a: no interaction effect."""
    zeta = 0
    valid = T > 0
    return zeta, valid


def mu_none(T, P):
    """n-n-n: no interaction effect."""
    mu = 0
    valid = T > 0
    return mu, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pitzer & Margoya (1973) ~~~~~
def bC_H_Cl_PM73(T, P):
    """ "c-a: hydrogen chloride [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.1775
    b1 = 0.2945
    b2 = 0
    Cphi = 0.0008
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["H"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_H_Br_PM73(T, P):
    """ "c-a: hydrogen bromide [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.196
    b1 = 0.3564
    b2 = 0
    Cphi = 0.00827
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["H"] * i2c["Br"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_H_I_PM73(T, P):
    """ "c-a: hydrogen iodide [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.2362
    b1 = 0.392
    b2 = 0
    Cphi = 0.0011
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["H"] * i2c["I"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_H_ClO4_PM73(T, P):
    """ "c-a: hydrogen perchlorate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.1747
    b1 = 0.2931
    b2 = 0
    Cphi = 0.00819
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["H"] * i2c["ClO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_H_NO3_PM73(T, P):
    """ "c-a: hydrogen nitrate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.1119
    b1 = 0.3206
    b2 = 0
    Cphi = 0.001
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["H"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Li_Cl_PM73(T, P):
    """ "c-a: lithium chloride [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.1494
    b1 = 0.3074
    b2 = 0
    Cphi = 0.00359
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Li"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Li_Br_PM73(T, P):
    """ "c-a: lithium bromide [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.1748
    b1 = 0.2547
    b2 = 0
    Cphi = 0.0053
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Li"] * i2c["Br"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Li_I_PM73(T, P):
    """ "c-a: lithium iodide [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.2104
    b1 = 0.373
    b2 = 0
    Cphi = 0.0
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Li"] * i2c["I"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Li_OH_PM73(T, P):
    """ "c-a: lithium hydroxide [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.015
    b1 = 0.14
    b2 = 0
    Cphi = 0.0
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Li"] * i2c["OH"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Li_ClO4_PM73(T, P):
    """ "c-a: lithium perchlorate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.1973
    b1 = 0.3996
    b2 = 0
    Cphi = 0.0008
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Li"] * i2c["ClO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Li_NO2_PM73(T, P):
    """ "c-a: lithium nitrite [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.1336
    b1 = 0.325
    b2 = 0
    Cphi = -0.0053
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Li"] * i2c["NO2"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Li_NO3_PM73(T, P):
    """ "c-a: lithium nitrate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.142
    b1 = 0.278
    b2 = 0
    Cphi = -0.00551
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Li"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_F_PM73(T, P):
    """ "c-a: sodium fluoride [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.0215
    b1 = 0.2107
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_Cl_PM73(T, P):
    """ "c-a: sodium chloride [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.0765
    b1 = 0.2664
    b2 = 0
    Cphi = 0.00127
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_Br_PM73(T, P):
    """ "c-a: sodium bromide [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.0973
    b1 = 0.2791
    b2 = 0
    Cphi = 0.00116
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["Br"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_I_PM73(T, P):
    """ "c-a: sodium iodide [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.1195
    b1 = 0.3439
    b2 = 0
    Cphi = 0.0018
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["I"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_OH_PM73(T, P):
    """ "c-a: sodium hydroxide [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.0864
    b1 = 0.253
    b2 = 0
    Cphi = 0.0044
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["OH"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_ClO3_PM73(T, P):
    """ "c-a: sodium chlorate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.0249
    b1 = 0.2455
    b2 = 0
    Cphi = 0.0004
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["ClO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_ClO4_PM73(T, P):
    """ "c-a: sodium perchlorate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.0554
    b1 = 0.2755
    b2 = 0
    Cphi = -0.00118
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["ClO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_BrO3_PM73(T, P):
    """ "c-a: sodium bromate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = -0.0205
    b1 = 0.191
    b2 = 0
    Cphi = 0.0059
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["BrO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_SCN_PM73(T, P):
    """ "c-a: sodium thiocyanate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.1005
    b1 = 0.3582
    b2 = 0
    Cphi = -0.00303
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["SCN"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_NO2_PM73(T, P):
    """ "c-a: sodium nitrite [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.0641
    b1 = 0.1015
    b2 = 0
    Cphi = -0.0049
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["NO2"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_NO3_PM73(T, P):
    """ "c-a: sodium nitrate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.0068
    b1 = 0.1783
    b2 = 0
    Cphi = -0.00072
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_H2PO4_PM73(T, P):
    """ "c-a: sodium dihydrogen-phosphate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = -0.0533
    b1 = 0.0396
    b2 = 0
    Cphi = 0.00795
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["H2PO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_H2AsO4_PM73(T, P):
    """ "c-a: sodium dihydrogen-arsenate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = -0.0442
    b1 = 0.2895
    b2 = 0
    Cphi = 0.0
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["H2AsO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_BO2_PM73(T, P):
    """ "c-a: sodium oxido(oxo)borane [PM73]."""
    # Coefficients from PM73 Table I
    b0 = -0.0526
    b1 = 0.1104
    b2 = 0
    Cphi = 0.0154
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["BO2"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_BF4_PM73(T, P):
    """ "c-a: sodium tetrafluoroborate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = -0.0252
    b1 = 0.1824
    b2 = 0
    Cphi = 0.0021
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["BF4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_F_PM73(T, P):
    """ "c-a: potassium fluoride [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.08089
    b1 = 0.2021
    b2 = 0
    Cphi = 0.00093
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["F"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_Cl_PM73(T, P):
    """ "c-a: potassium chloride [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.04835
    b1 = 0.2122
    b2 = 0
    Cphi = -0.00084
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_Br_PM73(T, P):
    """ "c-a: potassium bromide [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.0569
    b1 = 0.2212
    b2 = 0
    Cphi = -0.0018
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["Br"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_I_PM73(T, P):
    """ "c-a: potassium iodide [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.0746
    b1 = 0.2517
    b2 = 0
    Cphi = -0.00414
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["I"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_OH_PM73(T, P):
    """ "c-a: potassium hydroxide [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.1298
    b1 = 0.32
    b2 = 0
    Cphi = 0.0041
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["OH"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_ClO3_PM73(T, P):
    """ "c-a: potassium chlorate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = -0.096
    b1 = 0.2481
    b2 = 0
    Cphi = 0.0
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["ClO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_BrO3_PM73(T, P):
    """ "c-a: potassium bromate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = -0.129
    b1 = 0.2565
    b2 = 0
    Cphi = 0.0
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["BrO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_SCN_PM73(T, P):
    """ "c-a: potassium thiocyanate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.0416
    b1 = 0.2302
    b2 = 0
    Cphi = -0.00252
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["SCN"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_NO2_PM73(T, P):
    """ "c-a: potassium nitrite [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.0151
    b1 = 0.015
    b2 = 0
    Cphi = 0.0007
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["NO2"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_NO3_PM73(T, P):
    """ "c-a: potassium nitrate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = -0.0816
    b1 = 0.0494
    b2 = 0
    Cphi = 0.0066
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_H2PO4_PM73(T, P):
    """ "c-a: potassium dihydrogen-phosphate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = -0.0678
    b1 = -0.1042
    b2 = 0
    Cphi = 0.0
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["H2PO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_H2AsO4_PM73(T, P):
    """ "c-a: potassium dihydrogen-arsenate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = -0.0584
    b1 = 0.0626
    b2 = 0
    Cphi = 0.0
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["H2AsO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_PtF6_PM73(T, P):
    """ "c-a: potassium platinum-hexafluoride [PM73]."""
    # Coefficients from PM73 Table I
    b0 = -0.163
    b1 = -0.282
    b2 = 0
    Cphi = 0.0
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["PtF6"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Rb_F_PM73(T, P):
    """ "c-a: rubidium fluoride [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.1141
    b1 = 0.2842
    b2 = 0
    Cphi = -0.0105
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Rb"] * i2c["F"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Rb_Cl_PM73(T, P):
    """ "c-a: rubidium chloride [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.0441
    b1 = 0.1483
    b2 = 0
    Cphi = -0.00101
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Rb"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Rb_Br_PM73(T, P):
    """ "c-a: rubidium bromide [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.0396
    b1 = 0.153
    b2 = 0
    Cphi = -0.00144
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Rb"] * i2c["Br"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Rb_I_PM73(T, P):
    """ "c-a: rubidium iodide [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.0397
    b1 = 0.133
    b2 = 0
    Cphi = -0.00108
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Rb"] * i2c["I"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Rb_NO2_PM73(T, P):
    """ "c-a: rubidium nitrite [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.0269
    b1 = -0.1553
    b2 = 0
    Cphi = -0.00366
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Rb"] * i2c["NO2"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Rb_NO3_PM73(T, P):
    """ "c-a: rubidium nitrate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = -0.0789
    b1 = -0.0172
    b2 = 0
    Cphi = 0.00529
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Rb"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Cs_F_PM73(T, P):
    """ "c-a: caesium fluoride [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.1306
    b1 = 0.257
    b2 = 0
    Cphi = -0.0043
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Cs"] * i2c["F"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Cs_Cl_PM73(T, P):
    """ "c-a: caesium chloride [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.03
    b1 = 0.0558
    b2 = 0
    Cphi = 0.00038
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Cs"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Cs_Br_PM73(T, P):
    """ "c-a: caesium bromide [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.0279
    b1 = 0.0139
    b2 = 0
    Cphi = 4e-05
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Cs"] * i2c["Br"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Cs_I_PM73(T, P):
    """ "c-a: caesium iodide [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.0244
    b1 = 0.0262
    b2 = 0
    Cphi = -0.00365
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Cs"] * i2c["I"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Cs_OH_PM73(T, P):
    """ "c-a: caesium hydroxide [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.15
    b1 = 0.3
    b2 = 0
    Cphi = 0.0
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Cs"] * i2c["OH"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Cs_NO3_PM73(T, P):
    """ "c-a: caesium nitrate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = -0.0758
    b1 = -0.0669
    b2 = 0
    Cphi = 0.0
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Cs"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Cs_NO2_PM73(T, P):
    """ "c-a: caesium nitrite [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.0427
    b1 = 0.06
    b2 = 0
    Cphi = -0.0051
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Cs"] * i2c["NO2"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ag_NO3_PM73(T, P):
    """ "c-a: silver nitrate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = -0.0856
    b1 = 0.0025
    b2 = 0
    Cphi = 0.00591
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Ag"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Tl_ClO4_PM73(T, P):
    """ "c-a: thallium perchlorate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = -0.087
    b1 = -0.023
    b2 = 0
    Cphi = 0.0
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Tl"] * i2c["ClO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Tl_NO3_PM73(T, P):
    """ "c-a: thallium nitrate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = -0.105
    b1 = -0.378
    b2 = 0
    Cphi = 0.0
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Tl"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_NH4_Cl_PM73(T, P):
    """ "c-a: ammonium chloride [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.0522
    b1 = 0.1918
    b2 = 0
    Cphi = -0.00301
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["NH4"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_NH4_Br_PM73(T, P):
    """ "c-a: ammonium bromide [PM73]."""
    # Coefficients from PM73 Table I
    b0 = 0.0624
    b1 = 0.1947
    b2 = 0
    Cphi = -0.00436
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["NH4"] * i2c["Br"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_NH4_ClO4_PM73(T, P):
    """ "c-a: ammonium perchlorate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = -0.0103
    b1 = -0.0194
    b2 = 0
    Cphi = 0.0
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["NH4"] * i2c["ClO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_NH4_NO3_PM73(T, P):
    """ "c-a: ammonium nitrate [PM73]."""
    # Coefficients from PM73 Table I
    b0 = -0.0154
    b1 = 0.112
    b2 = 0
    Cphi = -3e-05
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["NH4"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mg_Cl_PM73(T, P):
    """ "c-a: magnesium chloride [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.4698 * 3 / 4
    b1 = 2.242 * 3 / 4
    b2 = 0
    Cphi = 0.00979 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Mg"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mg_Br_PM73(T, P):
    """ "c-a: magnesium bromide [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.5769 * 3 / 4
    b1 = 2.337 * 3 / 4
    b2 = 0
    Cphi = 0.00589 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Mg"] * i2c["Br"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mg_I_PM73(T, P):
    """ "c-a: magnesium iodide [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.6536 * 3 / 4
    b1 = 2.4055 * 3 / 4
    b2 = 0
    Cphi = 0.01496 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Mg"] * i2c["I"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mg_ClO4_PM73(T, P):
    """ "c-a: magnesium perchlorate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.6615 * 3 / 4
    b1 = 2.678 * 3 / 4
    b2 = 0
    Cphi = 0.01806 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Mg"] * i2c["ClO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mg_NO3_PM73(T, P):
    """ "c-a: magnesium nitrate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.4895 * 3 / 4
    b1 = 2.113 * 3 / 4
    b2 = 0
    Cphi = -0.03889 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Mg"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ca_Cl_PM73(T, P):
    """ "c-a: calcium chloride [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.4212 * 3 / 4
    b1 = 2.152 * 3 / 4
    b2 = 0
    Cphi = -0.00064 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Ca"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ca_Br_PM73(T, P):
    """ "c-a: calcium bromide [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.5088 * 3 / 4
    b1 = 2.151 * 3 / 4
    b2 = 0
    Cphi = -0.00485 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Ca"] * i2c["Br"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ca_I_PM73(T, P):
    """ "c-a: calcium iodide [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.5839 * 3 / 4
    b1 = 2.409 * 3 / 4
    b2 = 0
    Cphi = -0.00158 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Ca"] * i2c["I"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ca_ClO4_PM73(T, P):
    """ "c-a: calcium perchlorate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.6015 * 3 / 4
    b1 = 2.342 * 3 / 4
    b2 = 0
    Cphi = -0.00943 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Ca"] * i2c["ClO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ca_NO3_PM73(T, P):
    """ "c-a: calcium nitrate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.2811 * 3 / 4
    b1 = 1.879 * 3 / 4
    b2 = 0
    Cphi = -0.03798 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Ca"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Sr_Cl_PM73(T, P):
    """ "c-a: strontium chloride [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.381 * 3 / 4
    b1 = 2.223 * 3 / 4
    b2 = 0
    Cphi = -0.00246 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Sr"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Sr_Br_PM73(T, P):
    """ "c-a: strontium bromide [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.4415 * 3 / 4
    b1 = 2.282 * 3 / 4
    b2 = 0
    Cphi = 0.00231 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Sr"] * i2c["Br"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Sr_I_PM73(T, P):
    """ "c-a: strontium iodide [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.535 * 3 / 4
    b1 = 2.48 * 3 / 4
    b2 = 0
    Cphi = 0.00501 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Sr"] * i2c["I"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Sr_ClO4_PM73(T, P):
    """ "c-a: strontium perchlorate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.5692 * 3 / 4
    b1 = 2.089 * 3 / 4
    b2 = 0
    Cphi = -0.02472 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Sr"] * i2c["ClO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Sr_NO3_PM73(T, P):
    """ "c-a: strontium nitrate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.1795 * 3 / 4
    b1 = 1.84 * 3 / 4
    b2 = 0
    Cphi = -0.03757 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Sr"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ba_Cl_PM73(T, P):
    """ "c-a: barium chloride [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.3504 * 3 / 4
    b1 = 1.995 * 3 / 4
    b2 = 0
    Cphi = -0.03654 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Ba"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ba_Br_PM73(T, P):
    """ "c-a: barium bromide [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.4194 * 3 / 4
    b1 = 2.093 * 3 / 4
    b2 = 0
    Cphi = -0.03009 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Ba"] * i2c["Br"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ba_I_PM73(T, P):
    """ "c-a: barium iodide [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.5625 * 3 / 4
    b1 = 2.249 * 3 / 4
    b2 = 0
    Cphi = -0.03286 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Ba"] * i2c["I"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ba_OH_PM73(T, P):
    """ "c-a: barium hydroxide [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.229 * 3 / 4
    b1 = 1.6 * 3 / 4
    b2 = 0
    Cphi = 0.0 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Ba"] * i2c["OH"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ba_ClO4_PM73(T, P):
    """ "c-a: barium perchlorate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.4819 * 3 / 4
    b1 = 2.101 * 3 / 4
    b2 = 0
    Cphi = -0.05894 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Ba"] * i2c["ClO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ba_NO3_PM73(T, P):
    """ "c-a: barium nitrate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = -0.043 * 3 / 4
    b1 = 1.07 * 3 / 4
    b2 = 0
    Cphi = 0.0 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Ba"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mnjj_Cl_PM73(T, P):
    """ "c-a: manganese(II) chloride [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.1363 * 3 / 4
    b1 = 2.067 * 3 / 4
    b2 = 0
    Cphi = -0.03865 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Mnjj"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Fejj_Cl_PM73(T, P):
    """ "c-a: iron(II) chloride [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.4479 * 3 / 4
    b1 = 2.043 * 3 / 4
    b2 = 0
    Cphi = -0.01623 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Fejj"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Cojj_Cl_PM73(T, P):
    """ "c-a: cobalt(II) chloride [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.4857 * 3 / 4
    b1 = 1.936 * 3 / 4
    b2 = 0
    Cphi = -0.02869 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Cojj"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Cojj_Br_PM73(T, P):
    """ "c-a: cobalt(II) bromide [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.5693 * 3 / 4
    b1 = 2.213 * 3 / 4
    b2 = 0
    Cphi = -0.00127 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Cojj"] * i2c["Br"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Cojj_I_PM73(T, P):
    """ "c-a: cobalt(II) iodide [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.695 * 3 / 4
    b1 = 2.23 * 3 / 4
    b2 = 0
    Cphi = -0.0088 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Cojj"] * i2c["I"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Cojj_NO3_PM73(T, P):
    """ "c-a: cobalt(II) nitrate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.4159 * 3 / 4
    b1 = 2.254 * 3 / 4
    b2 = 0
    Cphi = -0.01436 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Cojj"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Nijj_Cl_PM73(T, P):
    """ "c-a: nickel(II) chloride [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.4639 * 3 / 4
    b1 = 2.108 * 3 / 4
    b2 = 0
    Cphi = -0.00702 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Nijj"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Cujj_Cl_PM73(T, P):
    """ "c-a: copper(II) chloride [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.4107 * 3 / 4
    b1 = 1.835 * 3 / 4
    b2 = 0
    Cphi = -0.07624 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Cujj"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Cujj_NO3_PM73(T, P):
    """ "c-a: copper(II) nitrate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.4224 * 3 / 4
    b1 = 1.907 * 3 / 4
    b2 = 0
    Cphi = -0.04136 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Cujj"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Znjj_Cl_PM73(T, P):
    """ "c-a: zinc(II) chloride [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.3469 * 3 / 4
    b1 = 2.19 * 3 / 4
    b2 = 0
    Cphi = -0.1659 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Znjj"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Znjj_Br_PM73(T, P):
    """ "c-a: zinc(II) bromide [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.6213 * 3 / 4
    b1 = 2.179 * 3 / 4
    b2 = 0
    Cphi = -0.2035 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Znjj"] * i2c["Br"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Znjj_I_PM73(T, P):
    """ "c-a: zinc(II) iodide [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.6428 * 3 / 4
    b1 = 2.594 * 3 / 4
    b2 = 0
    Cphi = -0.0269 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Znjj"] * i2c["I"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Znjj_ClO4_PM73(T, P):
    """ "c-a: zinc(II) perchlorate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.6747 * 3 / 4
    b1 = 2.396 * 3 / 4
    b2 = 0
    Cphi = 0.02134 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Znjj"] * i2c["ClO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Znjj_NO3_PM73(T, P):
    """ "c-a: zinc(II) nitrate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.4641 * 3 / 4
    b1 = 2.255 * 3 / 4
    b2 = 0
    Cphi = -0.02955 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Znjj"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Cdjj_NO3_PM73(T, P):
    """ "c-a: cadmium(II) nitrate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.382 * 3 / 4
    b1 = 2.224 * 3 / 4
    b2 = 0
    Cphi = -0.04836 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Cdjj"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Pbjj_ClO4_PM73(T, P):
    """ "c-a: lead(II) perchlorate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.4443 * 3 / 4
    b1 = 2.296 * 3 / 4
    b2 = 0
    Cphi = -0.01667 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Pbjj"] * i2c["ClO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Pbjj_NO3_PM73(T, P):
    """ "c-a: lead(II) nitrate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = -0.0482 * 3 / 4
    b1 = 0.38 * 3 / 4
    b2 = 0
    Cphi = 0.01005 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Pbjj"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_UO2_Cl_PM73(T, P):
    """ "c-a: uranium-dioxide chloride [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.5698 * 3 / 4
    b1 = 2.192 * 3 / 4
    b2 = 0
    Cphi = -0.06951 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["UO2"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_UO2_ClO4_PM73(T, P):
    """ "c-a: uranium-dioxide perchlorate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.8151 * 3 / 4
    b1 = 2.859 * 3 / 4
    b2 = 0
    Cphi = 0.04089 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["UO2"] * i2c["ClO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_UO2_NO3_PM73(T, P):
    """ "c-a: uranium-dioxide nitrate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.6143 * 3 / 4
    b1 = 2.151 * 3 / 4
    b2 = 0
    Cphi = -0.05948 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["UO2"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Li_SO4_PM73(T, P):
    """ "c-a: lithium sulfate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.1817 * 3 / 4
    b1 = 1.694 * 3 / 4
    b2 = 0
    Cphi = -0.00753 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Li"] * i2c["SO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_SO4_PM73(T, P):
    """ "c-a: sodium sulfate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.0261 * 3 / 4
    b1 = 1.484 * 3 / 4
    b2 = 0
    Cphi = 0.00938 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["SO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_S2O3_PM73(T, P):
    """ "c-a: sodium thiosulfate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.0882 * 3 / 4
    b1 = 1.701 * 3 / 4
    b2 = 0
    Cphi = 0.00705 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["S2O3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_CrO4_PM73(T, P):
    """ "c-a: sodium chromate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.125 * 3 / 4
    b1 = 1.826 * 3 / 4
    b2 = 0
    Cphi = -0.00407 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["CrO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_CO3_PM73(T, P):
    """ "c-a: sodium carbonate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.253 * 3 / 4
    b1 = 1.128 * 3 / 4
    b2 = 0
    Cphi = -0.09057 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["CO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_HPO4_PM73(T, P):
    """ "c-a: sodium hydrogen-phosphate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = -0.0777 * 3 / 4
    b1 = 1.954 * 3 / 4
    b2 = 0
    Cphi = 0.0554 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["HPO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_HAsO4_PM73(T, P):
    """ "c-a: sodium hydrogen-arsenate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.0407 * 3 / 4
    b1 = 2.173 * 3 / 4
    b2 = 0
    Cphi = 0.0034 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["HAsO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_SO4_PM73(T, P):
    """ "c-a: potassium sulfate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.0666 * 3 / 4
    b1 = 1.039 * 3 / 4
    b2 = 0
    Cphi = 0.0 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["SO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_CrO4_PM73(T, P):
    """ "c-a: potassium chromate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.1011 * 3 / 4
    b1 = 1.652 * 3 / 4
    b2 = 0
    Cphi = -0.00147 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["CrO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_PtCN4_PM73(T, P):
    """ "c-a: potassium platinocyanide [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.0881 * 3 / 4
    b1 = 3.164 * 3 / 4
    b2 = 0
    Cphi = 0.0247 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["PtCN4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_HPO4_PM73(T, P):
    """ "c-a: potassium hydrogen-phosphate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.033 * 3 / 4
    b1 = 1.699 * 3 / 4
    b2 = 0
    Cphi = 0.0309 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["HPO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_HAsO4_PM73(T, P):
    """ "c-a: potassium hydrogen-arsenate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.1728 * 3 / 4
    b1 = 2.198 * 3 / 4
    b2 = 0
    Cphi = -0.0336 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["HAsO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Rb_SO4_PM73(T, P):
    """ "c-a: rubidium sulfate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.0772 * 3 / 4
    b1 = 1.481 * 3 / 4
    b2 = 0
    Cphi = -0.00019 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Rb"] * i2c["SO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Cs_SO4_PM73(T, P):
    """ "c-a: caesium sulfate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.1184 * 3 / 4
    b1 = 1.481 * 3 / 4
    b2 = 0
    Cphi = -0.01131 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Cs"] * i2c["SO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_NH4_SO4_PM73(T, P):
    """ "c-a: ammonium sulfate [PM73]."""
    # Coefficients from PM73 Table VI
    b0 = 0.0545 * 3 / 4
    b1 = 0.878 * 3 / 4
    b2 = 0
    Cphi = -0.00219 * 3 / 2 ** (5 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["NH4"] * i2c["SO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Aljjj_Cl_PM73(T, P):
    """ "c-a: aluminium(III) chloride [PM73]."""
    # Coefficients from PM73 Table VIII
    b0 = 1.049 * 2 / 3
    b1 = 8.767 * 2 / 3
    b2 = 0
    Cphi = 0.0071 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Aljjj"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Srjjj_Cl_PM73(T, P):
    """ "c-a: strontium(III) chloride [PM73]."""
    # Coefficients from PM73 Table VIII
    b0 = 1.05 * 2 / 3
    b1 = 7.978 * 2 / 3
    b2 = 0
    Cphi = -0.084 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Srjjj"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Y_Cl_PM73(T, P):
    """ "c-a: yttrium chloride [PM73]."""
    # Coefficients from PM73 Table VIII
    b0 = 0.9599 * 2 / 3
    b1 = 8.166 * 2 / 3
    b2 = 0
    Cphi = -0.0587 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Y"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_La_Cl_PM73(T, P):
    """ "c-a: lanthanum chloride [PM73]."""
    # Coefficients from PM73 Table VIII
    b0 = 0.9158 * 2 / 3
    b1 = 8.231 * 2 / 3
    b2 = 0
    Cphi = -0.0831 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["La"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ce_Cl_PM73(T, P):
    """ "c-a: cerium chloride [PM73]."""
    # Coefficients from PM73 Table VIII
    b0 = 0.9187 * 2 / 3
    b1 = 8.227 * 2 / 3
    b2 = 0
    Cphi = -0.0809 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Ce"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Pr_Cl_PM73(T, P):
    """ "c-a: praeseodymium chloride [PM73]."""
    # Coefficients from PM73 Table VIII
    b0 = 0.903 * 2 / 3
    b1 = 8.181 * 2 / 3
    b2 = 0
    Cphi = -0.0727 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Pr"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Nd_Cl_PM73(T, P):
    """ "c-a: neodymium chloride [PM73]."""
    # Coefficients from PM73 Table VIII
    b0 = 0.9175 * 2 / 3
    b1 = 8.104 * 2 / 3
    b2 = 0
    Cphi = -0.0737 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Nd"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Sm_Cl_PM73(T, P):
    """ "c-a: samarium chloride [PM73]."""
    # Coefficients from PM73 Table VIII
    b0 = 0.933 * 2 / 3
    b1 = 8.273 * 2 / 3
    b2 = 0
    Cphi = -0.0728 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Sm"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Eujjj_Cl_PM73(T, P):
    """ "c-a: europium (III) chloride [PM73]."""
    # Coefficients from PM73 Table VIII
    b0 = 0.937 * 2 / 3
    b1 = 8.385 * 2 / 3
    b2 = 0
    Cphi = -0.0687 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Eu"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Cr_Cl_PM73(T, P):
    """ "c-a: chromium chloride [PM73]."""
    # Coefficients from PM73 Table VIII
    b0 = 1.1046 * 2 / 3
    b1 = 7.883 * 2 / 3
    b2 = 0
    Cphi = -0.1172 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Cr"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Cr_NO3_PM73(T, P):
    """ "c-a: chromium nitrate [PM73]."""
    # Coefficients from PM73 Table VIII
    b0 = 1.056 * 2 / 3
    b1 = 7.777 * 2 / 3
    b2 = 0
    Cphi = -0.1533 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Cr"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ga_ClO4_PM73(T, P):
    """ "c-a: gallium perchlorate [PM73]."""
    # Coefficients from PM73 Table VIII
    b0 = 1.2381 * 2 / 3
    b1 = 9.794 * 2 / 3
    b2 = 0
    Cphi = 0.0904 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Ga"] * i2c["ClO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_In_Cl_PM73(T, P):
    """ "c-a: indium chloride [PM73]."""
    # Coefficients from PM73 Table VIII
    b0 = -1.68 * 2 / 3
    b1 = -3.85 * 2 / 3
    b2 = 0
    Cphi = 0.0 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["In"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_PO4_PM73(T, P):
    """ "c-a: sodium phosphate [PM73]."""
    # Coefficients from PM73 Table VIII
    b0 = 0.2672 * 2 / 3
    b1 = 5.777 * 2 / 3
    b2 = 0
    Cphi = -0.1339 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["PO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_AsO4_PM73(T, P):
    """ "c-a: sodium arsenate [PM73]."""
    # Coefficients from PM73 Table VIII
    b0 = 0.3582 * 2 / 3
    b1 = 5.895 * 2 / 3
    b2 = 0
    Cphi = -0.124 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["AsO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_PO4_PM73(T, P):
    """ "c-a: potassium phosphate [PM73]."""
    # Coefficients from PM73 Table VIII
    b0 = 0.5594 * 2 / 3
    b1 = 5.958 * 2 / 3
    b2 = 0
    Cphi = -0.2255 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["PO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_P3O9_PM73(T, P):
    """ "c-a: potassium trimetaphosphate [PM73]."""
    # Coefficients from PM73 Table VIII
    b0 = 0.4867 * 2 / 3
    b1 = 8.349 * 2 / 3
    b2 = 0
    Cphi = -0.0886 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["P3O9"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_AsO4_PM73(T, P):
    """ "c-a: potassium arsenate [PM73]."""
    # Coefficients from PM73 Table VIII
    b0 = 0.7491 * 2 / 3
    b1 = 6.511 * 2 / 3
    b2 = 0
    Cphi = -0.3376 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["AsO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_FejjjCN6_PM73(T, P):
    """ "c-a: potassium ferricyanide [PM73]."""
    # Coefficients from PM73 Table VIII
    b0 = 0.5035 * 2 / 3
    b1 = 7.121 * 2 / 3
    b2 = 0
    Cphi = -0.1176 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["FejjjCN6"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_CoCN6_PM73(T, P):
    """ "c-a: potassium Co(CN)6 [PM73]."""
    # Coefficients from PM73 Table VIII
    b0 = 0.5603 * 2 / 3
    b1 = 5.815 * 2 / 3
    b2 = 0
    Cphi = -0.1603 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["CoCN6"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Coen3_Cl_PM73(T, P):
    """ "c-a: tris(ethylenediamine)cobalt(III) chloride [PM73]."""
    # Coefficients from PM73 Table VIII
    b0 = 0.2603 * 2 / 3
    b1 = 3.563 * 2 / 3
    b2 = 0
    Cphi = -0.0916 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Coen3"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Coen3_NO3_PM73(T, P):
    """ "c-a: tris(ethylenediamine)cobalt(III) nitrate [PM73]."""
    # Coefficients from PM73 Table VIII
    b0 = 0.1882 * 2 / 3
    b1 = 3.935 * 2 / 3
    b2 = 0
    Cphi = 0.0 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Coen3"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Coen3_ClO4_PM73(T, P):
    """ "c-a: tris(ethylenediamine)cobalt(III) perchlorate [PM73]."""
    # Coefficients from PM73 Table VIII
    b0 = 0.1619 * 2 / 3
    b1 = 5.395 * 2 / 3
    b2 = 0
    Cphi = 0.0 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Coen3"] * i2c["ClO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Copn3_ClO4_PM73(T, P):
    """ "c-a: Copn3 perchlorate [PM73]."""
    # Coefficients from PM73 Table VIII
    b0 = 0.2022 * 2 / 3
    b1 = 3.976 * 2 / 3
    b2 = 0
    Cphi = 0.0 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Copn3"] * i2c["ClO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Th_Cl_PM73(T, P):
    """ "c-a: thorium chloride [PM73]."""
    # Coefficients from PM73 Table IX
    b0 = 1.622 * 2 / 3
    b1 = 21.33 * 2 / 3
    b2 = 0
    Cphi = -0.3309 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Th"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Th_NO3_PM73(T, P):
    """ "c-a: thorium nitrate [PM73]."""
    # Coefficients from PM73 Table IX
    b0 = 1.546 * 2 / 3
    b1 = 18.22 * 2 / 3
    b2 = 0
    Cphi = -0.5906 * 2 / 3 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Th"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_P2O7_PM73(T, P):
    """ "c-a: sodium diphosphate [PM73]."""
    # Coefficients from PM73 Table IX
    b0 = 0.699 * 5 / 8
    b1 = 17.16 * 5 / 8
    b2 = 0
    Cphi = 0.0 * 5 / 16
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["P2O7"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_P2O7_PM73(T, P):
    """ "c-a: potassium diphosphate [PM73]."""
    # Coefficients from PM73 Table IX
    b0 = 0.977 * 5 / 8
    b1 = 17.88 * 5 / 8
    b2 = 0
    Cphi = -0.2418 * 5 / 16
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["P2O7"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_FejjCN6_PM73(T, P):
    """ "c-a: potassium ferrocyanide [PM73]."""
    # Coefficients from PM73 Table IX
    b0 = 1.021 * 5 / 8
    b1 = 16.23 * 5 / 8
    b2 = 0
    Cphi = -0.5579 * 5 / 16
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["FejjCN6"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_MoCN8_PM73(T, P):
    """ "c-a: potassium Mo(CN)8 [PM73]."""
    # Coefficients from PM73 Table IX
    b0 = 0.854 * 5 / 8
    b1 = 18.53 * 5 / 8
    b2 = 0
    Cphi = -0.3499 * 5 / 16
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["MoCN8"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_WCN8_PM73(T, P):
    """ "c-a: potassium W(CN)8 [PM73]."""
    # Coefficients from PM73 Table IX
    b0 = 1.032 * 5 / 8
    b1 = 18.49 * 5 / 8
    b2 = 0
    Cphi = -0.4937 * 5 / 16
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["WCN8"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_MeN_MoCN8_PM73(T, P):
    """ "c-a: MeN Mo(CN)8 [PM73]."""
    # Coefficients from PM73 Table IX
    b0 = 0.938 * 5 / 8
    b1 = 15.91 * 5 / 8
    b2 = 0
    Cphi = -0.333 * 5 / 16
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["MeN"] * i2c["MoCN8"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_P3O10_PM73(T, P):
    """ "c-a: sodium triphosphate-pentaanion [PM73]."""
    # Coefficients from PM73 Table IX
    b0 = 1.869 * 3 / 5
    b1 = 36.1 * 3 / 5
    b2 = 0
    Cphi = -0.163 * 3 / 5 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["P3O10"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_P3O10_PM73(T, P):
    """ "c-a: potassium triphosphate-pentaanion [PM73]."""
    # Coefficients from PM73 Table IX
    b0 = 1.939 * 3 / 5
    b1 = 39.64 * 3 / 5
    b2 = 0
    Cphi = -0.1055 * 3 / 5 ** (3 / 2)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["P3O10"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


# Manual additions - not from funcgen!
def bC_Na_acetate_PM73(T, P):
    """c-a: sodium acetate [PM73]."""
    # Coefficients from PM73 Table II
    b0 = 0.1426
    b1 = 0.3237
    b2 = 0
    Cphi = -0.00629
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["acetate"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_acetate_PM73(T, P):
    """c-a: potassium acetate [PM73]."""
    # Coefficients from PM73 Table II
    b0 = 0.1587
    b1 = 0.3251
    b2 = 0
    Cphi = -0.00660
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["acetate"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pitzer and Kim (1974) ~~~~~
def theta_Mg_Na_PK74(T, P):
    """c-c': magnesium sodium [PK74]."""
    theta = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_Ca_Na_PK74(T, P):
    """c-c': calcium sodium [PK74]."""
    theta = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_K_Na_PK74(T, P):
    """c-c': potassium sodium [PK74]."""
    theta = -0.012
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_Li_Na_PK74(T, P):
    """c-c': lithium sodium [PK74]."""
    theta = 0.012
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_Ba_Na_PK74(T, P):
    """c-c': barium sodium [PK74]."""
    theta = -0.003
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_Na_Znjj_PK74(T, P):
    """c-c': sodium zinc(II) [PK74]."""
    theta = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_Mnjj_Na_PK74(T, P):
    """c-c': manganese(II) sodium [PK74]."""
    theta = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_Cs_Na_PK74(T, P):
    """c-c': caesium sodium [PK74]."""
    theta = -0.033
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_H_Na_PK74(T, P):
    """c-c': hydrogen sodium [PK74]."""
    theta = 0.036
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_Ca_Mg_PK74(T, P):
    """c-c': calcium magnesium [PK74]."""
    theta = 0.01
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_Ca_K_PK74(T, P):
    """c-c': calcium potassium [PK74]."""
    theta = -0.04
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_K_Li_PK74(T, P):
    """c-c': potassium lithium [PK74]."""
    theta = -0.022
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_Ba_K_PK74(T, P):
    """c-c': barium potassium [PK74]."""
    theta = -0.072
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_Cs_K_PK74(T, P):
    """c-c': caesium potassium [PK74]."""
    theta = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_H_K_PK74(T, P):
    """c-c': hydrogen potassium [PK74]."""
    theta = 0.005
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_H_Sr_PK74(T, P):
    """c-c': hydrogen strontium [PK74]."""
    theta = -0.02
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_Ba_Li_PK74(T, P):
    """c-c': barium lithium [PK74]."""
    theta = -0.07
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_Cs_Li_PK74(T, P):
    """c-c': caesium lithium [PK74]."""
    theta = -0.095
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_H_Li_PK74(T, P):
    """c-c': hydrogen lithium [PK74]."""
    theta = 0.015
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_H_NH4_PK74(T, P):
    """c-c': hydrogen ammonium [PK74]."""
    theta = -0.016
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_Ba_Cs_PK74(T, P):
    """c-c': barium caesium [PK74]."""
    theta = -0.15
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_Ba_H_PK74(T, P):
    """c-c': barium hydrogen [PK74]."""
    theta = -0.036
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_H_Mnjj_PK74(T, P):
    """c-c': hydrogen manganese(II) [PK74]."""
    theta = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_Cs_H_PK74(T, P):
    """c-c': caesium hydrogen [PK74]."""
    theta = -0.044
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_Et4N_H_PK74(T, P):
    """c-c': tetraethylammonium hydrogen [PK74]."""
    theta = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_H_Me4N_PK74(T, P):
    """c-c': hydrogen tetramethylammonium [PK74]."""
    theta = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_Cl_SO4_PK74(T, P):
    """a-a': chloride sulfate [PK74]."""
    theta = -0.035
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_Br_Cl_PK74(T, P):
    """a-a': bromide chloride [PK74]."""
    theta = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_Cl_NO3_PK74(T, P):
    """a-a': chloride nitrate [PK74]."""
    theta = 0.016
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_Cl_OH_PK74(T, P):
    """a-a': chloride hydroxide [PK74]."""
    theta = -0.05
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_Br_OH_PK74(T, P):
    """a-a': bromide hydroxide [PK74]."""
    theta = -0.065
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Mg_Na_Cl_PK74(T, P):
    """c-c'-a: magnesium sodium chloride [PK74]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_Na_Cl_PK74(T, P):
    """c-c'-a: calcium sodium chloride [PK74]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Na_Cl_PK74(T, P):
    """c-c'-a: potassium sodium chloride [PK74]."""
    psi = -0.0018
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Li_Na_Cl_PK74(T, P):
    """c-c'-a: lithium sodium chloride [PK74]."""
    psi = -0.003
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ba_Na_Cl_PK74(T, P):
    """c-c'-a: barium sodium chloride [PK74]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mnjj_Na_Cl_PK74(T, P):
    """c-c'-a: manganese(II) sodium chloride [PK74]."""
    psi = -0.003
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Cs_Na_Cl_PK74(T, P):
    """c-c'-a: caesium sodium chloride [PK74]."""
    psi = -0.003
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Na_Cl_PK74(T, P):
    """c-c'-a: hydrogen sodium chloride [PK74]."""
    psi = -0.004
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_Mg_Cl_PK74(T, P):
    """c-c'-a: calcium magnesium chloride [PK74]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_K_Cl_PK74(T, P):
    """c-c'-a: calcium potassium chloride [PK74]."""
    psi = -0.015
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Li_Cl_PK74(T, P):
    """c-c'-a: potassium lithium chloride [PK74]."""
    psi = -0.01
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ba_K_Cl_PK74(T, P):
    """c-c'-a: barium potassium chloride [PK74]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Cs_K_Cl_PK74(T, P):
    """c-c'-a: caesium potassium chloride [PK74]."""
    psi = -0.0013
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_K_Cl_PK74(T, P):
    """c-c'-a: hydrogen potassium chloride [PK74]."""
    psi = -0.007
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Sr_Cl_PK74(T, P):
    """c-c'-a: hydrogen strontium chloride [PK74]."""
    psi = 0.018
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ba_Li_Cl_PK74(T, P):
    """c-c'-a: barium lithium chloride [PK74]."""
    psi = 0.019
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Cs_Li_Cl_PK74(T, P):
    """c-c'-a: caesium lithium chloride [PK74]."""
    psi = -0.0094
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Li_Cl_PK74(T, P):
    """c-c'-a: hydrogen lithium chloride [PK74]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_NH4_Cl_PK74(T, P):
    """c-c'-a: hydrogen ammonium chloride [PK74]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ba_Cs_Cl_PK74(T, P):
    """c-c'-a: barium caesium chloride [PK74]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ba_H_Cl_PK74(T, P):
    """c-c'-a: barium hydrogen chloride [PK74]."""
    psi = 0.024
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Mnjj_Cl_PK74(T, P):
    """c-c'-a: hydrogen manganese(II) chloride [PK74]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Cs_H_Cl_PK74(T, P):
    """c-c'-a: caesium hydrogen chloride [PK74]."""
    psi = -0.019
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Et4N_H_Cl_PK74(T, P):
    """c-c'-a: tetraethylammonium hydrogen chloride [PK74]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Me4N_Cl_PK74(T, P):
    """c-c'-a: hydrogen tetramethylammonium chloride [PK74]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_Na_SO4_PK74(T, P):
    """c-c'-a: magnesium sodium sulfate [PK74]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Na_SO4_PK74(T, P):
    """c-c'-a: potassium sodium sulfate [PK74]."""
    psi = -0.01
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Na_Br_PK74(T, P):
    """c-c'-a: potassium sodium bromide [PK74]."""
    psi = -0.0022
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Na_Znjj_Br_PK74(T, P):
    """c-c'-a: sodium zinc(II) bromide [PK74]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Na_Br_PK74(T, P):
    """c-c'-a: hydrogen sodium bromide [PK74]."""
    psi = -0.012
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_K_Br_PK74(T, P):
    """c-c'-a: hydrogen potassium bromide [PK74]."""
    psi = -0.021
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Li_Br_PK74(T, P):
    """c-c'-a: hydrogen lithium bromide [PK74]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Na_NO3_PK74(T, P):
    """c-c'-a: potassium sodium nitrate [PK74]."""
    psi = -0.0012
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Li_Na_NO3_PK74(T, P):
    """c-c'-a: lithium sodium nitrate [PK74]."""
    psi = -0.0072
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Li_Na_ClO4_PK74(T, P):
    """c-c'-a: lithium sodium perchlorate [PK74]."""
    psi = -0.008
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Na_ClO4_PK74(T, P):
    """c-c'-a: hydrogen sodium perchlorate [PK74]."""
    psi = -0.016
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Li_ClO4_PK74(T, P):
    """c-c'-a: hydrogen lithium perchlorate [PK74]."""
    psi = -0.0017
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Li_Na_OAc_PK74(T, P):
    """c-c'-a: lithium sodium OAc [PK74]."""
    psi = -0.0043
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Na_Cl_SO4_PK74(T, P):
    """c-a-a': sodium chloride sulfate [PK74]."""
    psi = 0.007
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Na_Br_Cl_PK74(T, P):
    """c-a-a': sodium bromide chloride [PK74]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Na_Cl_NO3_PK74(T, P):
    """c-a-a': sodium chloride nitrate [PK74]."""
    psi = -0.006
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Na_Cl_OH_PK74(T, P):
    """c-a-a': sodium chloride hydroxide [PK74]."""
    psi = -0.006
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Na_Br_OH_PK74(T, P):
    """c-a-a': sodium bromide hydroxide [PK74]."""
    psi = -0.018
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_Cl_SO4_PK74(T, P):
    """c-a-a': magnesium chloride sulfate [PK74]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_Cl_NO3_PK74(T, P):
    """c-a-a': magnesium chloride nitrate [PK74]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_Cl_NO3_PK74(T, P):
    """c-a-a': calcium chloride nitrate [PK74]."""
    psi = -0.017
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Cl_SO4_PK74(T, P):
    """c-a-a': potassium chloride sulfate [PK74]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Br_Cl_PK74(T, P):
    """c-a-a': potassium bromide chloride [PK74]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Cl_NO3_PK74(T, P):
    """c-a-a': potassium chloride nitrate [PK74]."""
    psi = -0.006
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Cl_OH_PK74(T, P):
    """c-a-a': potassium chloride hydroxide [PK74]."""
    psi = -0.008
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Br_OH_PK74(T, P):
    """c-a-a': potassium bromide hydroxide [PK74]."""
    psi = -0.014
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Li_Cl_NO3_PK74(T, P):
    """c-a-a': lithium chloride nitrate [PK74]."""
    psi = -0.003
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pitzer and Silvester (1976) ~~~~~
def lambd_H3PO4_H3PO4_PS76(T, P):
    """n-n: phosphoric-acid phosphoric-acid [PS76]."""
    lambd = 0.05031
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_H3PO4_H2PO4_PS76(T, P):
    """n-a: phosphoric-acid dihydrogen-phosphate [PS76]."""
    lambd = -0.400
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_H3PO4_K_PS76(T, P):
    """n-c: phosphoric-acid potassium [PS76]."""
    lambd = -0.070
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_H3PO4_H_PS76(T, P):
    """n-c: phosphoric-acid hydrogen [PS76]."""
    lambd = 0.290
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_H3PO4_Cl_PS76(T, P):
    """n-a: phosphoric-acid chloride [PS76]."""
    lambd = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def mu_H3PO4_H3PO4_H3PO4_PS76(T, P):
    """n-n-n: phosphoric-acid phosphoric-acid phosphoric-acid [PS76]."""
    mu = 0.01095
    valid = np.isclose(T, 298.15, **temperature_tol)
    return mu, valid


def theta_Cl_H2PO4_PS76(T, P):
    """a-a': chloride dihydrogen-phosphate [PS76]."""
    theta = 0.10
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_K_Cl_H2PO4_PS76(T, P):
    """c-a-a': potassium chloride dihydrogen-phosphate [PS76]."""
    psi = -0.0105
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Mackaskill et al. (1978) ~~~~~
def bC_Sr_Cl_MWRB78(T, P):
    """c-a: strontium chloride [MWRB78]."""
    # Valid up to ionic strength of 6, different set provided for up to 9
    # This set should be 'better' for values under 6
    b0 = 0.28994
    b1 = 1.5795
    b2 = 0
    Cphi = -0.003755
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Sr"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Sr_Cl_MWRB78hi(T, P):
    """c-a: strontium chloride [MWRB78]."""
    # Valid up to ionic strength of 9, different set provided for up to 6
    b0 = 0.27948
    b1 = 1.6745
    b2 = 0
    Cphi = 0.0003532
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Sr"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def theta_Na_Sr_MWRB78(T, P):
    """c-c': sodium strontium [MWRB78]."""
    theta = -0.0076
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_Sr_Cl_MWRB78(T, P):
    """c-c'-a: sodium strontium chloride [MWRB78]."""
    psi = -0.0052
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Silvester and Pitzer (1978) ~~~~~
# Auto-generated functions
# General procedure:
#  - Inherit 298.15 K value from PM73;
#  - Add temperature derivative correction from SP78.
def bC_H_Cl_SP78(T, P):
    """ "c-a: hydrogen chloride [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_H_Cl_PM73(T, P)
    b0 = b0 + (T - 298.15) * -3.081e-4
    b1 = b1 + (T - 298.15) * 1.419e-4
    C0 = C0 + (T - 298.15) * 6.213e-5 / 2 * np.sqrt(np.abs(i2c["H"] * i2c["Cl"]))
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_H_Br_SP78(T, P):
    """ "c-a: hydrogen bromide [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_H_Br_PM73(T, P)
    b0 = b0 + (T - 298.15) * -2.049e-4
    b1 = b1 + (T - 298.15) * 4.467e-4
    C0 = C0 + (T - 298.15) * -5.685e-5 / 2 * np.sqrt(np.abs(i2c["H"] * i2c["Br"]))
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_H_I_SP78(T, P):
    """ "c-a: hydrogen iodide [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_H_I_PM73(T, P)
    b0 = b0 + (T - 298.15) * -0.23e-4
    b1 = b1 + (T - 298.15) * 8.86e-4
    C0 = C0 + (T - 298.15) * -7.32e-5 / 2 * np.sqrt(np.abs(i2c["H"] * i2c["I"]))
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_H_ClO4_SP78(T, P):
    """ "c-a: hydrogen perchlorate [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_H_ClO4_PM73(T, P)
    b0 = b0 + (T - 298.15) * 4.905e-4
    b1 = b1 + (T - 298.15) * 19.31e-4
    C0 = C0 + (T - 298.15) * -11.77e-5 / 2 * np.sqrt(np.abs(i2c["H"] * i2c["ClO4"]))
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Li_Cl_SP78(T, P):
    """ "c-a: lithium chloride [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Li_Cl_PM73(T, P)
    b0 = b0 + (T - 298.15) * -1.685e-4
    b1 = b1 + (T - 298.15) * 5.366e-4
    C0 = C0 + (T - 298.15) * -4.52e-5 / 2 * np.sqrt(np.abs(i2c["Li"] * i2c["Cl"]))
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Li_Br_SP78(T, P):
    """ "c-a: lithium bromide [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Li_Br_PM73(T, P)
    b0 = b0 + (T - 298.15) * -1.819e-4
    b1 = b1 + (T - 298.15) * 6.636e-4
    C0 = C0 + (T - 298.15) * -2.813e-5 / 2 * np.sqrt(np.abs(i2c["Li"] * i2c["Br"]))
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Li_ClO4_SP78(T, P):
    """ "c-a: lithium perchlorate [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Li_ClO4_PM73(T, P)
    b0 = b0 + (T - 298.15) * 0.386e-4
    b1 = b1 + (T - 298.15) * 7.009e-4
    C0 = C0 + (T - 298.15) * -7.712e-5 / 2 * np.sqrt(np.abs(i2c["Li"] * i2c["ClO4"]))
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_F_SP78(T, P):
    """ "c-a: sodium fluoride [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Na_F_PM73(T, P)
    b0 = b0 + (T - 298.15) * 5.361e-4
    b1 = b1 + (T - 298.15) * 8.7e-4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_Cl_SP78(T, P):
    """ "c-a: sodium chloride [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Na_Cl_PM73(T, P)
    b0 = b0 + (T - 298.15) * 7.159e-4
    b1 = b1 + (T - 298.15) * 7.005e-4
    C0 = C0 + (T - 298.15) * -10.54e-5 / 2 * np.sqrt(np.abs(i2c["Na"] * i2c["Cl"]))
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_Br_SP78(T, P):
    """ "c-a: sodium bromide [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Na_Br_PM73(T, P)
    b0 = b0 + (T - 298.15) * 7.692e-4
    b1 = b1 + (T - 298.15) * 10.79e-4
    C0 = C0 + (T - 298.15) * -9.3e-5 / 2 * np.sqrt(np.abs(i2c["Na"] * i2c["Br"]))
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_I_SP78(T, P):
    """ "c-a: sodium iodide [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Na_I_PM73(T, P)
    b0 = b0 + (T - 298.15) * 8.355e-4
    b1 = b1 + (T - 298.15) * 8.28e-4
    C0 = C0 + (T - 298.15) * -8.35e-5 / 2 * np.sqrt(np.abs(i2c["Na"] * i2c["I"]))
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_OH_SP78(T, P):
    """ "c-a: sodium hydroxide [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Na_OH_PM73(T, P)
    b0 = b0 + (T - 298.15) * 7.0e-4
    b1 = b1 + (T - 298.15) * 1.34e-4
    C0 = C0 + (T - 298.15) * -18.94e-5 / 2 * np.sqrt(np.abs(i2c["Na"] * i2c["OH"]))
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_ClO3_SP78(T, P):
    """ "c-a: sodium chlorate [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Na_ClO3_PM73(T, P)
    b0 = b0 + (T - 298.15) * 10.35e-4
    b1 = b1 + (T - 298.15) * 19.07e-4
    C0 = C0 + (T - 298.15) * -9.29e-5 / 2 * np.sqrt(np.abs(i2c["Na"] * i2c["ClO3"]))
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_ClO4_SP78(T, P):
    """ "c-a: sodium perchlorate [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Na_ClO4_PM73(T, P)
    b0 = b0 + (T - 298.15) * 12.96e-4
    b1 = b1 + (T - 298.15) * 22.97e-4
    C0 = C0 + (T - 298.15) * -16.23e-5 / 2 * np.sqrt(np.abs(i2c["Na"] * i2c["ClO4"]))
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_BrO3_SP78(T, P):
    """ "c-a: sodium bromate [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Na_BrO3_PM73(T, P)
    b0 = b0 + (T - 298.15) * 5.59e-4
    b1 = b1 + (T - 298.15) * 34.37e-4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


# bC_Na_IO3_SP78: no corresponding PM73 function


def bC_Na_SCN_SP78(T, P):
    """ "c-a: sodium thiocyanate [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Na_SCN_PM73(T, P)
    b0 = b0 + (T - 298.15) * 7.8e-4
    b1 = b1 + (T - 298.15) * 20.0e-4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_NO3_SP78(T, P):
    """ "c-a: sodium nitrate [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Na_NO3_PM73(T, P)
    b0 = b0 + (T - 298.15) * 12.66e-4
    b1 = b1 + (T - 298.15) * 20.6e-4
    C0 = C0 + (T - 298.15) * -23.16e-5 / 2 * np.sqrt(np.abs(i2c["Na"] * i2c["NO3"]))
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_F_SP78(T, P):
    """ "c-a: potassium fluoride [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_K_F_PM73(T, P)
    b0 = b0 + (T - 298.15) * 2.14e-4
    b1 = b1 + (T - 298.15) * 5.44e-4
    C0 = C0 + (T - 298.15) * -5.95e-5 / 2 * np.sqrt(np.abs(i2c["K"] * i2c["F"]))
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_Cl_SP78(T, P):
    """ "c-a: potassium chloride [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_K_Cl_PM73(T, P)
    b0 = b0 + (T - 298.15) * 5.794e-4
    b1 = b1 + (T - 298.15) * 10.71e-4
    C0 = C0 + (T - 298.15) * -5.095e-5 / 2 * np.sqrt(np.abs(i2c["K"] * i2c["Cl"]))
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_Br_SP78(T, P):
    """ "c-a: potassium bromide [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_K_Br_PM73(T, P)
    b0 = b0 + (T - 298.15) * 7.39e-4
    b1 = b1 + (T - 298.15) * 17.4e-4
    C0 = C0 + (T - 298.15) * -7.004e-5 / 2 * np.sqrt(np.abs(i2c["K"] * i2c["Br"]))
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_I_SP78(T, P):
    """ "c-a: potassium iodide [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_K_I_PM73(T, P)
    b0 = b0 + (T - 298.15) * 9.914e-4
    b1 = b1 + (T - 298.15) * 11.86e-4
    C0 = C0 + (T - 298.15) * -9.44e-5 / 2 * np.sqrt(np.abs(i2c["K"] * i2c["I"]))
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_ClO3_SP78(T, P):
    """ "c-a: potassium chlorate [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_K_ClO3_PM73(T, P)
    b0 = b0 + (T - 298.15) * 19.87e-4
    b1 = b1 + (T - 298.15) * 31.8e-4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


# bC_K_ClO4_SP78: no corresponding PM73 function


def bC_K_SCN_SP78(T, P):
    """ "c-a: potassium thiocyanate [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_K_SCN_PM73(T, P)
    b0 = b0 + (T - 298.15) * 6.87e-4
    b1 = b1 + (T - 298.15) * 37.0e-4
    C0 = C0 + (T - 298.15) * 0.43e-5 / 2 * np.sqrt(np.abs(i2c["K"] * i2c["SCN"]))
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_NO3_SP78(T, P):
    """ "c-a: potassium nitrate [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_K_NO3_PM73(T, P)
    b0 = b0 + (T - 298.15) * 2.06e-4
    b1 = b1 + (T - 298.15) * 64.5e-4
    C0 = C0 + (T - 298.15) * 39.7e-5 / 2 * np.sqrt(np.abs(i2c["K"] * i2c["NO3"]))
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_H2PO4_SP78(T, P):
    """ "c-a: potassium dihydrogen-phosphate [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_K_H2PO4_PM73(T, P)
    b0 = b0 + (T - 298.15) * 6.045e-4
    b1 = b1 + (T - 298.15) * 28.6e-4
    C0 = C0 + (T - 298.15) * -10.11e-5 / 2 * np.sqrt(np.abs(i2c["K"] * i2c["H2PO4"]))
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Rb_F_SP78(T, P):
    """ "c-a: rubidium fluoride [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Rb_F_PM73(T, P)
    b0 = b0 + (T - 298.15) * -0.76e-4
    b1 = b1 + (T - 298.15) * 14.7e-4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Rb_Cl_SP78(T, P):
    """ "c-a: rubidium chloride [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Rb_Cl_PM73(T, P)
    b0 = b0 + (T - 298.15) * 5.522e-4
    b1 = b1 + (T - 298.15) * 15.06e-4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Rb_Br_SP78(T, P):
    """ "c-a: rubidium bromide [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Rb_Br_PM73(T, P)
    b0 = b0 + (T - 298.15) * 6.78e-4
    b1 = b1 + (T - 298.15) * 20.35e-4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Rb_I_SP78(T, P):
    """ "c-a: rubidium iodide [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Rb_I_PM73(T, P)
    b0 = b0 + (T - 298.15) * 8.578e-4
    b1 = b1 + (T - 298.15) * 23.83e-4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Cs_F_SP78(T, P):
    """ "c-a: caesium fluoride [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Cs_F_PM73(T, P)
    b0 = b0 + (T - 298.15) * 0.95e-4
    b1 = b1 + (T - 298.15) * 5.97e-4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Cs_Cl_SP78(T, P):
    """ "c-a: caesium chloride [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Cs_Cl_PM73(T, P)
    b0 = b0 + (T - 298.15) * 8.28e-4
    b1 = b1 + (T - 298.15) * 15.0e-4
    C0 = C0 + (T - 298.15) * -12.25e-5 / 2 * np.sqrt(np.abs(i2c["Cs"] * i2c["Cl"]))
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Cs_Br_SP78(T, P):
    """ "c-a: caesium bromide [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Cs_Br_PM73(T, P)
    b0 = b0 + (T - 298.15) * 7.8e-4
    b1 = b1 + (T - 298.15) * 28.44e-4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Cs_I_SP78(T, P):
    """ "c-a: caesium iodide [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Cs_I_PM73(T, P)
    b0 = b0 + (T - 298.15) * 9.75e-4
    b1 = b1 + (T - 298.15) * 34.77e-4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_NH4_Cl_SP78(T, P):
    """ "c-a: ammonium chloride [SP78]."""
    # Coefficients from SP78 Table I
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_NH4_Cl_PM73(T, P)
    b0 = b0 + (T - 298.15) * 0.779e-4
    b1 = b1 + (T - 298.15) * 12.58e-4
    C0 = C0 + (T - 298.15) * 2.1e-5 / 2 * np.sqrt(np.abs(i2c["NH4"] * i2c["Cl"]))
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


# bC_NH4_H2PO4_SP78: no corresponding PM73 function
# bC_Me4N_F_SP78: no corresponding PM73 function
# bC_Et4N_F_SP78: no corresponding PM73 function
# bC_Pr4N_F_SP78: no corresponding PM73 function
# bC_Bu4N_F_SP78: no corresponding PM73 function
# bC_MeH3N_Cl_SP78: no corresponding PM73 function
# bC_Me2H2N_Cl_SP78: no corresponding PM73 function
# bC_Me3HN_Cl_SP78: no corresponding PM73 function
# bC_Me4N_Cl_SP78: no corresponding PM73 function
# bC_Et4N_Cl_SP78: no corresponding PM73 function
# bC_Pr4N_Cl_SP78: no corresponding PM73 function
# bC_Bu4N_Cl_SP78: no corresponding PM73 function
# bC_Me4N_Br_SP78: no corresponding PM73 function
# bC_Et4N_Br_SP78: no corresponding PM73 function
# bC_Pr4N_Br_SP78: no corresponding PM73 function
# bC_Bu4N_Br_SP78: no corresponding PM73 function
# bC_Me4N_I_SP78: no corresponding PM73 function
# bC_Et4N_I_SP78: no corresponding PM73 function
# bC_Pr4N_I_SP78: no corresponding PM73 function


def bC_Mg_Cl_SP78(T, P):
    """ "c-a: magnesium chloride [SP78]."""
    # Coefficients from SP78 Table II
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Mg_Cl_PM73(T, P)
    b0 = b0 + (T - 298.15) * -0.259e-3 * 3 / 4
    b1 = b1 + (T - 298.15) * 3.7e-3 * 3 / 4
    C0 = C0 + (T - 298.15) * -3.11e-4 * 3 / 2 ** (5 / 2) / 2 * np.sqrt(
        np.abs(i2c["Mg"] * i2c["Cl"])
    )
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mg_Br_SP78(T, P):
    """ "c-a: magnesium bromide [SP78]."""
    # Coefficients from SP78 Table II
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Mg_Br_PM73(T, P)
    b0 = b0 + (T - 298.15) * -0.075e-3 * 3 / 4
    b1 = b1 + (T - 298.15) * 5.15e-3 * 3 / 4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mg_ClO4_SP78(T, P):
    """ "c-a: magnesium perchlorate [SP78]."""
    # Coefficients from SP78 Table II
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Mg_ClO4_PM73(T, P)
    b0 = b0 + (T - 298.15) * 0.697e-3 * 3 / 4
    b1 = b1 + (T - 298.15) * 6.0e-3 * 3 / 4
    C0 = C0 + (T - 298.15) * -6.65e-4 * 3 / 2 ** (5 / 2) / 2 * np.sqrt(
        np.abs(i2c["Mg"] * i2c["ClO4"])
    )
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mg_NO3_SP78(T, P):
    """ "c-a: magnesium nitrate [SP78]."""
    # Coefficients from SP78 Table II
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Mg_NO3_PM73(T, P)
    b0 = b0 + (T - 298.15) * 0.687e-3 * 3 / 4
    b1 = b1 + (T - 298.15) * 5.99e-3 * 3 / 4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ca_Cl_SP78(T, P):
    """ "c-a: calcium chloride [SP78]."""
    # Coefficients from SP78 Table II
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Ca_Cl_PM73(T, P)
    b0 = b0 + (T - 298.15) * -0.23e-3 * 3 / 4
    b1 = b1 + (T - 298.15) * 5.2e-3 * 3 / 4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ca_Br_SP78(T, P):
    """ "c-a: calcium bromide [SP78]."""
    # Coefficients from SP78 Table II
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Ca_Br_PM73(T, P)
    b0 = b0 + (T - 298.15) * -0.697e-3 * 3 / 4
    b1 = b1 + (T - 298.15) * 8.05e-3 * 3 / 4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ca_NO3_SP78(T, P):
    """ "c-a: calcium nitrate [SP78]."""
    # Coefficients from SP78 Table II
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Ca_NO3_PM73(T, P)
    b0 = b0 + (T - 298.15) * 0.706e-3 * 3 / 4
    b1 = b1 + (T - 298.15) * 12.25e-3 * 3 / 4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ca_ClO4_SP78(T, P):
    """ "c-a: calcium perchlorate [SP78]."""
    # Coefficients from SP78 Table II
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Ca_ClO4_PM73(T, P)
    b0 = b0 + (T - 298.15) * 1.106e-3 * 3 / 4
    b1 = b1 + (T - 298.15) * 6.77e-3 * 3 / 4
    C0 = C0 + (T - 298.15) * -5.83e-4 * 3 / 2 ** (5 / 2) / 2 * np.sqrt(
        np.abs(i2c["Ca"] * i2c["ClO4"])
    )
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Sr_Cl_SP78(T, P):
    """ "c-a: strontium chloride [SP78]."""
    # Coefficients from SP78 Table II
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Sr_Cl_PM73(T, P)
    b0 = b0 + (T - 298.15) * 0.956e-3 * 3 / 4
    b1 = b1 + (T - 298.15) * 3.79e-3 * 3 / 4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Sr_Br_SP78(T, P):
    """ "c-a: strontium bromide [SP78]."""
    # Coefficients from SP78 Table II
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Sr_Br_PM73(T, P)
    b0 = b0 + (T - 298.15) * -0.437e-3 * 3 / 4
    b1 = b1 + (T - 298.15) * 8.71e-3 * 3 / 4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Sr_NO3_SP78(T, P):
    """ "c-a: strontium nitrate [SP78]."""
    # Coefficients from SP78 Table II
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Sr_NO3_PM73(T, P)
    b0 = b0 + (T - 298.15) * 0.236e-3 * 3 / 4
    b1 = b1 + (T - 298.15) * 16.63e-3 * 3 / 4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Sr_ClO4_SP78(T, P):
    """ "c-a: strontium perchlorate [SP78]."""
    # Coefficients from SP78 Table II
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Sr_ClO4_PM73(T, P)
    b0 = b0 + (T - 298.15) * 1.524e-3 * 3 / 4
    b1 = b1 + (T - 298.15) * 7.19e-3 * 3 / 4
    C0 = C0 + (T - 298.15) * -5.86e-4 * 3 / 2 ** (5 / 2) / 2 * np.sqrt(
        np.abs(i2c["Sr"] * i2c["ClO4"])
    )
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ba_Cl_SP78(T, P):
    """ "c-a: barium chloride [SP78]."""
    # Coefficients from SP78 Table II
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Ba_Cl_PM73(T, P)
    b0 = b0 + (T - 298.15) * 0.854e-3 * 3 / 4
    b1 = b1 + (T - 298.15) * 4.31e-3 * 3 / 4
    C0 = C0 + (T - 298.15) * -2.9e-4 * 3 / 2 ** (5 / 2) / 2 * np.sqrt(
        np.abs(i2c["Ba"] * i2c["Cl"])
    )
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ba_Br_SP78(T, P):
    """ "c-a: barium bromide [SP78]."""
    # Coefficients from SP78 Table II
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Ba_Br_PM73(T, P)
    b0 = b0 + (T - 298.15) * -0.451e-3 * 3 / 4
    b1 = b1 + (T - 298.15) * 9.04e-3 * 3 / 4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ba_NO3_SP78(T, P):
    """ "c-a: barium nitrate [SP78]."""
    # Coefficients from SP78 Table II
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Ba_NO3_PM73(T, P)
    b0 = b0 + (T - 298.15) * -3.88e-3 * 3 / 4
    b1 = b1 + (T - 298.15) * 38.8e-3 * 3 / 4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


# bC_Mnjj_ClO4_SP78: no corresponding PM73 function
# bC_Cojj_ClO4_SP78: no corresponding PM73 function
# bC_Nijj_ClO4_SP78: no corresponding PM73 function


def bC_Cujj_Cl_SP78(T, P):
    """ "c-a: copper(II) chloride [SP78]."""
    # Coefficients from SP78 Table II
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Cujj_Cl_PM73(T, P)
    b0 = b0 + (T - 298.15) * -3.62e-3 * 3 / 4
    b1 = b1 + (T - 298.15) * 11.3e-3 * 3 / 4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Znjj_ClO4_SP78(T, P):
    """ "c-a: zinc(II) perchlorate [SP78]."""
    # Coefficients from SP78 Table II
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Znjj_ClO4_PM73(T, P)
    b0 = b0 + (T - 298.15) * 0.795e-3 * 3 / 4
    b1 = b1 + (T - 298.15) * 6.79e-3 * 3 / 4
    C0 = C0 + (T - 298.15) * -7.27e-4 * 3 / 2 ** (5 / 2) / 2 * np.sqrt(
        np.abs(i2c["Znjj"] * i2c["ClO4"])
    )
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Li_SO4_SP78(T, P):
    """ "c-a: lithium sulfate [SP78]."""
    # Coefficients from SP78 Table II
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Li_SO4_PM73(T, P)
    b0 = b0 + (T - 298.15) * 0.674e-3 * 3 / 4
    b1 = b1 + (T - 298.15) * 1.88e-3 * 3 / 4
    C0 = C0 + (T - 298.15) * -4.4e-4 * 3 / 2 ** (5 / 2) / 2 * np.sqrt(
        np.abs(i2c["Li"] * i2c["SO4"])
    )
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_SO4_SP78(T, P):
    """ "c-a: sodium sulfate [SP78]."""
    # Coefficients from SP78 Table II
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Na_SO4_PM73(T, P)
    b0 = b0 + (T - 298.15) * 3.156e-3 * 3 / 4
    b1 = b1 + (T - 298.15) * 7.51e-3 * 3 / 4
    C0 = C0 + (T - 298.15) * -9.2e-4 * 3 / 2 ** (5 / 2) / 2 * np.sqrt(
        np.abs(i2c["Na"] * i2c["SO4"])
    )
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_SO4_SP78(T, P):
    """ "c-a: potassium sulfate [SP78]."""
    # Coefficients from SP78 Table II
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_K_SO4_PM73(T, P)
    b0 = b0 + (T - 298.15) * 1.92e-3 * 3 / 4
    b1 = b1 + (T - 298.15) * 8.93e-3 * 3 / 4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Rb_SO4_SP78(T, P):
    """ "c-a: rubidium sulfate [SP78]."""
    # Coefficients from SP78 Table II
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Rb_SO4_PM73(T, P)
    b0 = b0 + (T - 298.15) * 1.25e-3 * 3 / 4
    b1 = b1 + (T - 298.15) * 11.52e-3 * 3 / 4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Cs_SO4_SP78(T, P):
    """ "c-a: caesium sulfate [SP78]."""
    # Coefficients from SP78 Table II
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Cs_SO4_PM73(T, P)
    b0 = b0 + (T - 298.15) * -1.19e-3 * 3 / 4
    b1 = b1 + (T - 298.15) * 19.31e-3 * 3 / 4
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_La_Cl_SP78(T, P):
    """ "c-a: lanthanum chloride [SP78]."""
    # Coefficients from SP78 Table III
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_La_Cl_PM73(T, P)
    b0 = b0 + (T - 298.15) * 0.253e-3
    b1 = b1 + (T - 298.15) * 0.798e-2
    C0 = C0 + (T - 298.15) * -0.371e-3 / 2 * np.sqrt(np.abs(i2c["La"] * i2c["Cl"]))
    # Validity range follows typical values assigned by MP98
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


# bC_La_ClO4_SP78: no corresponding PM73 function
# bC_La_NO3_SP78: no corresponding PM73 function
# bC_Na_FejjjCN6_SP78: no corresponding PM73 function
# bC_K_FejjjCN6_SP78: no corresponding PM73 function
# bC_K_FejjCN6_SP78: no corresponding PM73 function
# bC_Mg_SO4_SP78: no corresponding PM73 function
# bC_Ca_SO4_SP78: no corresponding PM73 function
# bC_Cujj_SO4_SP78: no corresponding PM73 function
# bC_Znjj_SO4_SP78: no corresponding PM73 function
# bC_Cdjj_SO4_SP78: no corresponding PM73 function

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Roy et al. (1980) ~~~~~
def theta_H_Mg_RGB80(T, P):
    """c-c': hydrogen magnesium [RGB80]."""
    # RGB80 do provide theta values at 5, 15, 25, 35 and 45 degC, but no
    #  equation to interpolate between them.
    # This function just returns the 25 degC value.
    theta = 0.0620
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_H_Mg_Cl_RGB80(T, P):
    """c-c': hydrogen magnesium chloride [RGB80]."""
    # RGB80 do provide theta values at 5, 15, 25, 35 and 45 degC, but no
    #  equation to interpolate between them.
    # This function just returns the 25 degC value.
    theta = 0.0010
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Rard and Miller (1981i) ~~~~~
def bC_Mg_SO4_RM81i(T, P):
    """c-a: magnesium sulfate [RM81i]."""
    b0 = 0.21499
    b1 = 3.3646
    b2 = -32.743
    Cphi = 0.02797
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Mg"] * i2c["SO4"])))
    C1 = 0
    alph1 = 1.4
    alph2 = 12
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Peiper and Pitzer (1982) ~~~~~
def PP82_eqMPH(T, q):
    """PP82 equation derived by MP Humphreys."""
    Tr = 298.15
    return q[0] + q[1] * (T - Tr) + q[2] * (T - Tr) ** 2 / 2


def bC_Na_CO3_PP82(T, P):
    """c-a: sodium carbonate [PP82]."""
    # I have no idea where MP98 got their T**2 parameters from
    #   or why they are so small.
    b0 = PP82_eqMPH(
        T,
        [
            0.0362,
            1.79e-3,
            -4.22e-5,
        ],
    )
    b1 = PP82_eqMPH(
        T,
        [
            1.51,
            2.05e-3,
            -16.8e-5,
        ],
    )
    b2 = 0
    Cphi = 0.0052
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["CO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_HCO3_PP82(T, P):
    """c-a: sodium bicarbonate [PP82]."""
    # I have no idea where MP98 got their T**2 parameters from
    #   or why they are so small.
    b0 = PP82_eqMPH(
        T,
        [
            0.028,
            1.00e-3,
            -2.6e-5,
        ],
    )
    b1 = PP82_eqMPH(
        T,
        [
            0.044,
            1.10e-3,
            -4.3e-5,
        ],
    )
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def theta_Cl_HCO3_PP82(T, P):
    """a-a': chloride bicarbonate [PP82]."""
    theta = 0.0359
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_CO3_Cl_PP82(T, P):
    """a-a': carbonate chloride [PP82]."""
    theta = -0.053
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_Cl_HCO3_PP82(T, P):
    """c-a-a': sodium chloride bicarbonate [PP82]."""
    psi = -0.0143
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Roy et al. (1981) ~~~~~
def theta_Ca_H_RGO81(T, P):
    """c-c': calcium hydrogen [RGO81]."""
    theta = 0.0612
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Ca_H_Cl_RGO81(T, P):
    """c-c': calcium hydrogen chloride [RGO81]."""
    theta = 0.0008
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Thurmond & Millero (1982) ~~~~~
def psi_Na_CO3_Cl_TM82(T, P):
    """c-a-a': sodium carbonate chloride [TM82]."""
    psi = 0.016
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ de Lima & Pitzer (1983) ~~~~~
def bC_Mg_Cl_dLP83(T, P):
    """c-a: magnesium chloride [dLP83]."""
    # dLP83 Eq. (11)
    b0 = 5.93915e-7 * T ** 2 - 9.31654e-4 * T + 0.576066
    b1 = 2.60169e-5 * T ** 2 - 1.09438e-2 * T + 2.60135
    b2 = 0
    Cphi = 3.01823e-7 * T ** 2 - 2.89125e-4 * T + 6.57867e-2
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Mg"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 298.15) & (T <= 523.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Holmes and Mesmer (1983) ~~~~~
def HM83_eq25(T, a):
    """HM83 equation 25."""
    TR = 298.15
    return (
        a[0]
        + a[1] * (1 / T - 1 / TR)
        + a[2] * np.log(T / TR)
        + a[3] * (T - TR)
        + a[4] * (T ** 2 - TR ** 2)
        + a[5] * np.log(T - 260)
    )


def bC_Cs_Cl_HM83(T, P):
    """c-a: caesium chloride [HM83]."""
    b0 = HM83_eq25(
        T,
        [
            0.03352,
            -1290.0,
            -8.4279,
            0.018502,
            -6.7942e-6,
            0,
        ],
    )
    b1 = HM83_eq25(
        T,
        [
            0.0429,
            -38.0,
            0,
            0.001306,
            0,
            0,
        ],
    )
    b2 = 0
    Cphi = HM83_eq25(
        T,
        [
            -2.62e-4,
            157.13,
            1.0860,
            -0.0025242,
            9.840e-7,
            0,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Cs"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 523.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_Cl_HM83(T, P):
    """c-a: potassium chloride [HM83]."""
    b0 = HM83_eq25(
        T,
        [
            0.04808,
            -758.48,
            -4.7062,
            0.010072,
            -3.7599e-6,
            0,
        ],
    )
    b1 = HM83_eq25(
        T,
        [
            0.0476,
            303.09,
            1.066,
            0,
            0,
            0.0470,
        ],
    )
    b2 = 0
    Cphi = HM83_eq25(
        T,
        [
            -7.88e-4,
            91.270,
            0.58643,
            -0.0012980,
            4.9567e-7,
            0,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 523.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Li_Cl_HM83(T, P):
    """c-a: lithium chloride [HM83]."""
    b0 = HM83_eq25(
        T,
        [
            0.14847,
            0,
            0,
            -1.546e-4,
            0,
            0,
        ],
    )
    b1 = HM83_eq25(
        T,
        [
            0.307,
            0,
            0,
            6.36e-4,
            0,
            0,
        ],
    )
    b2 = 0
    Cphi = HM83_eq25(
        T,
        [
            0.003710,
            4.115,
            0,
            0,
            -3.71e-9,
            0,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Li"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 523.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Millero (1983) ~~~~~
def theta_Cl_H2AsO4_M83(T, P):
    """a-a': chloride dihydrogen-arsenate [M83]."""
    # NOTE: this coefficient is for use only WITHOUT unsymmetrical mixing!
    theta = 0.228
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_Cl_HAsO4_M83(T, P):
    """a-a': chloride hydrogen-arsenate [M83]."""
    # NOTE: this coefficient is for use only WITHOUT unsymmetrical mixing!
    theta = 0.122
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_AsO4_Cl_M83(T, P):
    """a-a': arsenate chloride [M83]."""
    # NOTE: this coefficient is for use only WITHOUT unsymmetrical mixing!
    theta = 0.060
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_acetate_Cl_M83(T, P):
    """a-a': acetate chloride [M83]."""
    # NOTE: this coefficient is for use only WITHOUT unsymmetrical mixing!
    theta = -0.017
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_Cl_H2AsO4_M83(T, P):
    """c-a-a': sodium chloride dihydrogen-arsenate [M83]."""
    # NOTE: this coefficient is for use only WITHOUT unsymmetrical mixing!
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Na_Cl_HAsO4_M83(T, P):
    """c-a-a': sodium chloride hydrogen-arsenate [M83]."""
    # NOTE: this coefficient is for use only WITHOUT unsymmetrical mixing!
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Na_AsO4_Cl_M83(T, P):
    """c-a-a': sodium arsenate chloride [M83]."""
    # NOTE: this coefficient is for use only WITHOUT unsymmetrical mixing!
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_Na_acetate_Cl_M83(T, P):
    """c-a-a': sodium acetate chloride [M83]."""
    # NOTE: this coefficient is for use only WITHOUT unsymmetrical mixing!
    theta = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Roy et al. (1983) ~~~~~
def bC_K_HCO3_RGWW83(T, P):
    """c-a: potassium bicarbonate [RGWW83]."""
    b0 = -0.022 + 0.996e-3 * (T - 298.15)  # +/- 0.014 on constant term
    b1 = 0.09 + 1.104e-3 * (T - 298.15)  # +/- 0.04 on constant term
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    # Validity range declared by MP98, but they have different equations!
    valid = (T >= 278.15) & (T <= 318.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Harvie et al. (1984) ~~~~~
# c-a functions auto-generated by HMW84_funcgen_bC.py
# c-a-a' functions auto-generated by HMW84_funcgen_caa.py
# c-c'-a functions auto-generated by HMW84_funcgen_cca.py
# n-x functions written out manually
def bC_Na_Cl_HMW84(T, P):
    """c-a: sodium chloride [HMW84]."""
    b0 = 0.0765
    b1 = 0.2644
    b2 = 0.0
    Cphi = 0.00127
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_SO4_HMW84(T, P):
    """c-a: sodium sulfate [HMW84]."""
    b0 = 0.01958
    b1 = 1.113
    b2 = 0.0
    Cphi = 0.00497
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["SO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_HSO4_HMW84(T, P):
    """c-a: sodium bisulfate [HMW84]."""
    b0 = 0.0454
    b1 = 0.398
    b2 = 0.0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_OH_HMW84(T, P):
    """c-a: sodium hydroxide [HMW84]."""
    b0 = 0.0864
    b1 = 0.253
    b2 = 0.0
    Cphi = 0.0044
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["OH"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_HCO3_HMW84(T, P):
    """c-a: sodium bicarbonate [HMW84]."""
    b0 = 0.0277
    b1 = 0.0411
    b2 = 0.0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_CO3_HMW84(T, P):
    """c-a: sodium carbonate [HMW84]."""
    b0 = 0.0399
    b1 = 1.389
    b2 = 0.0
    Cphi = 0.0044
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["CO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_Cl_HMW84(T, P):
    """c-a: potassium chloride [HMW84]."""
    b0 = 0.04835
    b1 = 0.2122
    b2 = 0.0
    Cphi = -0.00084
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_SO4_HMW84(T, P):
    """c-a: potassium sulfate [HMW84]."""
    b0 = 0.04995
    b1 = 0.7793
    b2 = 0.0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_HSO4_HMW84(T, P):
    """c-a: potassium bisulfate [HMW84]."""
    b0 = -0.0003
    b1 = 0.1735
    b2 = 0.0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_OH_HMW84(T, P):
    """c-a: potassium hydroxide [HMW84]."""
    b0 = 0.1298
    b1 = 0.32
    b2 = 0.0
    Cphi = 0.0041
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["OH"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_HCO3_HMW84(T, P):
    """c-a: potassium bicarbonate [HMW84]."""
    b0 = 0.0296
    b1 = -0.013
    b2 = 0.0
    Cphi = -0.008
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["HCO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_CO3_HMW84(T, P):
    """c-a: potassium carbonate [HMW84]."""
    b0 = 0.1488
    b1 = 1.43
    b2 = 0.0
    Cphi = -0.0015
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["CO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ca_Cl_HMW84(T, P):
    """c-a: calcium chloride [HMW84]."""
    b0 = 0.3159
    b1 = 1.614
    b2 = 0.0
    Cphi = -0.00034
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Ca"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ca_SO4_HMW84(T, P):
    """c-a: calcium sulfate [HMW84]."""
    b0 = 0.2
    b1 = 3.1973
    b2 = -54.24
    C0 = 0
    C1 = 0
    alph1 = 1.4
    alph2 = 12
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ca_HSO4_HMW84(T, P):
    """c-a: calcium bisulfate [HMW84]."""
    b0 = 0.2145
    b1 = 2.53
    b2 = 0.0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ca_OH_HMW84(T, P):
    """c-a: calcium hydroxide [HMW84]."""
    b0 = -0.1747
    b1 = -0.2303
    b2 = -5.72
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = 12
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ca_HCO3_HMW84(T, P):
    """c-a: calcium bicarbonate [HMW84]."""
    b0 = 0.4
    b1 = 2.977
    b2 = 0.0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ca_CO3_HMW84(T, P):
    """c-a: calcium carbonate [HMW84]."""
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    C0 = 0
    C1 = 0
    alph1 = 1.4
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mg_Cl_HMW84(T, P):
    """c-a: magnesium chloride [HMW84]."""
    b0 = 0.35235
    b1 = 1.6815
    b2 = 0.0
    Cphi = 0.00519
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Mg"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mg_SO4_HMW84(T, P):
    """c-a: magnesium sulfate [HMW84]."""
    b0 = 0.221
    b1 = 3.343
    b2 = -37.23
    Cphi = 0.025
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Mg"] * i2c["SO4"])))
    C1 = 0
    alph1 = 1.4
    alph2 = 12
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mg_HSO4_HMW84(T, P):
    """c-a: magnesium bisulfate [HMW84]."""
    b0 = 0.4746
    b1 = 1.729
    b2 = 0.0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mg_OH_HMW84(T, P):
    """c-a: magnesium hydroxide [HMW84]."""
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mg_HCO3_HMW84(T, P):
    """c-a: magnesium bicarbonate [HMW84]."""
    b0 = 0.329
    b1 = 0.6072
    b2 = 0.0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mg_CO3_HMW84(T, P):
    """c-a: magnesium carbonate [HMW84]."""
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    C0 = 0
    C1 = 0
    alph1 = 1.4
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_MgOH_Cl_HMW84(T, P):
    """c-a: magnesium-hydroxide chloride [HMW84]."""
    b0 = -0.1
    b1 = 1.658
    b2 = 0.0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_MgOH_SO4_HMW84(T, P):
    """c-a: magnesium-hydroxide sulfate [HMW84]."""
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_MgOH_HSO4_HMW84(T, P):
    """c-a: magnesium-hydroxide bisulfate [HMW84]."""
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_MgOH_OH_HMW84(T, P):
    """c-a: magnesium-hydroxide hydroxide [HMW84]."""
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_MgOH_HCO3_HMW84(T, P):
    """c-a: magnesium-hydroxide bicarbonate [HMW84]."""
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_MgOH_CO3_HMW84(T, P):
    """c-a: magnesium-hydroxide carbonate [HMW84]."""
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_H_Cl_HMW84(T, P):
    """c-a: hydrogen chloride [HMW84]."""
    b0 = 0.1775
    b1 = 0.2945
    b2 = 0.0
    Cphi = 0.0008
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["H"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_H_SO4_HMW84(T, P):
    """c-a: hydrogen sulfate [HMW84]."""
    b0 = 0.0298
    b1 = 0.0
    b2 = 0.0
    Cphi = 0.0438
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["H"] * i2c["SO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_H_HSO4_HMW84(T, P):
    """c-a: hydrogen bisulfate [HMW84]."""
    b0 = 0.2065
    b1 = 0.5556
    b2 = 0.0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_H_OH_HMW84(T, P):
    """c-a: hydrogen hydroxide [HMW84]."""
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_H_HCO3_HMW84(T, P):
    """c-a: hydrogen bicarbonate [HMW84]."""
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_H_CO3_HMW84(T, P):
    """c-a: hydrogen carbonate [HMW84]."""
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def theta_Cl_SO4_HMW84(T, P):
    """a-a': chloride sulfate [HMW84]."""
    theta = 0.02
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_Cl_SO4_HMW84(T, P):
    """c-a-a': sodium chloride sulfate [HMW84]."""
    psi = 0.0014
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Cl_SO4_HMW84(T, P):
    """c-a-a': potassium chloride sulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_Cl_SO4_HMW84(T, P):
    """c-a-a': calcium chloride sulfate [HMW84]."""
    psi = -0.018
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_Cl_SO4_HMW84(T, P):
    """c-a-a': magnesium chloride sulfate [HMW84]."""
    psi = -0.004
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_MgOH_Cl_SO4_HMW84(T, P):
    """c-a-a': magnesium-hydroxide chloride sulfate [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Cl_SO4_HMW84(T, P):
    """c-a-a': hydrogen chloride sulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_Cl_HSO4_HMW84(T, P):
    """a-a': chloride bisulfate [HMW84]."""
    theta = -0.006
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_Cl_HSO4_HMW84(T, P):
    """c-a-a': sodium chloride bisulfate [HMW84]."""
    psi = -0.006
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Cl_HSO4_HMW84(T, P):
    """c-a-a': potassium chloride bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_Cl_HSO4_HMW84(T, P):
    """c-a-a': calcium chloride bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_Cl_HSO4_HMW84(T, P):
    """c-a-a': magnesium chloride bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_MgOH_Cl_HSO4_HMW84(T, P):
    """c-a-a': magnesium-hydroxide chloride bisulfate [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Cl_HSO4_HMW84(T, P):
    """c-a-a': hydrogen chloride bisulfate [HMW84]."""
    psi = 0.013
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_Cl_OH_HMW84(T, P):
    """a-a': chloride hydroxide [HMW84]."""
    theta = -0.05
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_Cl_OH_HMW84(T, P):
    """c-a-a': sodium chloride hydroxide [HMW84]."""
    psi = -0.006
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Cl_OH_HMW84(T, P):
    """c-a-a': potassium chloride hydroxide [HMW84]."""
    psi = -0.006
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_Cl_OH_HMW84(T, P):
    """c-a-a': calcium chloride hydroxide [HMW84]."""
    psi = -0.025
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_Cl_OH_HMW84(T, P):
    """c-a-a': magnesium chloride hydroxide [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_MgOH_Cl_OH_HMW84(T, P):
    """c-a-a': magnesium-hydroxide chloride hydroxide [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Cl_OH_HMW84(T, P):
    """c-a-a': hydrogen chloride hydroxide [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_Cl_HCO3_HMW84(T, P):
    """a-a': chloride bicarbonate [HMW84]."""
    theta = 0.03
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_Cl_HCO3_HMW84(T, P):
    """c-a-a': sodium chloride bicarbonate [HMW84]."""
    psi = -0.15
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Cl_HCO3_HMW84(T, P):
    """c-a-a': potassium chloride bicarbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_Cl_HCO3_HMW84(T, P):
    """c-a-a': calcium chloride bicarbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_Cl_HCO3_HMW84(T, P):
    """c-a-a': magnesium chloride bicarbonate [HMW84]."""
    psi = -0.096
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_MgOH_Cl_HCO3_HMW84(T, P):
    """c-a-a': magnesium-hydroxide chloride bicarbonate [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Cl_HCO3_HMW84(T, P):
    """c-a-a': hydrogen chloride bicarbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_CO3_Cl_HMW84(T, P):
    """a-a': carbonate chloride [HMW84]."""
    theta = -0.02
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_CO3_Cl_HMW84(T, P):
    """c-a-a': sodium carbonate chloride [HMW84]."""
    psi = 0.0085
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_CO3_Cl_HMW84(T, P):
    """c-a-a': potassium carbonate chloride [HMW84]."""
    psi = 0.004
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_CO3_Cl_HMW84(T, P):
    """c-a-a': calcium carbonate chloride [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_CO3_Cl_HMW84(T, P):
    """c-a-a': magnesium carbonate chloride [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_MgOH_CO3_Cl_HMW84(T, P):
    """c-a-a': magnesium-hydroxide carbonate chloride [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_CO3_Cl_HMW84(T, P):
    """c-a-a': hydrogen carbonate chloride [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_HSO4_SO4_HMW84(T, P):
    """a-a': bisulfate sulfate [HMW84]."""
    theta = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_HSO4_SO4_HMW84(T, P):
    """c-a-a': sodium bisulfate sulfate [HMW84]."""
    psi = -0.0094
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_HSO4_SO4_HMW84(T, P):
    """c-a-a': potassium bisulfate sulfate [HMW84]."""
    psi = -0.0677
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_HSO4_SO4_HMW84(T, P):
    """c-a-a': calcium bisulfate sulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_HSO4_SO4_HMW84(T, P):
    """c-a-a': magnesium bisulfate sulfate [HMW84]."""
    psi = -0.0425
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_MgOH_HSO4_SO4_HMW84(T, P):
    """c-a-a': magnesium-hydroxide bisulfate sulfate [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_HSO4_SO4_HMW84(T, P):
    """c-a-a': hydrogen bisulfate sulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_OH_SO4_HMW84(T, P):
    """a-a': hydroxide sulfate [HMW84]."""
    theta = -0.013
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_OH_SO4_HMW84(T, P):
    """c-a-a': sodium hydroxide sulfate [HMW84]."""
    psi = -0.009
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_OH_SO4_HMW84(T, P):
    """c-a-a': potassium hydroxide sulfate [HMW84]."""
    psi = -0.05
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_OH_SO4_HMW84(T, P):
    """c-a-a': calcium hydroxide sulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_OH_SO4_HMW84(T, P):
    """c-a-a': magnesium hydroxide sulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_MgOH_OH_SO4_HMW84(T, P):
    """c-a-a': magnesium-hydroxide hydroxide sulfate [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_OH_SO4_HMW84(T, P):
    """c-a-a': hydrogen hydroxide sulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_HCO3_SO4_HMW84(T, P):
    """a-a': bicarbonate sulfate [HMW84]."""
    theta = 0.01
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_HCO3_SO4_HMW84(T, P):
    """c-a-a': sodium bicarbonate sulfate [HMW84]."""
    psi = -0.005
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_HCO3_SO4_HMW84(T, P):
    """c-a-a': potassium bicarbonate sulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_HCO3_SO4_HMW84(T, P):
    """c-a-a': calcium bicarbonate sulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_HCO3_SO4_HMW84(T, P):
    """c-a-a': magnesium bicarbonate sulfate [HMW84]."""
    psi = -0.161
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_MgOH_HCO3_SO4_HMW84(T, P):
    """c-a-a': magnesium-hydroxide bicarbonate sulfate [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_HCO3_SO4_HMW84(T, P):
    """c-a-a': hydrogen bicarbonate sulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_CO3_SO4_HMW84(T, P):
    """a-a': carbonate sulfate [HMW84]."""
    theta = 0.02
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_CO3_SO4_HMW84(T, P):
    """c-a-a': sodium carbonate sulfate [HMW84]."""
    psi = -0.005
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_CO3_SO4_HMW84(T, P):
    """c-a-a': potassium carbonate sulfate [HMW84]."""
    psi = -0.009
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_CO3_SO4_HMW84(T, P):
    """c-a-a': calcium carbonate sulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_CO3_SO4_HMW84(T, P):
    """c-a-a': magnesium carbonate sulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_MgOH_CO3_SO4_HMW84(T, P):
    """c-a-a': magnesium-hydroxide carbonate sulfate [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_CO3_SO4_HMW84(T, P):
    """c-a-a': hydrogen carbonate sulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_HSO4_OH_HMW84(T, P):
    """a-a': bisulfate hydroxide [HMW84]."""
    theta = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_HSO4_OH_HMW84(T, P):
    """c-a-a': sodium bisulfate hydroxide [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_HSO4_OH_HMW84(T, P):
    """c-a-a': potassium bisulfate hydroxide [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_HSO4_OH_HMW84(T, P):
    """c-a-a': calcium bisulfate hydroxide [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_HSO4_OH_HMW84(T, P):
    """c-a-a': magnesium bisulfate hydroxide [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_MgOH_HSO4_OH_HMW84(T, P):
    """c-a-a': magnesium-hydroxide bisulfate hydroxide [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_HSO4_OH_HMW84(T, P):
    """c-a-a': hydrogen bisulfate hydroxide [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_HCO3_HSO4_HMW84(T, P):
    """a-a': bicarbonate bisulfate [HMW84]."""
    theta = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_HCO3_HSO4_HMW84(T, P):
    """c-a-a': sodium bicarbonate bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_HCO3_HSO4_HMW84(T, P):
    """c-a-a': potassium bicarbonate bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_HCO3_HSO4_HMW84(T, P):
    """c-a-a': calcium bicarbonate bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_HCO3_HSO4_HMW84(T, P):
    """c-a-a': magnesium bicarbonate bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_MgOH_HCO3_HSO4_HMW84(T, P):
    """c-a-a': magnesium-hydroxide bicarbonate bisulfate [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_HCO3_HSO4_HMW84(T, P):
    """c-a-a': hydrogen bicarbonate bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_CO3_HSO4_HMW84(T, P):
    """a-a': carbonate bisulfate [HMW84]."""
    theta = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_CO3_HSO4_HMW84(T, P):
    """c-a-a': sodium carbonate bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_CO3_HSO4_HMW84(T, P):
    """c-a-a': potassium carbonate bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_CO3_HSO4_HMW84(T, P):
    """c-a-a': calcium carbonate bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_CO3_HSO4_HMW84(T, P):
    """c-a-a': magnesium carbonate bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_MgOH_CO3_HSO4_HMW84(T, P):
    """c-a-a': magnesium-hydroxide carbonate bisulfate [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_CO3_HSO4_HMW84(T, P):
    """c-a-a': hydrogen carbonate bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_HCO3_OH_HMW84(T, P):
    """a-a': bicarbonate hydroxide [HMW84]."""
    theta = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_HCO3_OH_HMW84(T, P):
    """c-a-a': sodium bicarbonate hydroxide [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_HCO3_OH_HMW84(T, P):
    """c-a-a': potassium bicarbonate hydroxide [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_HCO3_OH_HMW84(T, P):
    """c-a-a': calcium bicarbonate hydroxide [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_HCO3_OH_HMW84(T, P):
    """c-a-a': magnesium bicarbonate hydroxide [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_MgOH_HCO3_OH_HMW84(T, P):
    """c-a-a': magnesium-hydroxide bicarbonate hydroxide [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_HCO3_OH_HMW84(T, P):
    """c-a-a': hydrogen bicarbonate hydroxide [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_CO3_OH_HMW84(T, P):
    """a-a': carbonate hydroxide [HMW84]."""
    theta = 0.1
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_CO3_OH_HMW84(T, P):
    """c-a-a': sodium carbonate hydroxide [HMW84]."""
    psi = -0.017
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_CO3_OH_HMW84(T, P):
    """c-a-a': potassium carbonate hydroxide [HMW84]."""
    psi = -0.01
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_CO3_OH_HMW84(T, P):
    """c-a-a': calcium carbonate hydroxide [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_CO3_OH_HMW84(T, P):
    """c-a-a': magnesium carbonate hydroxide [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_MgOH_CO3_OH_HMW84(T, P):
    """c-a-a': magnesium-hydroxide carbonate hydroxide [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_CO3_OH_HMW84(T, P):
    """c-a-a': hydrogen carbonate hydroxide [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_CO3_HCO3_HMW84(T, P):
    """a-a': carbonate bicarbonate [HMW84]."""
    theta = -0.04
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_CO3_HCO3_HMW84(T, P):
    """c-a-a': sodium carbonate bicarbonate [HMW84]."""
    psi = 0.002
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_CO3_HCO3_HMW84(T, P):
    """c-a-a': potassium carbonate bicarbonate [HMW84]."""
    psi = 0.012
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_CO3_HCO3_HMW84(T, P):
    """c-a-a': calcium carbonate bicarbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_CO3_HCO3_HMW84(T, P):
    """c-a-a': magnesium carbonate bicarbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_MgOH_CO3_HCO3_HMW84(T, P):
    """c-a-a': magnesium-hydroxide carbonate bicarbonate [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_CO3_HCO3_HMW84(T, P):
    """c-a-a': hydrogen carbonate bicarbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_K_Na_HMW84(T, P):
    """c-c': potassium sodium [HMW84]."""
    theta = -0.012
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_K_Na_Cl_HMW84(T, P):
    """c-c'-a: potassium sodium chloride [HMW84]."""
    psi = -0.0018
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Na_SO4_HMW84(T, P):
    """c-c'-a: potassium sodium sulfate [HMW84]."""
    psi = -0.01
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Na_HSO4_HMW84(T, P):
    """c-c'-a: potassium sodium bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Na_OH_HMW84(T, P):
    """c-c'-a: potassium sodium hydroxide [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Na_HCO3_HMW84(T, P):
    """c-c'-a: potassium sodium bicarbonate [HMW84]."""
    psi = -0.003
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Na_CO3_HMW84(T, P):
    """c-c'-a: potassium sodium carbonate [HMW84]."""
    psi = 0.003
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_Ca_Na_HMW84(T, P):
    """c-c': calcium sodium [HMW84]."""
    theta = 0.07
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Ca_Na_Cl_HMW84(T, P):
    """c-c'-a: calcium sodium chloride [HMW84]."""
    psi = -0.007
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_Na_SO4_HMW84(T, P):
    """c-c'-a: calcium sodium sulfate [HMW84]."""
    psi = -0.055
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_Na_HSO4_HMW84(T, P):
    """c-c'-a: calcium sodium bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_Na_OH_HMW84(T, P):
    """c-c'-a: calcium sodium hydroxide [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_Na_HCO3_HMW84(T, P):
    """c-c'-a: calcium sodium bicarbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_Na_CO3_HMW84(T, P):
    """c-c'-a: calcium sodium carbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_Mg_Na_HMW84(T, P):
    """c-c': magnesium sodium [HMW84]."""
    theta = 0.07
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Mg_Na_Cl_HMW84(T, P):
    """c-c'-a: magnesium sodium chloride [HMW84]."""
    psi = -0.012
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_Na_SO4_HMW84(T, P):
    """c-c'-a: magnesium sodium sulfate [HMW84]."""
    psi = -0.015
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_Na_HSO4_HMW84(T, P):
    """c-c'-a: magnesium sodium bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_Na_OH_HMW84(T, P):
    """c-c'-a: magnesium sodium hydroxide [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_Na_HCO3_HMW84(T, P):
    """c-c'-a: magnesium sodium bicarbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_Na_CO3_HMW84(T, P):
    """c-c'-a: magnesium sodium carbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_MgOH_Na_HMW84(T, P):
    """c-c': magnesium-hydroxide sodium [HMW84]."""
    theta = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_MgOH_Na_Cl_HMW84(T, P):
    """c-c'-a: magnesium-hydroxide sodium chloride [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_MgOH_Na_SO4_HMW84(T, P):
    """c-c'-a: magnesium-hydroxide sodium sulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_MgOH_Na_HSO4_HMW84(T, P):
    """c-c'-a: magnesium-hydroxide sodium bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_MgOH_Na_OH_HMW84(T, P):
    """c-c'-a: magnesium-hydroxide sodium hydroxide [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_MgOH_Na_HCO3_HMW84(T, P):
    """c-c'-a: magnesium-hydroxide sodium bicarbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_MgOH_Na_CO3_HMW84(T, P):
    """c-c'-a: magnesium-hydroxide sodium carbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_H_Na_HMW84(T, P):
    """c-c': hydrogen sodium [HMW84]."""
    theta = 0.036
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_H_Na_Cl_HMW84(T, P):
    """c-c'-a: hydrogen sodium chloride [HMW84]."""
    psi = -0.004
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Na_SO4_HMW84(T, P):
    """c-c'-a: hydrogen sodium sulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Na_HSO4_HMW84(T, P):
    """c-c'-a: hydrogen sodium bisulfate [HMW84]."""
    psi = -0.0129
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Na_OH_HMW84(T, P):
    """c-c'-a: hydrogen sodium hydroxide [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Na_HCO3_HMW84(T, P):
    """c-c'-a: hydrogen sodium bicarbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Na_CO3_HMW84(T, P):
    """c-c'-a: hydrogen sodium carbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_Ca_K_HMW84(T, P):
    """c-c': calcium potassium [HMW84]."""
    theta = 0.032
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Ca_K_Cl_HMW84(T, P):
    """c-c'-a: calcium potassium chloride [HMW84]."""
    psi = -0.025
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_K_SO4_HMW84(T, P):
    """c-c'-a: calcium potassium sulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_K_HSO4_HMW84(T, P):
    """c-c'-a: calcium potassium bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_K_OH_HMW84(T, P):
    """c-c'-a: calcium potassium hydroxide [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_K_HCO3_HMW84(T, P):
    """c-c'-a: calcium potassium bicarbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_K_CO3_HMW84(T, P):
    """c-c'-a: calcium potassium carbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_K_Mg_HMW84(T, P):
    """c-c': potassium magnesium [HMW84]."""
    theta = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_K_Mg_Cl_HMW84(T, P):
    """c-c'-a: potassium magnesium chloride [HMW84]."""
    psi = -0.022
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Mg_SO4_HMW84(T, P):
    """c-c'-a: potassium magnesium sulfate [HMW84]."""
    psi = -0.048
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Mg_HSO4_HMW84(T, P):
    """c-c'-a: potassium magnesium bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Mg_OH_HMW84(T, P):
    """c-c'-a: potassium magnesium hydroxide [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Mg_HCO3_HMW84(T, P):
    """c-c'-a: potassium magnesium bicarbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Mg_CO3_HMW84(T, P):
    """c-c'-a: potassium magnesium carbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_K_MgOH_HMW84(T, P):
    """c-c': potassium magnesium-hydroxide [HMW84]."""
    theta = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_K_MgOH_Cl_HMW84(T, P):
    """c-c'-a: potassium magnesium-hydroxide chloride [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_MgOH_SO4_HMW84(T, P):
    """c-c'-a: potassium magnesium-hydroxide sulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_MgOH_HSO4_HMW84(T, P):
    """c-c'-a: potassium magnesium-hydroxide bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_MgOH_OH_HMW84(T, P):
    """c-c'-a: potassium magnesium-hydroxide hydroxide [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_MgOH_HCO3_HMW84(T, P):
    """c-c'-a: potassium magnesium-hydroxide bicarbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_MgOH_CO3_HMW84(T, P):
    """c-c'-a: potassium magnesium-hydroxide carbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_H_K_HMW84(T, P):
    """c-c': hydrogen potassium [HMW84]."""
    theta = 0.005
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_H_K_Cl_HMW84(T, P):
    """c-c'-a: hydrogen potassium chloride [HMW84]."""
    psi = -0.011
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_K_SO4_HMW84(T, P):
    """c-c'-a: hydrogen potassium sulfate [HMW84]."""
    psi = 0.197
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_K_HSO4_HMW84(T, P):
    """c-c'-a: hydrogen potassium bisulfate [HMW84]."""
    psi = -0.0265
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_K_OH_HMW84(T, P):
    """c-c'-a: hydrogen potassium hydroxide [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_K_HCO3_HMW84(T, P):
    """c-c'-a: hydrogen potassium bicarbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_K_CO3_HMW84(T, P):
    """c-c'-a: hydrogen potassium carbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_Ca_Mg_HMW84(T, P):
    """c-c': calcium magnesium [HMW84]."""
    theta = 0.007
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Ca_Mg_Cl_HMW84(T, P):
    """c-c'-a: calcium magnesium chloride [HMW84]."""
    psi = -0.012
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_Mg_SO4_HMW84(T, P):
    """c-c'-a: calcium magnesium sulfate [HMW84]."""
    psi = 0.024
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_Mg_HSO4_HMW84(T, P):
    """c-c'-a: calcium magnesium bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_Mg_OH_HMW84(T, P):
    """c-c'-a: calcium magnesium hydroxide [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_Mg_HCO3_HMW84(T, P):
    """c-c'-a: calcium magnesium bicarbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_Mg_CO3_HMW84(T, P):
    """c-c'-a: calcium magnesium carbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_Ca_MgOH_HMW84(T, P):
    """c-c': calcium magnesium-hydroxide [HMW84]."""
    theta = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Ca_MgOH_Cl_HMW84(T, P):
    """c-c'-a: calcium magnesium-hydroxide chloride [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_MgOH_SO4_HMW84(T, P):
    """c-c'-a: calcium magnesium-hydroxide sulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_MgOH_HSO4_HMW84(T, P):
    """c-c'-a: calcium magnesium-hydroxide bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_MgOH_OH_HMW84(T, P):
    """c-c'-a: calcium magnesium-hydroxide hydroxide [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_MgOH_HCO3_HMW84(T, P):
    """c-c'-a: calcium magnesium-hydroxide bicarbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_MgOH_CO3_HMW84(T, P):
    """c-c'-a: calcium magnesium-hydroxide carbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_Ca_H_HMW84(T, P):
    """c-c': calcium hydrogen [HMW84]."""
    theta = 0.092
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Ca_H_Cl_HMW84(T, P):
    """c-c'-a: calcium hydrogen chloride [HMW84]."""
    psi = -0.015
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_H_SO4_HMW84(T, P):
    """c-c'-a: calcium hydrogen sulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_H_HSO4_HMW84(T, P):
    """c-c'-a: calcium hydrogen bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_H_OH_HMW84(T, P):
    """c-c'-a: calcium hydrogen hydroxide [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_H_HCO3_HMW84(T, P):
    """c-c'-a: calcium hydrogen bicarbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Ca_H_CO3_HMW84(T, P):
    """c-c'-a: calcium hydrogen carbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_Mg_MgOH_HMW84(T, P):
    """c-c': magnesium magnesium-hydroxide [HMW84]."""
    theta = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Mg_MgOH_Cl_HMW84(T, P):
    """c-c'-a: magnesium magnesium-hydroxide chloride [HMW84]."""
    psi = 0.028
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_MgOH_SO4_HMW84(T, P):
    """c-c'-a: magnesium magnesium-hydroxide sulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_MgOH_HSO4_HMW84(T, P):
    """c-c'-a: magnesium magnesium-hydroxide bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_MgOH_OH_HMW84(T, P):
    """c-c'-a: magnesium magnesium-hydroxide hydroxide [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_MgOH_HCO3_HMW84(T, P):
    """c-c'-a: magnesium magnesium-hydroxide bicarbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_MgOH_CO3_HMW84(T, P):
    """c-c'-a: magnesium magnesium-hydroxide carbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_H_Mg_HMW84(T, P):
    """c-c': hydrogen magnesium [HMW84]."""
    theta = 0.1
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_H_Mg_Cl_HMW84(T, P):
    """c-c'-a: hydrogen magnesium chloride [HMW84]."""
    psi = -0.011
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Mg_SO4_HMW84(T, P):
    """c-c'-a: hydrogen magnesium sulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Mg_HSO4_HMW84(T, P):
    """c-c'-a: hydrogen magnesium bisulfate [HMW84]."""
    psi = -0.0178
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Mg_OH_HMW84(T, P):
    """c-c'-a: hydrogen magnesium hydroxide [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Mg_HCO3_HMW84(T, P):
    """c-c'-a: hydrogen magnesium bicarbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Mg_CO3_HMW84(T, P):
    """c-c'-a: hydrogen magnesium carbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_H_MgOH_HMW84(T, P):
    """c-c': hydrogen magnesium-hydroxide [HMW84]."""
    theta = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_H_MgOH_Cl_HMW84(T, P):
    """c-c'-a: hydrogen magnesium-hydroxide chloride [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_MgOH_SO4_HMW84(T, P):
    """c-c'-a: hydrogen magnesium-hydroxide sulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_MgOH_HSO4_HMW84(T, P):
    """c-c'-a: hydrogen magnesium-hydroxide bisulfate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_MgOH_OH_HMW84(T, P):
    """c-c'-a: hydrogen magnesium-hydroxide hydroxide [HMW84]."""
    psi = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_MgOH_HCO3_HMW84(T, P):
    """c-c'-a: hydrogen magnesium-hydroxide bicarbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_MgOH_CO3_HMW84(T, P):
    """c-c'-a: hydrogen magnesium-hydroxide carbonate [HMW84]."""
    psi = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def lambd_CO2_H_HMW84(T, P):
    """n-c: carbon-dioxide hydrogen [HMW84]."""
    lambd = 0.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_CO2_Na_HMW84(T, P):
    """n-c: carbon-dioxide sodium [HMW84]."""
    lambd = 0.1
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_CO2_K_HMW84(T, P):
    """n-c: carbon-dioxide potassium [HMW84]."""
    lambd = 0.051
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_CO2_Ca_HMW84(T, P):
    """n-c: carbon-dioxide calcium [HMW84]."""
    lambd = 0.183
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_CO2_Mg_HMW84(T, P):
    """n-c: carbon-dioxide magnesium [HMW84]."""
    lambd = 0.183
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_CO2_Cl_HMW84(T, P):
    """n-a: carbon-dioxide chloride [HMW84]."""
    lambd = -0.005
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_CO2_SO4_HMW84(T, P):
    """n-a: carbon-dioxide sulfate [HMW84]."""
    lambd = 0.097
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_CO2_HSO4_HMW84(T, P):
    """n-c: carbon-dioxide bisulfate [HMW84]."""
    lambd = -0.003
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pitzer et al. (1985) ~~~~~
def bC_Mg_HCO3_POS85(T, P):
    """c-a: magnesium bicarbonate [POS85]."""
    b0 = 0.033
    b1 = 0.85
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ca_HCO3_POS85(T, P):
    """c-a: calcium bicarbonate [POS85]."""
    b0 = 0.28
    b1 = 0.3
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Felmy & Weare (1986) ~~~~~
def bC_Na_BOH4_FW86(T, P):
    """c-a: sodium borate [FW86]."""
    b0 = -0.0427
    b1 = 0.089
    b2 = 0
    Cphi = 0.0114
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["BOH4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_B3O3OH4_FW86(T, P):
    """c-a: sodium triborate [FW86]."""
    b0 = -0.056
    b1 = -0.910
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_B4O5OH4_FW86(T, P):
    """c-a: sodium tetraborate [FW86]."""
    b0 = -0.11
    b1 = -0.40
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_BOH4_FW86(T, P):
    """c-a: potassium borate [FW86]."""
    b0 = 0.035
    b1 = 0.14
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_B3O3OH4_FW86(T, P):
    """c-a: potassium triborate [FW86]."""
    b0 = -0.13
    b1 = 0
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_B4O5OH4_FW86(T, P):
    """c-a: potassium tetraborate [FW86]."""
    b0 = -0.022
    b1 = 0
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_MgBOH4_Cl_FW86(T, P):
    """c-a: magnesium-borate chloride [FW86]."""
    b0 = 0.16
    b1 = 0
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_CaBOH4_Cl_FW86(T, P):
    """c-a: calcium-borate chloride [FW86]."""
    b0 = 0.12
    b1 = 0
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def theta_BOH4_Cl_FW86(T, P):
    """a-a': borate chloride [FW86]."""
    theta = -0.065
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_BOH4_Cl_FW86(T, P):
    """c-a-a': sodium borate chloride [FW86]."""
    psi = -0.0073
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_BOH4_SO4_FW86(T, P):
    """a-a': borate sulfate [FW86]."""
    theta = -0.012
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_B3O3OH4_Cl_FW86(T, P):
    """a-a': triborate chloride [FW86]."""
    theta = 0.12
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_B3O3OH4_Cl_FW86(T, P):
    """c-a-a': sodium triborate chloride [FW86]."""
    psi = -0.024
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_B3O3OH4_SO4_FW86(T, P):
    """a-a': triborate sulfate [FW86]."""
    theta = 0.10
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_B3O3OH4_HCO3_FW86(T, P):
    """a-a': triborate bicarbonate [FW86]."""
    theta = -0.10
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_B4O5OH4_Cl_FW86(T, P):
    """a-a': tetraborate chloride [FW86]."""
    theta = 0.074
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_B4O5OH4_Cl_FW86(T, P):
    """c-a-a': sodium tetraborate chloride [FW86]."""
    psi = 0.026
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_B4O5OH4_SO4_FW86(T, P):
    """a-a': tetraborate sulfate [FW86]."""
    theta = 0.12
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_B4O5OH4_HCO3_FW86(T, P):
    """a-a': tetraborate bicarbonate [FW86]."""
    theta = -0.087
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def lambd_BOH3_Cl_FW86(T, P):
    """n-a: boric-acid chloride [FW86]."""
    lambd = 0.091
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_BOH3_SO4_FW86(T, P):
    """n-a: boric-acid sulfate [FW86]."""
    lambd = 0.018
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_BOH3_B3O3OH4_FW86(T, P):
    """n-a: boric-acid triborate [FW86]."""
    lambd = -0.20
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_BOH3_Na_FW86(T, P):
    """n-c: boric-acid sodium [FW86]."""
    lambd = -0.097
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_BOH3_K_FW86(T, P):
    """n-c: boric-acid potassium [FW86]."""
    lambd = -0.14
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def zeta_BOH3_H_Cl_FW86(T, P):
    """n-c-a: boric-acid hydrogen chloride [FW86]."""
    zeta = -0.0102
    valid = np.isclose(T, 298.15, **temperature_tol)
    return zeta, valid


def zeta_BOH3_Na_SO4_FW86(T, P):
    """n-c-a: boric-acid sodium sulfate [FW86]."""
    zeta = 0.046
    valid = np.isclose(T, 298.15, **temperature_tol)
    return zeta, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Phutela & Pitzer (1986) ~~~~~
PP86ii_Tr = 298.15


def PP86ii_eq28(T, q):
    """PP86ii equation 28."""
    Tr = PP86ii_Tr
    return (
        (T ** 2 - Tr ** 2) * q[0] / 2
        + (T ** 3 - Tr ** 3) * q[1] / 3
        + (T ** 4 - Tr ** 4) * q[2] / 4
        + (T ** 5 - Tr ** 5) * q[3] / 5
        + Tr ** 2 * q[4]
    ) / T ** 2


def PP86ii_eq29(T, q):
    """PP86ii equation 29."""
    # q[x]     b0         b1         b2         C0
    #   0      q6         q10        q12        q15
    #   1      q7         q11        q13        q16
    #   2      q8          0         q14        q17
    #   3      q9          0          0         q18
    #   4    b0L(Tr)    b1L(Tr)    b2L(Tr)    C0L(Tr)
    #   5     b0(Tr)     b1(Tr)     b2(Tr)     C0(Tr)    from RM81
    Tr = PP86ii_Tr
    # Substitution to avoid 'difference of two large numbers' error
    t = T / Tr
    # Original fourth line was:
    #  + q[3] * (T**4/20 + Tr**5/(5*T) - Tr**4/4)
    return (
        q[0] * (T / 2 + Tr ** 2 / (2 * T) - Tr)
        + q[1] * (T ** 2 / 6 + Tr ** 3 / (3 * T) - Tr ** 2 / 2)
        + q[2] * (T ** 3 / 12 + Tr ** 4 / (4 * T) - Tr ** 3 / 3)
        + q[3] * (t ** 5 + 4 - 5 * t) * Tr ** 5 / (20 * T)
        + q[4] * (Tr - Tr ** 2 / T)
        + q[5]
    )


def bC_Mg_SO4_PP86ii(T, P):
    """c-a: magnesium sulfate [PP86ii]."""
    b0r, b1r, b2r, C0r, C1, alph1, alph2, omega, _ = bC_Mg_SO4_RM81i(T, P)
    b0 = PP86ii_eq29(
        T,
        [
            -1.0282,
            8.4790e-3,
            -2.3366e-5,
            2.1575e-8,
            6.8402e-4,
            b0r,
        ],
    )
    b1 = PP86ii_eq29(
        T,
        [
            -2.9596e-1,
            9.4564e-4,
            0,
            0,
            1.1028e-2,
            b1r,
        ],
    )
    b2 = PP86ii_eq29(
        T,
        [
            -1.3764e1,
            1.2121e-1,
            -2.7642e-4,
            0,
            -2.1515e-1,
            b2r,
        ],
    )
    C0 = PP86ii_eq29(
        T,
        [
            1.0541e-1,
            -8.9316e-4,
            2.5100e-6,
            -2.3436e-9,
            -8.7899e-5,
            C0r,
        ],
    )
    valid = T <= 473
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Holmes and Mesmer (1986) ~~~~~
# Note that HM86 use alph1 of 1.4 even where there is no beta2 term (p. 502)
# Also HM86 contains functions for caesium and lithium sulfates, not yet coded
def HM86_eq8(T, a):
    """HM86 equation 8."""
    TR = 298.15
    # Typo in a[5] term in HM86 has been corrected here
    return (
        a[0]
        + a[1] * (TR - TR ** 2 / T)
        + a[2] * (T ** 2 + 2 * TR ** 3 / T - 3 * TR ** 2)
        + a[3] * (T + TR ** 2 / T - 2 * TR)
        + a[4] * (np.log(T / TR) + TR / T - 1)
        + a[5] * (1 / (T - 263) + (263 * T - TR ** 2) / (T * (TR - 263) ** 2))
        + a[6] * (1 / (680 - T) + (TR ** 2 - 680 * T) / (T * (680 - TR) ** 2))
    )


def bC_K_SO4_HM86(T, P):
    """c-a: potassium sulfate [HM86]."""
    b0 = HM86_eq8(
        T,
        [
            0,
            7.476e-4,
            0,
            4.265e-3,
            -3.088,
            0,
            0,
        ],
    )
    b1 = HM86_eq8(
        T,
        [
            0.6179,
            6.85e-3,
            5.576e-5,
            -5.841e-2,
            0,
            -0.90,
            0,
        ],
    )
    b2 = 0
    Cphi = HM86_eq8(
        T,
        [
            9.15467e-3,
            0,
            0,
            -1.81e-4,
            0,
            0,
            0,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["SO4"])))
    C1 = 0
    alph1 = 1.4
    alph2 = -9
    omega = -9
    valid = (T >= 298.15) & (T <= 523.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_SO4_HM86(T, P):
    """c-a: sodium sulfate [HM86]."""
    b0 = HM86_eq8(
        T,
        [
            -1.727e-2,
            1.7828e-3,
            9.133e-6,
            0,
            -6.552,
            0,
            -96.90,
        ],
    )
    b1 = HM86_eq8(
        T,
        [
            0.7534,
            5.61e-3,
            -5.7513e-4,
            1.11068,
            -378.82,
            0,
            1861.3,
        ],
    )
    b2 = 0
    Cphi = HM86_eq8(
        T,
        [
            1.1745e-2,
            -3.3038e-4,
            1.85794e-5,
            -3.9200e-2,
            14.2130,
            0,
            -24.950,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["SO4"])))
    C1 = 0
    alph1 = 1.4
    alph2 = -9
    omega = -9
    valid = (T >= 298.15) & (T <= 523.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Roy et al. (1986) ~~~~~
def bC_Sr_Cl_RGRG86(T, P):
    """c-a: strontium chloride [RGRG86]."""
    b0, b1, b2, C0, C1, alph1, alph2, omega, _ = bC_Sr_Cl_PM73(T, P)
    b0 = b0 + (T - 298.15) * -3.073e-4
    b1 = b1 + (T - 298.15) * 122.379e-4
    C0 = C0 + (T - 298.15) * -6.688e-4 / 2 * np.sqrt(np.abs(i2c["Sr"] * i2c["Cl"]))
    valid = (T >= 278.15) & (T <= 318.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def theta_H_Sr_RGRG86(T, P):
    """c-c': hydrogen strontium [RGRG86]."""
    theta = 0.0591 + (T - 298.15) * 0.00045
    valid = (T >= 278.15) & (T <= 318.15)
    return theta, valid


def psi_H_Sr_Cl_RGRG86(T, P):
    """c-c'-a: hydrogen strontium chloride [RGRG86]."""
    psi = 0.0054 + (T - 298.15) * 0.00021
    valid = (T >= 278.15) & (T <= 318.15)
    return psi, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pabalan and Pitzer (1987i) ~~~~~
# Note that there are two Pabalan & Pitzer (1987)'s: one compiling a suite of
#  electrolytes (PP87ii), and one just for NaOH (PP87i).
# There are also a bunch of Phutela & Pitzer papers in similar years, so take
#  care with reference codes!
def PP87i_eqNaOH(T, P, a):
    """PP87i equation for sodium hydroxide, with pressure in bar."""
    return (
        a[0]
        + a[1] * P
        + a[2] / T
        + a[3] * P / T
        + a[4] * np.log(T)
        + a[5] * T
        + a[6] * T * P
        + a[7] * T ** 2
        + a[8] * T ** 2 * P
        + a[9] / (T - 227)
        + a[10] / (647 - T)
        + a[11] * P / (647 - T)
    )


def bC_Na_OH_PP87i(T, P):
    """c-a: sodium hydroxide [PP87i]."""
    P_bar = P / 10  # convert dbar to bar
    b0 = PP87i_eqNaOH(
        T,
        P_bar,
        [
            2.7682478e2,
            -2.8131778e-3,
            -7.3755443e3,
            3.7012540e-1,
            -4.9359970e1,
            1.0945106e-1,
            7.1788733e-6,
            -4.0218506e-5,
            -5.8847404e-9,
            1.1931122e1,
            2.4824963e00,
            -4.8217410e-3,
        ],
    )
    b1 = PP87i_eqNaOH(
        T,
        P_bar,
        [
            4.6286977e2,
            0,
            -1.0294181e4,
            0,
            -8.5960581e1,
            2.3905969e-1,
            0,
            -1.0795894e-4,
            0,
            0,
            0,
            0,
        ],
    )
    b2 = 0
    Cphi = PP87i_eqNaOH(
        T,
        P_bar,
        [
            -1.6686897e01,
            4.0534778e-04,
            4.5364961e02,
            -5.1714017e-02,
            2.9680772e000,
            -6.5161667e-03,
            -1.0553037e-06,
            2.3765786e-06,
            8.9893405e-10,
            -6.8923899e-01,
            -8.1156286e-02,
            0,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["OH"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 298.15) & (T <= 523.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mg_Cl_PP87i(T, P):
    """c-a: magnesium chloride [PP87i]."""
    b0, b1, b2, _, C1, alph1, alph2, omega, _ = bC_Mg_Cl_dLP83(T, P)
    Cphi = 2.41831e-7 * T ** 2 - 2.49949e-4 * T + 5.95320e-2
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Mg"] * i2c["Cl"])))
    valid = (T >= 298.15) & (T <= 473.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pabalan and Pitzer (1987ii) ~~~~~
def theta_K_Na_PP87ii(T, P):
    """c-c': potassium sodium [PP87ii]."""
    theta = -0.012
    # Validity range declared by MP98 for theta(Na, Mg) from this study
    valid = (T >= 298.15) & (T <= 523.25)
    return theta, valid


def theta_Mg_Na_PP87ii(T, P):
    """c-c': magnesium sodium [PP87ii]."""
    theta = 0.07
    # Validity range declared by MP98 for theta(Na, Mg) from this study
    valid = (T >= 298.15) & (T <= 523.25)
    return theta, valid


def theta_K_Mg_PP87ii(T, P):
    """c-c': potassium magnesium [PP87ii]."""
    theta = 0
    # Validity range declared by MP98 for theta(Na, Mg) from this study
    valid = (T >= 298.15) & (T <= 523.25)
    return theta, valid


def theta_Cl_SO4_PP87ii(T, P):
    """a-a': chloride sulfate [PP87ii]."""
    theta = 0.030
    # Validity range declared by MP98 for theta(Na, Mg) from this study
    valid = (T >= 298.15) & (T <= 523.25)
    return theta, valid


def theta_Cl_OH_PP87ii(T, P):
    """a-a': chloride hydroxide [PP87ii]."""
    theta = -0.050
    # Validity range declared by MP98 for theta(Na, Mg) from this study
    valid = (T >= 298.15) & (T <= 523.25)
    return theta, valid


def theta_OH_SO4_PP87ii(T, P):
    """a-a': hydroxide sulfate [PP87ii]."""
    theta = -0.013
    # Validity range declared by MP98 for theta(Na, Mg) from this study
    valid = (T >= 298.15) & (T <= 523.25)
    return theta, valid


def psi_K_Na_Cl_PP87ii(T, P):
    """c-c'-a: potassium sodium chloride [PP87ii]."""
    psi = -6.81e-3 + 1.68e-5 * T
    # Validity range declared by MP98 for theta(Na, Mg) from this study
    valid = (T >= 298.15) & (T <= 523.25)
    return psi, valid


def psi_Mg_Na_Cl_PP87ii(T, P):
    """c-c'-a: magnesium sodium chloride [PP87ii]."""
    psi = 1.99e-2 - 9.51 / T
    # Validity range declared by MP98 for theta(Na, Mg) from this study
    valid = (T >= 298.15) & (T <= 523.25)
    return psi, valid


def psi_K_Mg_Cl_PP87ii(T, P):
    """c-c'-a: potassium magnesium chloride [PP87ii]."""
    psi = 2.586e-2 - 14.27 / T
    # Validity range declared by MP98 for theta(Na, Mg) from this study
    valid = (T >= 298.15) & (T <= 523.25)
    return psi, valid


def psi_Na_Cl_SO4_PP87ii(T, P):
    """c-a-a': sodium chloride sulfate [PP87ii]."""
    psi = 0
    # Validity range declared by MP98 for theta(Na, Mg) from this study
    valid = (T >= 298.15) & (T <= 523.25)
    return psi, valid


def psi_K_Cl_SO4_PP87ii(T, P):
    """c-a-a': potassium chloride sulfate [PP87ii]."""
    psi = -5e-3
    # Validity range declared by MP98 for theta(Na, Mg) from this study
    valid = (T >= 298.15) & (T <= 523.25)
    return psi, valid


def psi_Mg_Cl_SO4_PP87ii(T, P):
    """c-a-a': magnesium chloride sulfate [PP87ii]."""
    psi = -1.174e-1 + 32.63 / T
    # Validity range declared by MP98 for theta(Na, Mg) from this study
    valid = (T >= 298.15) & (T <= 523.25)
    return psi, valid


def psi_Na_Cl_OH_PP87ii(T, P):
    """c-a-a': sodium chloride hydroxide [PP87ii]."""
    psi = 2.73e-2 - 9.93 / T
    # Validity range declared by MP98 for theta(Na, Mg) from this study
    valid = (T >= 298.15) & (T <= 523.25)
    return psi, valid


def psi_Na_OH_SO4_PP87ii(T, P):
    """c-a-a': sodium hydroxide sulfate [PP87ii]."""
    psi = 3.02e-2 - 11.69 / T
    # Validity range declared by MP98 for theta(Na, Mg) from this study
    valid = (T >= 298.15) & (T <= 523.25)
    return psi, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Simonson, Roy & Gibbons (1987) ~~~~~
def bC_K_CO3_SRG87(T, P):
    """c-a: potassium carbonate [SRG87]."""
    b0 = 0.1288 + 1.1e-3 * (T - 298.15) - 5.1e-6 * (T - 298.15) ** 2
    b1 = 1.433 + 4.36e-3 * (T - 298.15) + 2.07e-5 * (T - 298.15) ** 2
    b2 = 0
    # MP98 declare Cphi = 0.0005, but I can't find that anywhere in SRG87
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 278.15) & (T <= 368.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Simonson et al. (1987) ~~~~~
def SRRJ87_eq7(T, a):
    """SRRJ87 equation 7."""
    Tr = 298.15
    return a[0] + a[1] * 1e-3 * (T - Tr) + a[2] * 1e-5 * (T - Tr) ** 2


def bC_K_Cl_SRRJ87(T, P):
    """c-a: potassium chloride [SRRJ87]."""
    # Parameters from SRRJ87 Table III
    b0 = SRRJ87_eq7(
        T,
        [
            0.0481,
            0.592,
            -0.562,
        ],
    )
    b1 = SRRJ87_eq7(
        T,
        [
            0.2188,
            1.500,
            -1.085,
        ],
    )
    b2 = 0
    Cphi = (
        SRRJ87_eq7(
            T,
            [
                -0.790,
                -0.639,
                0.613,
            ],
        )
        * 1e-3
    )  # *1e-3 presumably?
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 278.15) & (T <= 328.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_Cl_SRRJ87(T, P):
    """c-a: sodium chloride [SRRJ87]."""
    # Parameters from SRRJ87 Table III
    b0 = SRRJ87_eq7(
        T,
        [
            0.0754,
            0.792,
            -0.935,
        ],
    )
    b1 = SRRJ87_eq7(
        T,
        [
            0.2770,
            1.006,
            -0.756,
        ],
    )
    b2 = 0
    Cphi = (
        SRRJ87_eq7(
            T,
            [
                1.40,
                -1.20,
                1.15,
            ],
        )
        * 1e-3
    )  # *1e-3 presumably?
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 278.15) & (T <= 328.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_BOH4_SRRJ87(T, P):
    """c-a: potassium borate [SRRJ87]."""
    # Parameters from SRRJ87 Table III
    b0 = SRRJ87_eq7(
        T,
        [
            0.1469,
            2.881,
            0,
        ],
    )
    b1 = SRRJ87_eq7(
        T,
        [
            -0.0989,
            -6.876,
            0,
        ],
    )
    b2 = 0
    Cphi = (
        SRRJ87_eq7(
            T,
            [
                -56.43,
                -9.56,
                0,
            ],
        )
        * 1e-3
    )  # *1e-3 presumably?
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["BOH4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 278.15) & (T <= 328.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_BOH4_SRRJ87(T, P):
    """c-a: sodium borate [SRRJ87]."""
    # Parameters from SRRJ87 Table III
    b0 = SRRJ87_eq7(
        T,
        [
            -0.0510,
            5.264,
            0,
        ],
    )
    b1 = SRRJ87_eq7(
        T,
        [
            0.0961,
            -10.68,
            0,
        ],
    )
    b2 = 0
    Cphi = (
        SRRJ87_eq7(
            T,
            [
                14.98,
                -15.7,
                0,
            ],
        )
        * 1e-3
    )  # *1e-3 presumably?
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["BOH4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 278.15) & (T <= 328.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def theta_BOH4_Cl_SRRJ87(T, P):
    """a-a': borate chloride [SRRJ87]."""
    # Parameter from SRRJ87 Table III
    theta = -0.056
    valid = (T >= 278.15) & (T <= 328.15)
    return theta, valid


def psi_K_BOH4_Cl_SRRJ87(T, P):
    """c-a-a': potassium borate chloride [SRRJ87]."""
    psi = 0
    valid = (T >= 278.15) & (T <= 328.15)
    return psi, valid


def psi_Na_BOH4_Cl_SRRJ87(T, P):
    """c-a-a': sodium borate chloride [SRRJ87]."""
    # Parameter from SRRJ87 Table III
    psi = -0.019
    valid = (T >= 278.15) & (T <= 328.15)
    return psi, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Simonson et al. (1987b) ~~~~~
def SRM87_eqTableIII(T, abc):
    """SRM87 equation from Table III."""
    return abc[0] + abc[1] * 1e-3 * (T - 298.15) + abc[2] * 1e-3 * (T - 303.15) ** 2


def bC_Mg_BOH4_SRM87(T, P):
    """c-a: magnesium borate [SRM87]."""
    b0 = SRM87_eqTableIII(
        T,
        [
            -0.6230,
            6.496,
            0,
        ],
    )
    b1 = SRM87_eqTableIII(
        T,
        [
            0.2515,
            -17.13,
            0,
        ],
    )
    b2 = SRM87_eqTableIII(
        T,
        [
            -11.47,
            0,
            -3.240,
        ],
    )
    C0 = 0
    C1 = 0
    alph1 = 1.4
    alph2 = 6
    omega = -9
    valid = (T >= 278.15) & (T <= 528.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ca_BOH4_SRM87(T, P):
    """c-a: calcium borate [SRM87]."""
    b0 = SRM87_eqTableIII(
        T,
        [
            -0.4462,
            5.393,
            0,
        ],
    )
    b1 = SRM87_eqTableIII(
        T,
        [
            -0.8680,
            -18.20,
            0,
        ],
    )
    b2 = SRM87_eqTableIII(
        T,
        [
            -15.88,
            0,
            -2.858,
        ],
    )
    C0 = 0
    C1 = 0
    alph1 = 1.4
    alph2 = 6
    omega = -9
    valid = (T >= 278.15) & (T <= 528.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Hershey et al. (1988) ~~~~~
def bC_Na_HS_HPM88(T, P):
    """c-a: sodium bisulfide [HPM88]."""
    b0 = 3.66e-1 - 6.75e1 / T
    b1 = 0
    b2 = 0
    Cphi = -1.27e-2
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["HS"])))
    C1 = 0
    alph1 = -9
    alph2 = -9
    omega = -9
    valid = (T >= 278.15) & (T <= 318.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_HS_HPM88(T, P):
    """c-a: potassium bisulfide [HPM88]."""
    b0 = 6.37e-1 - 1.40e2 / T
    b1 = 0
    b2 = 0
    Cphi = -1.94e-1
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["HS"])))
    C1 = 0
    alph1 = -9
    alph2 = -9
    omega = -9
    valid = (T >= 278.15) & (T <= 298.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mg_HS_HPM88(T, P):
    """c-a: magnesium bisulfide [HPM88]."""
    b0 = 1.70e-1
    b1 = 2.78
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ca_HS_HPM88(T, P):
    """c-a: calcium bisulfide [HPM88]."""
    b0 = -1.05e-1
    b1 = 3.43
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Møller (1988) ~~~~~
def M88_eq13(T, a):
    """M88 equation 13."""
    return (
        a[0]
        + a[1] * T
        + a[2] / T
        + a[3] * np.log(T)
        + a[4] / (T - 263)
        + a[5] * T ** 2
        + a[6] / (680 - T)
        + a[7] / (T - 227)
    )


def b0_Ca_Cl_M88(T, P):
    """beta0: calcium chloride [M88]."""
    return M88_eq13(
        T,
        [
            -9.41895832e1,
            -4.04750026e-2,
            2.34550368e3,
            1.70912300e1,
            -9.22885841e-1,
            1.51488122e-5,
            -1.39082000e00,
            0,
        ],
    )


def b1_Ca_Cl_M88(T, P):
    """beta1: calcium chloride [M88]."""
    return M88_eq13(
        T,
        [
            3.47870000e00,
            -1.54170000e-2,
            0,
            0,
            0,
            3.17910000e-5,
            0,
            0,
        ],
    )


def Cphi_Ca_Cl_M88(T, P):
    """Cphi: calcium chloride [M88]."""
    return M88_eq13(
        T,
        [
            -3.03578731e1,
            -1.36264728e-2,
            7.64582238e2,
            5.50458061e00,
            -3.27377782e-1,
            5.69405869e-6,
            -5.36231106e-1,
            0,
        ],
    )


def bC_Ca_Cl_M88(T, P):
    """c-a: calcium chloride [M88]."""
    b0 = b0_Ca_Cl_M88(T, P)
    b1 = b1_Ca_Cl_M88(T, P)
    b2 = 0
    Cphi = Cphi_Ca_Cl_M88(T, P)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Ca"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 298.15) & (T <= 523.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ca_SO4_M88(T, P):
    """c-a: calcium sulfate [M88]."""
    b0 = 0.15
    b1 = 3.00
    b2 = M88_eq13(
        T,
        [
            -1.29399287e2,
            4.00431027e-1,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    C0 = 0
    C1 = 0
    alph1 = 1.4
    alph2 = 12
    omega = -9
    valid = (T >= 298.15) & (T <= 523.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_Cl_M88(T, P):
    """c-a: sodium chloride [M88]."""
    b0 = M88_eq13(
        T,
        [
            1.43783204e1,
            5.60767406e-3,
            -4.22185236e2,
            -2.51226677e00,
            0,
            -2.61718135e-6,
            4.43854508e00,
            -1.70502337e00,
        ],
    )
    b1 = M88_eq13(
        T,
        [
            -4.83060685e-1,
            1.40677479e-3,
            1.19311989e2,
            0,
            0,
            0,
            0,
            -4.23433299e00,
        ],
    )
    b2 = 0
    Cphi = M88_eq13(
        T,
        [
            -1.00588714e-1,
            -1.80529413e-5,
            8.61185543e00,
            1.24880954e-2,
            0,
            3.41172108e-8,
            6.83040995e-2,
            2.93922611e-1,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 573.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_SO4_M88(T, P):
    """c-a: sodium sulfate [M88]."""
    b0 = M88_eq13(
        T,
        [
            8.16920027e1,
            3.01104957e-2,
            -2.32193726e3,
            -1.43780207e1,
            -6.66496111e-1,
            -1.03923656e-5,
            0,
            0,
        ],
    )
    b1 = M88_eq13(
        T,
        [
            1.00463018e3,
            5.77453682e-1,
            -2.18434467e4,
            -1.89110656e2,
            -2.03550548e-1,
            -3.23949532e-4,
            1.46772243e3,
            0,
        ],
    )
    b2 = 0
    Cphi = M88_eq13(
        T,
        [
            -8.07816886e1,
            -3.54521126e-2,
            2.02438830e3,
            1.46197730e1,
            -9.16974740e-2,
            1.43946005e-5,
            -2.42272049e00,
            0,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["SO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 573.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def theta_Ca_Na_M88(T, P):
    """c-c': calcium sodium [M88]."""
    theta = 0.05
    valid = (T >= 298.15) & (T <= 523.15)
    return theta, valid


def theta_Cl_SO4_M88(T, P):
    """a-a': chloride sulfate [M88]."""
    theta = 0.07
    valid = (T >= 298.15) & (T <= 423.15)
    return theta, valid


def psi_Ca_Na_Cl_M88(T, P):
    """c-c'-a: calcium sodium chloride [M88]."""
    psi = -0.003
    valid = (T >= 298.15) & (T <= 523.15)
    return psi, valid


def psi_Ca_Na_SO4_M88(T, P):
    """c-c'-a: calcium sodium sulfate [M88]."""
    psi = -0.012
    valid = (T >= 298.15) & (T <= 523.15)
    return psi, valid


def psi_Ca_Cl_SO4_M88(T, P):
    """c-a-a': calcium chloride sulfate [M88]."""
    psi = -0.018
    valid = (T >= 298.15) & (T <= 523.15)
    return psi, valid


def psi_Na_Cl_SO4_M88(T, P):
    """c-a-a': sodium chloride sulfate [M88]."""
    psi = -0.009
    valid = (T >= 298.15) & (T <= 423.15)
    return psi, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Clegg & Brimblecombe (1989) ~~~~~
def lambd_NH3_NH3_CB89(T, P):
    """n-n: ammonia ammonia [CB89]."""
    lambd = 0.033161 - 21.12816 / T + 4665.1461 / T ** 2
    valid = (T >= 273.15) & (T <= 313.15)
    return lambd, valid


def lambd_NH3_Mg_CB89(T, P):
    """n-c: ammonia magnesium [CB89]."""
    lambd = -0.21
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_NH3_Ca_CB89(T, P):
    """n-c: ammonia calcium [CB89]."""
    lambd = -0.081
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_NH3_Sr_CB89(T, P):
    """n-c: ammonia strontium [CB89]."""
    lambd = -0.041
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_NH3_Ba_CB89(T, P):
    """n-c: ammonia barium [CB89]."""
    lambd = -0.021
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_NH3_Li_CB89(T, P):
    """n-c: ammonia lithium [CB89]."""
    lambd = -0.038
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_NH3_Na_CB89(T, P):
    """n-c: ammonia sodium [CB89]."""
    lambd = 0.0175
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_NH3_K_CB89(T, P):
    """n-c: ammonia potassium [CB89]."""
    lambd = 0.0454 + (T - 298.15) * -0.000141
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_NH3_NH4_CB89(T, P):
    """n-c: ammonia ammonium [CB89]."""
    lambd = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_NH3_F_CB89(T, P):
    """n-a: ammonia fluoride [CB89]."""
    lambd = 0.091
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_NH3_Cl_CB89(T, P):
    """n-a: ammonia chloride [CB89]."""
    lambd = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_NH3_Br_CB89(T, P):
    """n-a: ammonia bromide [CB89]."""
    lambd = -0.022
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_NH3_I_CB89(T, P):
    """n-a: ammonia iodide [CB89]."""
    lambd = -0.051
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_NH3_OH_CB89(T, P):
    """n-a: ammonia hydroxide [CB89]."""
    lambd = 0.103
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_NH3_ClO3_CB89(T, P):
    """n-a: ammonia chlorate [CB89]."""
    lambd = -0.004
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_NH3_ClO4_CB89(T, P):
    """n-a: ammonia perchlorate [CB89]."""
    lambd = -0.056
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_NH3_NO2_CB89(T, P):
    """n-a: ammonia nitrite [CB89]."""
    lambd = -0.003
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_NH3_NO3_CB89(T, P):
    """n-a: ammonia nitrate [CB89]."""
    lambd = -0.01
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_NH3_SCN_CB89(T, P):
    """n-a: ammonia thiocyanide [CB89]."""
    lambd = -0.017
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_NH3_S_CB89(T, P):
    """n-a: ammonia sulfide [CB89]."""
    lambd = 0.174
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_NH3_SO3_CB89(T, P):
    """n-a: ammonia sulfite [CB89]."""
    lambd = 0.158
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_NH3_SO4_CB89(T, P):
    """n-a: ammonia sulfate [CB89]."""
    lambd = 0.140
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_NH3_CO3_CB89(T, P):
    """n-a: ammonia carbonate [CB89]."""
    lambd = 0.180
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_NH3_HCOO_CB89(T, P):
    """n-a: ammonia methanoate [CB89]."""
    lambd = 0.048
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_NH3_CH3COO_CB89(T, P):
    """n-a: ammonia ethanoate [CB89]."""
    lambd = 0.036
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_NH3_COO2_CB89(T, P):
    """n-a: ammonia oxalate [CB89]."""
    lambd = 0.012
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def mun2i_NH3_NH3_Na_CB89(T, P):
    """n-n-c: ammonia ammonia sodium [CB89]."""
    mun2i = -0.000311
    valid = np.isclose(T, 298.15, **temperature_tol)
    return mun2i, valid


def mun2i_NH3_NH3_K_CB89(T, P):
    """n-n-c: ammonia ammonia potassium [CB89]."""
    mun2i = -0.000321
    valid = np.isclose(T, 298.15, **temperature_tol)
    return mun2i, valid


def mun2i_NH3_NH3_NH4_CB89(T, P):
    """n-n-c: ammonia ammonia ammonium [CB89]."""
    mun2i = -0.00075
    valid = np.isclose(T, 298.15, **temperature_tol)
    return mun2i, valid


def mun2i_NH3_NH3_Cl_CB89(T, P):
    """n-n-a: ammonia ammonia chloride [CB89]."""
    mun2i = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return mun2i, valid


def mun2i_NH3_NH3_NO3_CB89(T, P):
    """n-n-a: ammonia ammonia nitrate [CB89]."""
    mun2i = -0.000437
    valid = np.isclose(T, 298.15, **temperature_tol)
    return mun2i, valid


def mun2i_NH3_NH3_CO3_CB89(T, P):
    """n-n-a: ammonia ammonia carbonate [CB89]."""
    mun2i = 0.000625
    valid = np.isclose(T, 298.15, **temperature_tol)
    return mun2i, valid


def zeta_NH3_Ca_Cl_CB89(T, P):
    """n-c-a: ammonia calcium chloride [CB89]."""
    zeta = -0.00134
    valid = np.isclose(T, 298.15, **temperature_tol)
    return zeta, valid


def zeta_NH3_K_OH_CB89(T, P):
    """n-c-a: ammonia potassium hydroxide [CB89]."""
    zeta = 0.00385
    valid = np.isclose(T, 298.15, **temperature_tol)
    return zeta, valid


def munii_NH3_NH4_SO4_CB89(T, P):
    """n-a-a': ammonia ammonium sulfate [CB89]."""
    munii = -0.00153
    valid = np.isclose(T, 298.15, **temperature_tol)
    return munii, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Greenberg & Møller (1989) ~~~~~
def GM89_eq3(T, a):
    """GM89 equation 3."""
    return M88_eq13(T, a)


def Cphi_Ca_Cl_GM89(T, P):
    """Cphi: calcium chloride [GM89]."""
    return GM89_eq3(
        T,
        [
            1.93056024e1,
            9.77090932e-3,
            -4.28383748e2,
            -3.57996343e00,
            8.82068538e-2,
            -4.62270238e-6,
            9.91113465e00,
            0,
        ],
    )


def bC_Ca_Cl_GM89(T, P):
    """c-a: calcium chloride [GM89]."""
    b0, b1, b2, _, C1, alph1, alph2, omega, valid = bC_Ca_Cl_M88(T, P)
    Cphi = Cphi_Ca_Cl_GM89(T, P)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Ca"] * i2c["Cl"])))
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_Cl_GM89(T, P):
    """c-a: potassium chloride [GM89]."""
    b0 = GM89_eq3(
        T,
        [
            2.67375563e1,
            1.00721050e-2,
            -7.58485453e2,
            -4.70624175e00,
            0,
            -3.75994338e-6,
            0,
            0,
        ],
    )
    b1 = GM89_eq3(
        T,
        [
            -7.41559626e00,
            0,
            3.22892989e2,
            1.16438557e00,
            0,
            0,
            0,
            -5.94578140e00,
        ],
    )
    b2 = 0
    Cphi = GM89_eq3(
        T,
        [
            -3.30531334e00,
            -1.29807848e-3,
            9.12712100e1,
            5.86450181e-1,
            0,
            4.95713573e-7,
            0,
            0,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 523.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_SO4_GM89(T, P):
    """c-a: potassium sulfate [GM89]."""
    b0 = GM89_eq3(
        T,
        [
            4.07908797e1,
            8.26906675e-3,
            -1.41842998e3,
            -6.74728848e00,
            0,
            0,
            0,
            0,
        ],
    )
    b1 = GM89_eq3(
        T,
        [
            -1.31669651e1,
            2.35793239e-2,
            2.06712594e3,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    b2 = 0
    Cphi = -0.0188
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["SO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 523.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def theta_Ca_K_GM89(T, P):
    """c-c': calcium potassium [GM89]."""
    theta = 0.1156
    valid = (T >= 273.15) & (T <= 523.15)
    return theta, valid


def theta_K_Na_GM89(T, P):
    """c-c': potassium sodium [GM89]."""
    theta = GM89_eq3(
        T,
        [
            -5.02312111e-2,
            0,
            1.40213141e1,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    valid = (T >= 273.15) & (T <= 523.15)
    return theta, valid


def psi_Ca_K_Cl_GM89(T, P):
    """c-c'-a: calcium potassium chloride [GM89]."""
    psi = GM89_eq3(
        T,
        [
            4.76278977e-2,
            0,
            -2.70770507e1,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    valid = (T >= 273.15) & (T <= 523.15)
    return psi, valid


def psi_Ca_K_SO4_GM89(T, P):
    """c-c'-a: calcium potassium sulfate [GM89]."""
    psi = 0
    valid = (T >= 273.15) & (T <= 523.15)
    return psi, valid


def psi_K_Na_Cl_GM89(T, P):
    """c-c'-a: potassium sodium chloride [GM89]."""
    psi = GM89_eq3(
        T,
        [
            1.34211308e-2,
            0,
            -5.10212917e00,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    valid = (T >= 273.15) & (T <= 523.15)
    return psi, valid


def psi_K_Na_SO4_GM89(T, P):
    """c-c'-a: potassium sodium sulfate [GM89]."""
    psi = GM89_eq3(
        T,
        [
            3.48115174e-2,
            0,
            -8.21656777e00,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    valid = (T >= 273.15) & (T <= 423.15)
    return psi, valid


def psi_K_Cl_SO4_GM89(T, P):
    """c-a-a': potassium chloride sulfate [GM89]."""
    psi = GM89_eq3(
        T,
        [
            -2.12481475e-1,
            2.84698333e-4,
            3.75619614e1,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    valid = (T >= 273.15) & (T <= 523.15)
    return psi, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Hershey et al. (1989) ~~~~~
def bC_Mg_H2PO4_HFM89(T, P):
    """c-a: magnesium dihydrogen-phosphate [HFM89]."""
    b0 = -3.55
    b1 = 16.9
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mg_HPO4_HFM89(T, P):
    """c-a: magnesium hydrogen-phosphate [HFM89]."""
    b0 = -17.5
    b1 = 27.4
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 1.4
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def theta_Cl_H2PO4_HFM89(T, P):
    """a-a': chloride dihydrogen-phosphate [HFM89]."""
    theta = 0.10
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_Cl_H2PO4_HFM89(T, P):
    """c-a-a': sodium chloride dihydrogen-phosphate [HFM89]."""
    psi = -0.028
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_Cl_HPO4_HFM89(T, P):
    """a-a': chloride hydrogen-phosphate [HFM89]."""
    theta = -0.105
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_Cl_HPO4_HFM89(T, P):
    """c-a-a': sodium chloride hydrogen-phosphate [HFM89]."""
    psi = -0.003
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_Cl_PO4_HFM89(T, P):
    """a-a': chloride phosphate [HFM89]."""
    theta = -0.59
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_Cl_PO4_HFM89(T, P):
    """c-a-a': sodium chloride phosphate [HFM89]."""
    psi = 0.110
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def lambd_H3PO4_Na_HFM89(T, P):
    """n-c: phosphoric-acid sodium [HFM89]."""
    lambd = 0.075
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_H3PO4_Cl_HFM89(T, P):
    """n-a: phosphoric-acid chloride [HFM89]."""
    lambd = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_MgHPO4_Na_HFM89(T, P):
    """n-n': magnesium-hydrogen-phosphate sodium [HFM89]."""
    lambd = -0.124
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Millero et al. (1989) ~~~~~
def bC_Na_SO3_MHJZ89(T, P):
    """c-a: sodium sulfite [MHJZ89]."""
    b0 = 5.88444 - 1730.55 / T  # Eq. (36)
    b1 = -19.4549 + 6153.78 / T  # Eq. (37)
    b2 = 0
    Cphi = -1.2355 + 367.07 / T  # Eq. (38)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["SO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_HSO3_MHJZ89(T, P):
    """c-a: sodium hydrogen-sulfite [MHJZ89]."""
    b0 = 4.3407 - 1248.66 / T  # Eq. (29)
    b1 = -13.146 + 4014.80 / T  # Eq. (30)
    b2 = 0
    Cphi = 0.9565 + 277.85 / T  # Eq. (31), note difference from MP98 Table A3
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["HSO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def theta_Cl_SO3_MHJZ89(T, P):
    """a-a': chloride sulfite [MHJZ89]."""
    theta = 0.099  # +/- 0.004
    valid = (T >= 273.15) & (T <= 323.15)
    return theta, valid


def psi_Na_Cl_SO3_MHJZ89(T, P):
    """c-a-a': sodium chloride sulfite [MHJZ89]."""
    psi = -0.0156  # +/- 0.001
    valid = (T >= 273.15) & (T <= 323.15)
    return psi, valid


def lambd_SO2_Na_MHJZ89(T, P):
    """n-c: sulfur-dioxide sodium [MHJZ89]."""
    # RZM93 and MP98 both cite MHJZ89 but can't find this value therein
    lambd = 0.0283
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_SO2_Cl_MHJZ89(T, P):
    """n-a: sulfur-dioxide chloride [MHJZ89]."""
    # RZM93 and MP98 both cite MHJZ89 but can't find this value therein
    lambd = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Roy et al. (1991) ~~~~~
def bC_Mg_HSO3_RZM91(T, P):
    """c-a: magnesium bisulfite [RZM91]."""
    b0 = 0.35
    b1 = 1.22
    b2 = 0
    Cphi = -0.072
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Mg"] * i2c["HSO3"])))
    C1 = 0
    alph1 = 2.0
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mg_SO3_RZM91(T, P):
    """c-a: magnesium sulfite [RZM91]."""
    b0 = -2.8
    b1 = 12.9
    b2 = -201
    C0 = 0
    C1 = 0
    alph1 = 1.4
    alph2 = 12
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def lambd_SO2_Mg_RZM91(T, P):
    """n-c: sulfur-dioxide magnesium [RZM91]."""
    lambd = 0.085
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Archer (1992) ~~~~~
def A92ii_eq36(T, P, a):
    """A92ii equation 36, with pressure in MPa."""
    # a[5] and a[6] multipliers are corrected for typos in A92ii
    return (
        a[0]
        + a[1] * 10 ** -3 * T
        + a[2] * 4e-6 * T ** 2
        + a[3] * 1 / (T - 200)
        + a[4] * 1 / T
        + a[5] * 100 / (T - 200) ** 2
        + a[6] * 200 / T ** 2
        + a[7] * 8e-9 * T ** 3
        + a[8] * 1 / (650 - T) ** 0.5
        + a[9] * 10 ** -5 * P
        + a[10] * 2e-4 * P / (T - 225)
        + a[11] * 100 * P / (650 - T) ** 3
        + a[12] * 2e-8 * P * T
        + a[13] * 2e-4 * P / (650 - T)
        + a[14] * 10 ** -7 * P ** 2
        + a[15] * 2e-6 * P ** 2 / (T - 225)
        + a[16] * P ** 2 / (650 - T) ** 3
        + a[17] * 2e-10 * P ** 2 * T
        + a[18] * 4e-13 * P ** 2 * T ** 2
        + a[19] * 0.04 * P / (T - 225) ** 2
        + a[20] * 4e-11 * P * T ** 2
        + a[21] * 2e-8 * P ** 3 / (T - 225)
        + a[22] * 0.01 * P ** 3 / (650 - T) ** 3
        + a[23] * 200 / (650 - T) ** 3
    )


def bC_Na_Cl_A92ii(T, P):
    """c-a: sodium chloride [A92ii]."""
    P_MPa = P / 100  # Convert dbar to MPa
    # Parameters from A92ii Table 2, with noted corrections
    b0 = A92ii_eq36(
        T,
        P_MPa,
        [
            0.242408292826506,
            0,
            -0.162683350691532,
            1.38092472558595,
            0,
            0,
            -67.2829389568145,
            0,
            0.625057580755179,
            -21.2229227815693,
            81.8424235648693,
            -1.59406444547912,
            0,
            0,
            28.6950512789644,
            -44.3370250373270,
            1.92540008303069,
            -32.7614200872551,
            0,
            0,
            30.9810098813807,
            2.46955572958185,
            -0.725462987197141,
            10.1525038212526,
        ],
    )
    b1 = A92ii_eq36(
        T,
        P_MPa,
        [
            -1.90196616618343,
            5.45706235080812,
            0,
            -40.5376417191367,
            0,
            0,
            4.85065273169753 * 1e2,
            -0.661657744698137,
            0,
            0,
            2.42206192927009 * 1e2,
            0,
            -99.0388993875343,
            0,
            0,
            -59.5815563506284,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    b2 = 0
    C0 = A92ii_eq36(
        T,
        P_MPa,
        [
            0,
            -0.0412678780636594,
            0.0193288071168756,
            -0.338020294958017,  # typo in A92ii
            0,
            0.0426735015911910,
            4.14522615601883,
            -0.00296587329276653,
            0,
            1.39697497853107,
            -3.80140519885645,
            0.06622025084,  # typo in A92ii - "Rard's letter"
            0,
            -16.8888941636379,
            -2.49300473562086,
            3.14339757137651,
            0,
            2.79586652877114,
            0,
            0,
            0,
            0,
            0,
            -0.502708980699711,
        ],
    )
    C1 = A92ii_eq36(
        T,
        P_MPa,
        [
            0.788987974218570,
            -3.67121085194744,
            1.12604294979204,
            0,
            0,
            -10.1089172644722,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            16.6503495528290,
        ],
    )
    alph1 = 2
    alph2 = -9
    omega = 2.5
    valid = (T >= 250) & (T <= 600) & (P_MPa <= 100)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Campbell et al. (1993) ~~~~~
def CMR93_eq31(T, a):
    """CMR93 equation 31."""
    return M88_eq13(T, a)


def bC_H_Cl_CMR93(T, P):
    """c-a: hydrogen chloride [CMR93]."""
    # b0 a[1] term corrected here for typo, following WM13
    b0 = CMR93_eq31(
        T,
        [
            1.2859,
            -2.1197e-3,
            -142.5877,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    b1 = CMR93_eq31(
        T,
        [
            -4.4474,
            8.425698e-3,
            665.7882,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    b2 = 0
    Cphi = CMR93_eq31(
        T,
        [
            -0.305156,
            5.16e-4,
            45.52154,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["H"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 328.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def theta_H_K_CMR93(T, P):
    """c-c': hydrogen potassium [CMR93]."""
    # assuming CMR93's lowercase t means temperature in degC
    theta = 0.005 - 0.0002275 * (T - Tzero)
    valid = (T >= 273.15) & (T <= 328.15)
    return theta, valid


def theta_H_Na_CMR93(T, P):
    """c-c': hydrogen sodium [CMR93]."""
    # assuming CMR93's lowercase t means temperature in degC
    theta = 0.0342 - 0.000209 * (T - Tzero)
    valid = (T >= 273.15) & (T <= 328.15)
    return theta, valid


def psi_H_K_Cl_CMR93(T, P):
    """c-c'-a: hydrogen potassium chloride [CMR93]."""
    psi = 0
    valid = (T >= 273.15) & (T <= 523.15)
    return psi, valid


def psi_H_Na_Cl_CMR93(T, P):
    """c-c'-a: hydrogen sodium chloride [CMR93]."""
    psi = 0
    valid = (T >= 273.15) & (T <= 523.15)
    return psi, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ He & Morse (1993) ~~~~~
# Note that HM93 also contains beta/C equations for Na, K, Mg and Ca
# interactions with HCO3 and CO3 (not yet coded here)
def HM93_eq(T, A, B, C, D, E):
    """HM93 parameter equation from p. 3548."""
    return A + B * T + C * T ** 2 + D / T + E * np.log(T)


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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Hovey et al. (1993) ~~~~~
def HPR93_eq36(T, a):
    """HPR93 equation 36."""
    Tref = 298.15
    return a[0] + a[1] * (1 / T - 1 / Tref) + a[2] * np.log(T / Tref)


def bC_Na_SO4_HPR93(T, P):
    """c-a: sodium sulfate [HPR93]."""
    b0 = HPR93_eq36(
        T,
        [
            0.006536438,
            -30.197349,
            -0.20084955,
        ],
    )
    b1 = HPR93_eq36(
        T,
        [
            0.87426420,
            -70.014123,
            0.2962095,
        ],
    )
    b2 = 0
    Cphi = HPR93_eq36(
        T,
        [
            0.007693706,
            4.5879201,
            0.019471746,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["SO4"])))
    C1 = 0
    alph1 = 1.7
    alph2 = -9
    omega = -9
    valid = (T >= 273.0) & (T <= 373.0)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_HSO4_HPR93(T, P):
    """c-a: sodium bisulfate, low ionic strengths [HPR93]."""
    # Parameters from HPR93 Table 3 for low ionic strengths
    b0 = 0.0670967
    b1 = 0.3826401
    b2 = 0
    Cphi = -0.0039056
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["HSO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Clegg et al. (1994) ~~~~~
CRP94_Tr = 328.15  # K


def CRP94_eq24(T, q):
    return q[0] + 1e-3 * (
        (T - CRP94_Tr) * q[1]
        + (T - CRP94_Tr) ** 2 * q[2] / 2
        + (T - CRP94_Tr) ** 3 * q[3] / 6
    )


def bC_H_HSO4_CRP94(T, P):
    """c-a: hydrogen bisulfate [CRP94]."""
    # Parameters from CRP94 Table 6
    b0 = CRP94_eq24(
        T,
        [
            0.227784933,
            -3.78667718,
            -0.124645729,
            -0.00235747806,
        ],
    )
    b1 = CRP94_eq24(
        T,
        [
            0.372293409,
            1.50,
            0.207494846,
            0.00448526492,
        ],
    )
    b2 = 0
    C0 = CRP94_eq24(
        T,
        [
            -0.00280032520,
            0.216200279,
            0.0101500824,
            0.000208682230,
        ],
    )
    C1 = CRP94_eq24(
        T,
        [
            -0.025,
            18.1728946,
            0.382383535,
            0.0025,
        ],
    )
    alph1 = 2
    alph2 = -9
    omega = 2.5
    valid = (T >= 273.15) & (T <= 328.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_H_SO4_CRP94(T, P):
    """c-a: hydrogen sulfate [CRP94]."""
    # Evaluate parameters from CRP94 Table 6
    b0 = CRP94_eq24(
        T,
        [
            0.0348925351,
            4.97207803,
            0.317555182,
            0.00822580341,
        ],
    )
    b1 = CRP94_eq24(
        T,
        [
            -1.06641231,
            -74.6840429,
            -2.26268944,
            -0.0352968547,
        ],
    )
    b2 = 0
    C0 = CRP94_eq24(
        T,
        [
            0.00764778951,
            -0.314698817,
            -0.0211926525,
            -0.000586708222,
        ],
    )
    C1 = CRP94_eq24(
        T,
        [
            0,
            -0.176776695,
            -0.731035345,
            0,
        ],
    )
    alph1 = 2 - 1842.843 * (1 / T - 1 / 298.15)
    alph2 = -9
    omega = 2.5
    valid = (T >= 273.15) & (T <= 328.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def theta_HSO4_SO4_CRP94(T, P):
    """a-a': bisulfate sulfate [CRP94]."""
    theta = 0
    valid = (T >= 273.15) & (T <= 328.15)
    return theta, valid


def psi_H_HSO4_SO4_CRP94(T, P):
    """c-a-a': hydrogen bisulfate sulfate [CRP94]."""
    psi = 0
    valid = (T >= 273.15) & (T <= 328.15)
    return psi, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pierrot et al. (1997) ~~~~~
def bC_Na_HSO4_PMR97(T, P):
    """c-a: sodium bisulfate [PMR97]."""
    b0 = 0.030101 - 0.362e-3 * (T - 298.15)  # Eq. (26)
    b1 = 0.818686 - 0.019671 * (T - 298.15)  # Eq. (27)
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 278.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def psi_H_Na_Cl_PMR97(T, P):
    """c-c'-a: hydrogen sodium chloride [PMR97]."""
    psi = 0.0002
    valid = (T >= 273.15) & (T <= 323.15)
    return psi, valid


def psi_H_Na_SO4_PMR97(T, P):
    """c-c'-a: hydrogen sodium sulfate [PMR97]."""
    psi = 0
    valid = (T >= 273.15) & (T <= 323.15)
    return psi, valid


def psi_H_Na_HSO4_PMR97(T, P):
    """c-c'-a: hydrogen sodium bisulfate [PMR97]."""
    psi = 0
    valid = (T >= 273.15) & (T <= 323.15)
    return psi, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Millero and Pierrot (1998) ~~~~~
def MP98_eq15(T, q):
    # Note typo in location of first BJ w.r.t. brackets in MP98 vs PM16 model
    # Here we use the PM16 model equation
    Tr = 298.15
    BR = q[0]
    BJ = q[1] * 1e-5
    BLR = q[2] * 1e-4
    return (
        BR
        + (BJ * (Tr ** 3 / 3) - Tr ** 2 * BLR) * (1 / T - 1 / Tr)
        + (BJ / 6) * (T ** 2 - Tr ** 2)
    )


def bC_Na_I_MP98(T, P):
    """c-a: sodium iodide [MP98]."""
    b0 = MP98_eq15(
        T,
        [
            0.1195,
            -1.01,
            8.355,
        ],
    )
    b1 = MP98_eq15(
        T,
        [
            0.3439,
            -2.54,
            8.28,
        ],
    )
    b2 = 0
    Cphi = MP98_eq15(
        T,
        [
            0.0018,
            0,
            -0.835,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["I"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_Br_MP98(T, P):
    """c-a: sodium bromide [MP98]."""
    b0 = MP98_eq15(
        T,
        [
            0.0973,
            -1.3,
            7.692,
        ],
    )
    b1 = MP98_eq15(
        T,
        [
            0.2791,
            -1.06,
            10.79,
        ],
    )
    b2 = 0
    Cphi = MP98_eq15(
        T,
        [
            0.00116,
            0.058 * 2 ** 1.5,
            -0.93,
        ],  # more accurate following PM16 model
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["Br"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_F_MP98(T, P):
    """c-a: sodium fluoride [MP98]."""
    b0 = MP98_eq15(
        T,
        [
            0.0215,  # typo in MP98 Table A7 vs PM16 code
            -2.37,
            5.361,  # *1e-4 in MP98 Table A7 presumably a typo vs PM16 code
        ],
    )
    b1 = MP98_eq15(
        T,
        [
            0.2107,
            0,
            8.7,
        ],  # *1e-4 in MP98 Table A7 presumably a typo vs PM16 code
    )
    b2 = 0
    Cphi = MP98_eq15(
        T,
        [
            0,
            0,
            -0.93,
        ],  # 0 in MP98 Table A7 presumably a typo vs PM16 code
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["F"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_Br_MP98(T, P):
    """c-a: potassium bromide [MP98]."""
    b0 = MP98_eq15(
        T,
        [
            0.0569,
            -1.43,
            7.39,
        ],
    )
    b1 = MP98_eq15(
        T,
        [
            0.2122,
            -0.762,
            1.74,
        ],
    )
    b2 = 0
    Cphi = MP98_eq15(
        T,
        [
            -0.0018,
            0.216,
            -0.7004,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["Br"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_F_MP98(T, P):
    """c-a: potassium fluoride [MP98]."""
    b0 = MP98_eq15(
        T,
        [
            0.08089,
            -1.39,
            2.14,
        ],
    )
    b1 = MP98_eq15(
        T,
        [
            0.2021,
            0,
            5.44,
        ],
    )
    b2 = 0
    Cphi = MP98_eq15(
        T,
        [
            0.00093,
            0,
            -0.595,  # typo in MP98 table vs PM16 model (latter has -, former +)
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["F"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_OH_MP98(T, P):
    """c-a: potassium hydroxide [MP98]."""
    b0 = MP98_eq15(
        T,
        [
            0.1298,
            -0.946,
            9.914,
        ],
    )  # copy of KI
    b1 = MP98_eq15(
        T,
        [
            0.32,
            -2.59,
            11.86,
        ],
    )  # copy of KI
    b2 = 0
    Cphi = MP98_eq15(
        T,
        [
            0.0041,
            0.0638,
            -0.944,
        ],
    )  # copy of KI
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["OH"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_I_MP98(T, P):
    """c-a: potassium iodide [MP98]."""
    b0 = MP98_eq15(
        T,
        [
            0.0746,
            -0.748,
            9.914,
        ],
    )
    b1 = MP98_eq15(
        T,
        [
            0.2517,
            -1.8,
            11.86,
        ],
    )
    b2 = 0
    Cphi = MP98_eq15(
        T,
        [
            -0.00414,
            0,
            -0.944,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["I"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_ClO3_MP98(T, P):
    """c-a: sodium chlorate [MP98]."""
    b0 = MP98_eq15(
        T,
        [
            0.0249,
            -1.56,
            10.35,
        ],
    )
    b1 = MP98_eq15(
        T,
        [
            0.2455,
            -2.69,
            19.07,
        ],
    )
    b2 = 0
    Cphi = MP98_eq15(
        T,
        [
            0.0004,
            0.222,
            9.29,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["ClO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_ClO3_MP98(T, P):
    """c-a: potassium chlorate [MP98]."""
    b0 = MP98_eq15(
        T,
        [
            -0.096,
            15.1,
            19.87,
        ],
    )
    b1 = MP98_eq15(
        T,
        [
            0.2841,
            -27,
            31.8,
        ],
    )
    b2 = 0
    Cphi = MP98_eq15(
        T,
        [
            0,
            -19.1,
            0,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["ClO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_ClO4_MP98(T, P):
    """c-a: sodium perchlorate [MP98]."""
    b0 = MP98_eq15(
        T,
        [
            0.0554,
            -0.611,
            12.96,
        ],
    )
    b1 = MP98_eq15(
        T,
        [
            0.2755,
            -6.35,
            22.97,
        ],
    )
    b2 = 0
    Cphi = MP98_eq15(
        T,
        [
            -0.00118,
            0.0562,
            -1.623,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["ClO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_BrO3_MP98(T, P):
    """c-a: sodium bromate [MP98]."""
    b0 = MP98_eq15(
        T,
        [
            -0.0205,
            -6.5,
            5.59,
        ],
    )
    b1 = MP98_eq15(
        T,
        [
            0.191,
            5.45,
            34.37,
        ],
    )
    b2 = 0
    Cphi = MP98_eq15(
        T,
        [
            0.0059,
            2.5,
            0,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["BrO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_BrO3_MP98(T, P):
    """c-a: potassium bromate [MP98]."""
    b0 = MP98_eq15(
        T,
        [
            -0.129,
            9.17,
            5.59,
        ],
    )
    b1 = MP98_eq15(
        T,
        [
            0.2565,
            -20.2,
            34.37,
        ],
    )
    b2 = 0
    Cphi = MP98_eq15(
        T,
        [
            0,
            -26.6,
            0,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["BrO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_NO3_MP98(T, P):
    """c-a: sodium nitrate [MP98]."""
    b0 = MP98_eq15(
        T,
        [
            0.0068,
            -2.24,
            12.66,
        ],
    )
    b1 = MP98_eq15(
        T,
        [
            0.1783,
            -2.96,
            20.6,
        ],
    )
    b2 = 0
    Cphi = MP98_eq15(
        T,
        [
            -0.00072,
            0.594,
            -2.316,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_NO3_MP98(T, P):
    """c-a: potassium nitrate [MP98]."""
    b0 = MP98_eq15(
        T,
        [
            -0.0816,
            -0.785,
            2.06,
        ],
    )
    b1 = MP98_eq15(
        T,
        [
            0.0494,
            -8.26,
            64.5,
        ],
    )
    b2 = 0
    Cphi = MP98_eq15(
        T,
        [
            0.0066,
            0,
            3.97,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mg_NO3_MP98(T, P):
    """c-a: magnesium nitrate [MP98]."""
    b0 = MP98_eq15(
        T,
        [
            0.367125,
            -1.2322,
            5.15,
        ],
    )
    b1 = MP98_eq15(
        T,
        [
            1.58475,
            4.0492,
            44.925,
        ],
    )
    b2 = 0
    Cphi = MP98_eq15(
        T,
        [
            -0.020625,
            0,
            0,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Mg"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ca_NO3_MP98(T, P):
    """c-a: calcium nitrate [MP98]."""
    b0 = MP98_eq15(
        T,
        [
            0.210825,
            4.0248,
            5.295,
        ],
    )
    b1 = MP98_eq15(
        T,
        [
            1.40925,
            -13.289,
            91.875,
        ],
    )
    b2 = 0
    Cphi = MP98_eq15(
        T,
        [
            -0.020142,
            -15.435,
            0,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Ca"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_H_Br_MP98(T, P):
    """c-a: hydrogen bromide [MP98]."""
    b0 = MP98_eq15(
        T,
        [
            0.196,
            -0.357,
            -2.049,
        ],
    )
    b1 = MP98_eq15(
        T,
        [
            0.3564,
            -0.913,
            4.467,
        ],
    )
    b2 = 0
    Cphi = MP98_eq15(
        T,
        [
            0.00827,
            0.01272,
            -0.5685,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["H"] * i2c["Br"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Sr_Cl_MP98(T, P):
    """c-a: strontium chloride [MP98]."""
    b0 = MP98_eq15(
        T,
        [
            0.28575,
            -0.18367,
            9.56 * 3 / 4,
        ],  # from Pierrot_2018_Interaction_Model.xlsm
    )
    b1 = MP98_eq15(
        T,
        [
            1.66725,
            0,
            37.9 * 3 / 4,
        ],
    )
    b2 = 0
    Cphi = MP98_eq15(
        T,
        [
            -0.0013,
            0,
            0,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Sr"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_NH4_Cl_MP98(T, P):
    """c-a: ammonium chloride [MP98]."""
    b0 = MP98_eq15(
        T,
        [
            0.0522,
            -0.597,
            0.779,
        ],
    )
    b1 = MP98_eq15(
        T,
        [
            0.1918,
            0.444,
            12.58,
        ],
    )
    b2 = 0
    Cphi = MP98_eq15(
        T,
        [
            -0.00301,
            0.0578,
            0.21,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["NH4"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_NH4_Br_MP98(T, P):
    """c-a: ammonium bromide [MP98]."""
    b0 = MP98_eq15(
        T,
        [
            0.0624,
            -0.597,
            0.779,
        ],
    )
    b1 = MP98_eq15(
        T,
        [
            0.1947,
            0,
            12.58,
        ],
    )
    b2 = 0
    Cphi = MP98_eq15(
        T,
        [
            -0.00436,
            0,
            0.21,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["NH4"] * i2c["Br"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_NH4_Br_MP98typo(T, P):
    """c-a: ammonium bromide [MP98typo]."""
    # PM16 model is missing 1e-5 multiplier on final Cphi term
    b0 = MP98_eq15(
        T,
        [
            0.0624,
            -0.597,
            0.779,
        ],
    )
    b1 = MP98_eq15(
        T,
        [
            0.1947,
            0,
            12.58,
        ],
    )
    b2 = 0
    Cphi = MP98_eq15(
        T,
        [
            -0.00436,
            0,
            0.21 * 1e5,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["NH4"] * i2c["Br"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_NH4_F_MP98(T, P):
    """c-a: ammonium fluoride [MP98]."""
    b0 = MP98_eq15(
        T,
        [
            0.1306,
            1.09,
            0.95,
        ],
    )
    b1 = MP98_eq15(
        T,
        [
            0.257,
            0,
            5.97,
        ],
    )
    b2 = 0
    Cphi = MP98_eq15(
        T,
        [
            -0.0043,
            0,
            0,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["NH4"] * i2c["F"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def MP98_eqTableA3(T, abc):
    """MP98 equation for Table A3."""
    Tr = 298.15
    return abc[0] + abc[1] * (T - Tr) + abc[2] * (T - Tr) ** 2


def bC_Na_HSO4_MP98(T, P):
    """c-a: sodium bisulfate [MP98]."""
    # MP98 cite Pierrot et al. (1997) J Solution Chem 26(1),
    #  but their equations look quite different, and there is no Cphi there.
    # This equation is therefore directly from MP98.
    b0 = MP98_eqTableA3(
        T,
        [
            0.0544,
            -1.8478e-3,
            5.3937e-5,
        ],
    )  # MP98 typo vs PM16 model
    b1 = MP98_eqTableA3(
        T,
        [
            0.3826401,
            -1.8431e-2,
            0,
        ],
    )
    b2 = 0
    Cphi = -0.0039056  # from PM16 code this should be negative (MP98 typo)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["HSO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ca_SO3_MP98(T, P):
    """c-a: calcium sulfite [MP98]."""
    return bC_Ca_SO4_M88(T, P)


def bC_Sr_SO4_MP98(T, P):
    """c-a: strontium sulfate [MP98]."""
    return bC_Ca_SO4_M88(T, P)


def bC_Sr_BOH4_MP98(T, P):
    """c-a: strontium borate [MP98]."""
    return bC_Ca_BOH4_SRM87(T, P)


def bC_Ca_HSO3_MP98(T, P):
    """c-a: calcium sulfite [MP98]."""
    return bC_Ca_HSO4_HMW84(T, P)


def bC_Sr_HSO4_MP98(T, P):
    """c-a: strontium bisulfate [MP98]."""
    return bC_Ca_HSO4_HMW84(T, P)


def bC_Sr_HCO3_MP98(T, P):
    """c-a: strontium bicarbonate [MP98]."""
    return bC_Ca_HCO3_HMW84(T, P)


def bC_Sr_HSO3_MP98(T, P):
    """c-a: strontium sulfite [MP98]."""
    return bC_Ca_HSO3_MP98(T, P)


def bC_Sr_OH_MP98(T, P):
    """c-a: strontium hydroxide [MP98]."""
    return bC_Ca_OH_HMW84(T, P)


def theta_Cl_F_MP98(T, P):
    """a-a': chloride fluoride [MP98]."""
    # MP98 state value is "determined from CB88 data"
    theta = 0.01
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Na_Cl_F_MP98(T, P):
    """c-a-a': sodium chloride fluoride [MP98]."""
    # MP98 state value is "determined from CB88 data"
    psi = 0.0023
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def theta_CO3_HCO3_MP98(T, P):
    """a-a': carbonate bicarbonate [MP98]."""
    theta = 0
    valid = (T >= 273.15) & (T <= 333.15)
    return theta, valid


def psi_Na_CO3_HCO3_MP98(T, P):
    """c-a-a': sodium carbonate bicarbonate [MP98]."""
    psi = 0
    valid = (T >= 273.15) & (T <= 333.15)
    return psi, valid


def psi_K_CO3_HCO3_MP98(T, P):
    """c-a-a': potassium carbonate bicarbonate [MP98]."""
    psi = 0
    valid = (T >= 273.15) & (T <= 333.15)
    return psi, valid


def psi_Na_BOH4_Cl_MP98(T, P):
    """c-a-a': sodium borate chloride [MP98]."""
    # MP98 say "determined from OK43 and Hershey et al. (1986b) data"
    psi = -0.0132
    valid = (T >= 273.15) & (T <= 318.15)
    return psi, valid


def psi_Mg_BOH4_Cl_MP98(T, P):
    """c-a-a': magnesium borate chloride [MP98]."""
    # MP98 say "determined from Hershey et al. (1986b) and Simonson et al.
    #     (1987b) data"
    psi = -0.235
    valid = (T >= 273.15) & (T <= 318.15)
    return psi, valid


def psi_Ca_BOH4_Cl_MP98(T, P):
    """c-a-a': calcium borate chloride [MP98]."""
    # MP98 say "determined from Hershey et al. (1986b) and Simonson et al.
    #     (1987b) data"
    psi = -0.8
    valid = (T >= 273.15) & (T <= 318.15)
    return psi, valid


def theta_Cl_OH_MP98(T, P):
    """a-a': chloride hydroxide [MP98]."""
    # MP98 say "determined from HO58 data"
    theta = -0.05 + (T - 298.15) * 3.125e-4 + (T - 298.15) ** 2 * -8.362e-6
    # MP98 don't give validity range
    valid = (T >= 273.15) & (T <= 323.15)
    return theta, valid


def theta_BOH4_Cl_MP98(T, P):
    """a-a': borate chloride [MP98]."""
    # MP98 say "determined from OK43 and Hershey et al. (1986b) data"
    theta = -0.0323 + (T - 298.15) * -0.42333e-4 + (T - 298.15) ** 2 * -21.926e-6
    valid = (T >= 273.15) & (T <= 318.15)
    return theta, valid


def theta_BOH4_Cl_MP98typo(T, P):
    """a-a': borate chloride [MP98typo]."""
    # This replicates a typo in the Pierrot_2018_Interaction_Model.xlsm code
    #   (298.25 instead of 298.15 in the T**2 term).
    # For proper modelling, should use theta_BOH4_Cl_MP98 instead.
    theta = -0.0323 + (T - 298.15) * -0.42333e-4 + (T - 298.25) ** 2 * -21.926e-6
    valid = (T >= 273.15) & (T <= 318.15)
    return theta, valid


def psi_Ca_K_Cl_MP98typo(T, P):
    """c-c'-a: calcium potassium chloride [MP98typo]."""
    # This replicates a typo in the Pierrot_2018_Interaction_Model.xlsm code
    #   (in the first term).
    # For proper modelling, should use psi_Ca_K_Cl_GM89 instead.
    psi = 0.047627877 - 27.0770507 / T
    valid = (T >= 273.15) & (T <= 523.15)
    return psi, valid


def bC_Mg_HCO3_MP98(T, P):
    """c-a: magnesium bicarbonate [MP98]."""
    # MP98 say "re-determined from TM82."
    b0 = 0.03
    b1 = 0.8
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def MP98_eqTableA9_2(T, a):
    """MP98 equation (2) from Table A9."""
    return a[0] + (T - 328.15) * 1e-3 * (
        a[1] + (T - 328.15) * (a[2] / 2 + (T - 328.15) * a[3] / 6)
    )


def bC_H_SO4_MP98(T, P):
    """c-a: hydrogen sulfate [MP98]."""
    # MP98 cite Pierrot et al. (1998, submitted), but that doesn't appear to
    # have been published, so these values are directly from MP98 Table A9
    # (with noted typo corrections).
    b0 = MP98_eqTableA9_2(
        T,
        [
            0.0065,
            0.134945,
            0.022374,
            7.2e-5,
        ],  # typo in MP98 table (vs 2016 VB model)
    )
    b1 = MP98_eqTableA9_2(
        T,
        [
            -0.15009,  # typo in MP98 table (vs 2016 VB model)
            -2.405945,
            0.335839,
            -0.004379,
        ],
    )
    b2 = 0
    C0 = MP98_eqTableA9_2(
        T,
        [
            0.008073,
            -0.113106,
            -0.003553,
            3.57e-5,
        ],
    )
    C1 = MP98_eqTableA9_2(
        T,
        [
            -0.050799,
            3.472545,
            -0.311463,
            0.004037,
        ],
    )
    alph1 = 2
    alph2 = -9
    omega = 2.5
    valid = (T >= 273.15) & (T <= 523.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_H_HSO4_MP98(T, P):
    """c-a: hydrogen bisulfate [MP98]."""
    # MP98 cite Pierrot et al. (1998, submitted), but that doesn't appear to
    # have been published, so these values are directly from the MP98 VB model.
    # HOWEVER, this interaction is mysteriously commented out in that model...
    b0 = MP98_eqTableA9_2(
        T,
        [
            0.239786,
            -1.209539,
            -0.004345,
            0.000185,
        ],
    )
    b1 = MP98_eqTableA9_2(
        T,
        [
            0.352843,
            0.087537,
            0.038379,
            -0.000744,
        ],
    )
    b2 = 0
    C0 = MP98_eqTableA9_2(
        T,
        [
            -0.003234,
            0.045792,
            0.000793,
            -0.0000222,
        ],
    )
    C1 = MP98_eqTableA9_2(
        T,
        [
            -0.098973,
            3.127618,
            -0.020184,
            -0.000316,
        ],
    )
    alph1 = 2
    alph2 = -9
    omega = 2.5
    valid = (T >= 273.15) & (T <= 523.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def theta_HSO4_SO4_MP98(T, P):
    """a-a': bisulfate sulfate [MP98]."""
    # MP98 cite Pierrot et al. (1998, submitted), but that doesn't appear to
    # have been published, so these values are directly from MP98.
    theta = 0
    valid = (T >= 273.15) & (T <= 473.15)
    return theta, valid


def psi_Na_HSO4_SO4_MP98(T, P):
    """c-a-a': sodium bisulfate sulfate [MP98]."""
    # MP98 cite Pierrot et al. (1998, submitted), but that doesn't appear to
    # have been published, so these values are directly from MP98.
    psi = 0
    valid = (T >= 273.15) & (T <= 473.15)
    return psi, valid


def theta_Na_Sr_MP98(T, P):
    """c-c': sodium strontium [MP98]."""
    # MP98 cite PK74 but I can't find this value in there
    theta = 0.07
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_K_Sr_MP98(T, P):
    """c-c': potassium strontium [MP98]."""
    # MP98 say this is set equal to the sodium value (i.e. theta_Na_Sr_MP98?)
    # but then state a different number (0.01)... 0.07 is used in the program
    # Pierrot_2018_Interaction_Model.xlsm
    theta = 0.07
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_H_Sr_Cl_MP98(T, P):
    """c-c'-a: hydrogen strontium chloride [MP98]."""
    # MP98 cite M85 book but can't find it there so this is from MP98 Table A10
    psi = 0.0054 - 2.1e-4 * (T - 298.15)
    valid = (T >= 273.15) & (T <= 323.15)
    return psi, valid


def psi_Na_Sr_Cl_MP98(T, P):
    """c-c'-a: sodium strontium chloride [MP98]."""
    # MP98 cite PK74 but I can't find this value in there
    psi = -0.015
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Sr_Cl_MP98(T, P):
    """c-c'-a: potassium strontium chloride [MP98]."""
    return psi_Na_Sr_Cl_MP98(T, P)


def psi_H_K_Br_MP98(T, P):
    """c-c'-a: hydrogen potassium bromide [MP98]."""
    # MP98 cite HMW84 but I can't find this value in there
    psi = -0.021
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Mg_Br_MP98(T, P):
    """c-c'-a: hydrogen magnesium bromide [MP98]."""
    # MP98 cite PK74 but I can't find this value in there
    psi = -0.005
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_K_Cl_H2PO4_MP98(T, P):
    """c-a-a': potassium chloride dihydrogen-phosphate [MP98]."""
    # MP98 cite Pitzer & Silvester (1976) but I can't find that paper
    psi = -0.0105
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def lambd_HF_Cl_MP98(T, P):
    """n-a: hydrogen-fluoride chloride [MP98]."""
    # MP98 Table A12 says this is derived "from CB88 data"
    lambd = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_HF_Na_MP98(T, P):
    """n-c: hydrogen-fluoride sodium [MP98]."""
    # MP98 Table A12 says this is derived "from CB88 data"
    lambd = 0.011
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def zeta_H3PO4_Na_Cl_MP98(T, P):
    """phosphoric-acid sodium chloride [MP98]."""
    # MP98 say this comes from PS76 but there's no sodium in there
    zeta = 0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return zeta, valid


def theta_H_K_MP98(T, P):
    """c-c': hydrogen potassium [MP98]."""
    # Direct from Pierrot_2018_Interaction_Model.xlsm, conflicts with CMR93
    theta = 0.005 - 0.0002275 * (T - 298.15)
    valid = (T >= 273.15) & (T <= 328.15)
    return theta, valid


def theta_H_Na_MP98(T, P):
    """c-c': hydrogen sodium [MP98]."""
    # Direct from Pierrot_2018_Interaction_Model.xlsm, conflicts with CMR93
    theta = 0.03416 - 0.000209 * (T - Tzero)
    valid = (T >= 273.15) & (T <= 328.15)
    return theta, valid


def bC_K_CO3_MP98(T, P):
    """c-a: potassium carbonate [MP98]."""
    # Direct from Pierrot_2018_Interaction_Model.xlsm, conflicts with SRG87
    b0 = 0.1288 + 1.1e-3 * (T - 298.15) - 5.1e-6 * (T - 298.15) ** 2
    b1 = 1.433 + 4.36e-3 * (T - 298.15) + 2.07e-5 * (T - 298.15) ** 2
    b2 = 0
    Cphi = 0.0005  # not in SRG87!
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["CO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 278.15) & (T <= 368.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def theta_H_Mg_MP98(T, P):
    """c-c': hydrogen magnesium [MP98]."""
    # RGB80 has no temperature term, as declared by MP98
    theta = 0.0620 + 0.0003275 * (T - 298.15)
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_H_Mg_Cl_MP98(T, P):
    """c-c': hydrogen magnesium chloride [MP98]."""
    # RGB80 has no temperature term, as declared by MP98
    theta = 0.001 - 0.0007325 * (T - 298.15)
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_Ca_H_MP98(T, P):
    """c-c': calcium hydrogen [MP98]."""
    # MP98 have really messed this one up? (see notes on theta_Ca_H_RGO81)
    theta = 0.0612 + 0.0003275 * (T - 298.15)
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Ca_H_Cl_MP98(T, P):
    """c-c': calcium hydrogen chloride [MP98]."""
    # RGO81 has no temperature term, as declared by MP98
    theta = 0.0008 - 0.000725 * (T - 298.15)
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_K_Cl_SO4_MP98(T, P):
    """c-a-a': potassium chloride sulfate [MP98]."""
    # MP98 say this is GM89 but don't use full precision for the first
    #   constant in the PM program
    psi = GM89_eq3(
        T,
        [
            -2.12481e-1,
            2.84698333e-4,
            3.75619614e1,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    valid = (T >= 273.15) & (T <= 523.15)
    return psi, valid


def bC_Na_CO3_MP98(T, P):
    """c-a: sodium carbonate [MP98]."""
    # I have no idea where MP98 got their T**2 terms from
    b0 = 0.0362 + 0.00179 * (T - 298.15) + 1.694e-21 * (T - 298.15) ** 2
    b1 = 1.51 + 0.00205 * (T - 298.15) + 1.626e-19 * (T - 298.15) ** 2
    b2 = 0
    Cphi = 0.0052
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["CO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_HCO3_MP98(T, P):
    """c-a: sodium bicarbonate [MP98]."""
    # I have no idea where MP98 got their T**2 terms from
    b0 = 0.028 + 0.001 * (T - 298.15) + 5.082001e-21 * (T - 298.15) ** 2
    b1 = 0.044 + 0.0011 * (T - 298.15) - 3.88e-21 * (T - 298.15) ** 2
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 273.15) & (T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_HSO4_MP98(T, P):
    """c-a: potassium bisulfate [MP98]."""
    # MP98 cite Pierrot & Millero (1997) in the PM16 code for this
    b0 = -1.8949 - 0.00059751 * (T - 298.15)
    b1 = 5.0284 - 0.0284 * (T - 298.15)
    b2 = 0.0
    Cphi = 0.9246 + 0.0039751 * (T - 298.15)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["HSO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mg_HSO4_MP98(T, P):
    """c-a: magnesium bisulfate [MP98]."""
    # MP98 cite Pierrot & Millero (1997) in the PM16 code for this
    b0 = -0.61656 - 0.00075174 * (T - 298.15)
    b1 = 7.716066 - 0.0164302 * (T - 298.15)
    b2 = 0.0
    Cphi = 0.43026 + 0.00199601 * (T - 298.15)
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Mg"] * i2c["HSO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_BOH4_MP98(T, P):
    """c-a: potassium borate [MP98]."""
    # MP98 say this is SRRJ87 but then use different coefficients for Cphi
    b0 = SRRJ87_eq7(
        T,
        [
            0.1469,
            2.881,
            0,
        ],
    )
    b1 = SRRJ87_eq7(
        T,
        [
            -0.0989,
            -6.876,
            0,
        ],
    )
    b2 = 0
    Cphi = SRRJ87_eq7(
        T,
        [
            -56.43e-3,
            -0.956,
            0,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["BOH4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 278.15) & (T <= 328.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_BOH4_MP98(T, P):
    """c-a: sodium borate [MP98]."""
    # MP98 say this is SRRJ87 but then use different coefficients for Cphi
    b0 = SRRJ87_eq7(
        T,
        [
            -0.0510,
            5.264,
            0,
        ],
    )
    b1 = SRRJ87_eq7(
        T,
        [
            0.0961,
            -10.68,
            0,
        ],
    )
    b2 = 0
    Cphi = SRRJ87_eq7(
        T,
        [
            14.98e-3,
            -1.57,
            0,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["BOH4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 278.15) & (T <= 328.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Archer (1999) ~~~~~
def A99_eq22(T, a):
    """A99 equation 22."""
    Tref = 298.15
    return (
        a[0]
        + a[1] * (T - Tref) * 1e-2
        + a[2] * (T - Tref) ** 2 * 1e-5
        + a[3] * 1e2 / (T - 225)
        + a[4] * 1e3 / T
        + a[5] * 1e6 / (T - 225) ** 3
    )


def bC_K_Cl_A99(T, P):
    """c-a: potassium chloride [A99]."""
    # KCl T parameters from A99 Table 4
    b0 = A99_eq22(
        T,
        [
            0.413229483398493,
            -0.0870121476114027,
            0.101413736179231,
            -0.0199822538522801,
            -0.0998120581680816,
            0,
        ],
    )
    b1 = A99_eq22(
        T,
        [
            0.206691413598171,
            0.102544606022162,
            0,
            0,
            0,
            -0.00188349608000903,
        ],
    )
    b2 = 0
    C0 = A99_eq22(
        T,
        [
            -0.00133515934994478,
            0,
            0,
            0.00234117693834228,
            -0.00075896583546707,
            0,
        ],
    )
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = (T >= 260) & (T <= 420)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Rard and Clegg (1999) ~~~~~
def bC_Mg_HSO4_RC99(T, P):
    """c-a: magnesium bisulfate [RC99]."""
    # RC99 Table 6, left column
    b0 = 0.40692
    b1 = 1.6466
    b2 = 0
    C0 = 0.024293
    C1 = -0.127194
    alph1 = 2
    alph2 = -9
    omega = 1
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def psi_H_Mg_HSO4_RC99(T, P):
    """c-c'-a: hydrogen magnesium bisulfate [RC99]."""
    # RC99 Table 6, left column
    psi = -0.027079
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_H_Mg_SO4_RC99(T, P):
    """c-c'-a: hydrogen magnesium sulfate [RC99]."""
    # RC99 Table 6, left column
    psi = -0.047368
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def psi_Mg_HSO4_SO4_RC99(T, P):
    """c-a-a': magnesium bisulfate sulfate [RC99]."""
    # RC99 Table 6, left column
    psi = -0.078418
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Miladinović et al. (2008) ~~~~~
def bC_Mg_Cl_MNTR08(T, P):
    """c-a: magnesium chloride [MNTR08]."""
    b0 = 0.68723
    b1 = 1.56760
    b2 = 0
    C0 = -0.0007594
    C1 = -0.35497
    alph1 = 3.0
    alph2 = -9
    omega = 1.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Mg_SO4_MNTR08(T, P):
    """c-a: magnesium sulfate [MNTR08]."""
    b0 = -0.03089
    b1 = 3.7687
    b2 = -37.3659
    C0 = 0.016406
    C1 = 0.34549
    alph1 = 1.4
    alph2 = 12.0
    omega = 1.0
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def theta_Cl_SO4_MNTR08(
    T,
):
    """a-a': chloride sulfate [MNTR08]."""
    theta = -0.07122
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_Mg_Cl_SO4_MNTR08(
    T,
):
    """c-a-a': magnesium chloride sulfate [MNTR08]."""
    psi = -0.038505
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Waters and Millero (2013) ~~~~~
# Some are functions that WM13 declared came from another source, but I
#  couldn't find them there, so copied directly from WM13 instead.
# Others were just declared by WM13 as zero. These all seem to agree with
#  HMW84; it's unclear why HMW84 wasn't cited by WM13 for these.
# First, a few functions that WM13 constructed by taking 298.15 K parameters
#  from HMW84, and correcting for temperature using derivatives from P91.


def bC_Ca_SO4_WM13(T, P):
    """c-a: calcium sulfate [WM13]."""
    TR = 298.15
    b0, b1, b2, C0, C1, alph1, alph2, omega, valid = bC_Ca_SO4_HMW84(T, P)
    # WM13 use temperature derivatives from P91
    # The b0 temperature correction in P91 is zero
    b1 = b1 + (T - TR) * 5.460e-2
    b2 = b2 + (T - TR) * -5.16e-1
    # The C0 temperature correction in P91 is zero
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Ca_HSO4_WM13(T, P):
    """c-a: calcium bisulfate [WM13]."""
    TR = 298.15
    b0, b1, b2, C0, C1, alph1, alph2, omega, valid = bC_Ca_HSO4_HMW84(T, P)
    # WM13 use temperature derivatives for Ca-ClO4 from P91, but with typos
    b0 = b0 + (T - TR) * 0.830e-3
    b1 = b1 + (T - TR) * 5.08e-3
    C0 = C0 + (T - TR) * -1.090e-4
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_HSO4_WM13(T, P):
    """c-a: potassium bisulfate [WM13]."""
    TR = 298.15
    b0, b1, b2, C0, C1, alph1, alph2, omega, valid = bC_K_HSO4_HMW84(T, P)
    # WM13 use temperature derivatives for K-ClO4 from P91
    b0 = b0 + (T - TR) * 0.600e-4
    b1 = b1 + (T - TR) * 100.700e-4
    # The Cphi temperature correction in P91 is zero
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_HSO4_HPR93viaWM13(T, P):
    """c-a: sodium sulfate [HPR93 via WM13]."""
    # WM13 Table A1 - can't find where HPR93 state this
    return bC_none(T, P)


def theta_HSO4_SO4_WM13(T, P):
    """a-a': bisulfate sulfate [WM13]."""
    return theta_none(T, P)  # WM13 Table A7


def psi_H_Cl_SO4_WM13(T, P):
    """c-a-a': hydrogen chloride sulfate [WM13]."""
    return psi_none(T, P)  # WM13 Table A8


def psi_H_Cl_OH_WM13(T, P):
    """c-a-a': hydrogen chloride hydroxide [WM13]."""
    return psi_none(T, P)  # WM13 Table A8


def psi_Mg_Cl_OH_WM13(T, P):
    """c-a-a': magnesium chloride hydroxide [WM13]."""
    return psi_none(T, P)  # WM13 Table A8


def psi_Ca_HSO4_SO4_WM13(T, P):
    """c-a-a': calcium bisulfate sulfate [WM13]."""
    return psi_none(T, P)  # WM13 Table A8


def psi_H_OH_SO4_WM13(T, P):
    """c-a-a': hydrogen hydroxide sulfate [WM13]."""
    return psi_none(T, P)  # WM13 Table A8


def psi_Mg_OH_SO4_WM13(T, P):
    """c-a-a': magnesium hydroxide sulfate [WM13]."""
    return psi_none(T, P)  # WM13 Table A8


def psi_Ca_OH_SO4_WM13(T, P):
    """c-a-a': calcium hydroxide sulfate [WM13]."""
    return psi_none(T, P)  # WM13 Table A8


def psi_H_Na_SO4_WM13(T, P):
    """c-c'-a: hydrogen sodium sulfate [WM13]."""
    return psi_none(T, P)  # WM13 Table A9


def psi_Ca_H_SO4_WM13(T, P):
    """c-c'-a: calcium hydrogen sulfate [WM13]."""
    return psi_none(T, P)  # WM13 Table A9


def psi_Ca_H_HSO4_WM13(T, P):
    """c-c'-a: calcium hydrogen bisulfate [WM13]."""
    return psi_none(T, P)  # WM13 Table A9


def psi_Mg_Na_HSO4_WM13(T, P):
    """c-c'-a: magnesium sodium bisulfate [WM13]."""
    return psi_none(T, P)  # WM13 Table A9


def psi_Ca_Na_HSO4_WM13(T, P):
    """c-c'-a: calcium sodium bisulfate [WM13]."""
    return psi_none(T, P)  # WM13 Table A9


def psi_K_Na_HSO4_WM13(T, P):
    """c-c'-a: potassium sodium bisulfate [WM13]."""
    return psi_none(T, P)  # WM13 Table A9


def psi_Ca_Mg_HSO4_WM13(T, P):
    """c-c'-a: calcium magnesium bisulfate [WM13]."""
    return psi_none(T, P)  # WM13 Table A9


def psi_K_Mg_HSO4_WM13(T, P):
    """c-c'-a: potassium magnesium bisulfate [WM13]."""
    return psi_none(T, P)  # WM13 Table A9


def psi_Ca_K_SO4_WM13(T, P):
    """c-c'-a: calcium potassium sulfate [WM13]."""
    return psi_none(T, P)  # WM13 Table A9


def psi_Ca_K_HSO4_WM13(T, P):
    """c-c'-a: calcium potassium bisulfate [WM13]."""
    return psi_none(T, P)  # WM13 Table A9


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Gallego-Urrea and Turner (2017) ~~~~~
# From G17 Supp. Info. Table S6, 'simultaneous optimisation'.
def bC_Na_Cl_GT17simopt(T, P):
    """c-a: sodium chloride [GT17simopt]."""
    b0 = 0.07722
    b1 = 0.26768
    b2 = 0
    Cphi = 0.001628
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_trisH_Cl_GT17simopt(T, P):
    """c-a: trisH+ chloride [GT17simopt]."""
    b0 = 0.04181
    b1 = 0.16024
    b2 = 0
    Cphi = -0.00132
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["trisH"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_trisH_SO4_GT17simopt(T, P):
    """c-a: trisH+ sulfate [GT17simopt]."""
    b0 = 0.09746
    b1 = 0.52936
    b2 = 0
    Cphi = -0.004957
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["trisH"] * i2c["SO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def theta_H_trisH_GT17simopt(T, P):
    """c-c': hydrogen trisH [GT17simopt]."""
    theta = -0.00575
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def psi_H_trisH_Cl_GT17simopt(T, P):
    """c-c'-a: hydrogen trisH chloride [GT17simopt]."""
    psi = -0.00700
    valid = np.isclose(T, 298.15, **temperature_tol)
    return psi, valid


def lambd_tris_trisH_GT17simopt(T, P):
    """n-c: tris trisH [GT17simopt]."""
    lambd = 0.06306
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_tris_Na_GT17simopt(T, P):
    """n-c: tris sodium [GT17simopt]."""
    lambd = 0.01580
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_tris_K_GT17simopt(T, P):
    """n-c: tris potassium [GT17simopt]."""
    lambd = 0.02895
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_tris_Mg_GT17simopt(T, P):
    """n-c: tris magnesium [GT17simopt]."""
    lambd = -0.14505
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_tris_Ca_GT17simopt(T, P):
    """n-c: tris calcium [GT17simopt]."""
    lambd = -0.31081
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Zezin and Driesner (2017) ~~~~~
def ZD17_eq8(T, P, b):
    """ZD17 equation 8, pressure in MPa."""
    return (
        b[0]
        + b[1] * T / 1000
        + b[2] * (T / 500) ** 2
        + b[3] / (T - 215)
        + b[4] * 1e4 / (T - 215) ** 3
        + b[5] * 1e2 / (T - 215) ** 2
        + b[6] * 2e2 / T ** 2
        + b[7] * (T / 500) ** 3
        + b[8] / (650 - T) ** 0.5
        + b[9] * 1e-5 * P
        + b[10] * 2e-4 * P / (T - 225)
        + b[11] * 1e2 * P / (650 - T) ** 3
        + b[12] * 1e-5 * P * T / 500
        + b[13] * 2e-4 * P / (650 - T)
        + b[14] * 1e-7 * P ** 2
        + b[15] * 2e-6 * P ** 2 / (T - 225)
        + b[16] * P ** 2 / (650 - T) ** 3
        + b[17] * 1e-7 * P ** 2 * T / 500
        + b[18] * 1e-7 * P ** 2 * (T / 500) ** 2
        + b[19] * 4e-2 * P / (T - 225) ** 2
        + b[20] * 1e-5 * P * (T / 500) ** 2
        + b[21] * 2e-8 * P ** 3 / (T - 225)
        + b[22] * 1e-2 * P ** 3 / (650 - T) ** 3
        + b[23] * 2e2 / (650 - T) ** 3
    )


def bC_K_Cl_ZD17(T, P):
    """c-a: potassium chloride [ZD17]."""
    P_MPa = P / 100  # Convert dbar to MPa
    # KCl T and P parameters from ZD17 Table 2
    b0 = ZD17_eq8(
        T,
        P_MPa,
        [
            0.0263285,
            0.0713524,
            -0.008957,
            -1.3320169,
            -0.6454779,
            -0.758977,
            9.4585163,
            -0.0186077,
            0.211171,
            0,
            22.686075,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    b1 = ZD17_eq8(
        T,
        P_MPa,
        [
            -0.1191678,
            0.7216226,
            0,
            8.5388026,
            4.3794936,
            -11.743658,
            -25.744757,
            -0.1638556,
            3.444429,
            0,
            0.7549375,
            -7.2651892,
            0,
            0,
            0,
            0,
            4.0457998,
            0,
            0,
            -162.81428,
            296.7078,
            0,
            -0.7343191,
            46.340392,
        ],
    )
    b2 = 0
    C0 = ZD17_eq8(
        T,
        P_MPa,
        [
            -0.0005981,
            0.002905,
            -0.0028921,
            -0.1711606,
            0.0479309,
            0.141835,
            0,
            0.0009746,
            0.0084333,
            0,
            10.518644,
            0,
            1.1917209,
            -9.3262105,
            0,
            0,
            0,
            0,
            0,
            -5.4129002,
            0,
            0,
            0,
            0,
        ],
    )
    C1 = ZD17_eq8(
        T,
        P_MPa,
        [
            0,
            1.0025605,
            0,
            0,
            3.0805818,
            0,
            -86.99429,
            -0.3005514,
            0,
            -47.235583,
            -901.18412,
            -2.326187,
            0,
            -504.46628,
            0,
            0,
            -4.7090241,
            0,
            0,
            542.1083,
            0,
            0,
            1.6548655,
            59.165704,
        ],
    )
    alph1 = 2
    alph2 = -9
    omega = 2.5
    valid = T <= 600
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MarChemSpec project ~~~~~
def theta_Ca_H_MarChemSpec(T, P):
    """c-c': calcium hydrogen [MarChemSpec]."""
    # 1. WM13 cite the wrong reference for this (they say RXX80)
    # 2. The equation given by WM13 doesn't match RGO81
    # 3. RGO81 give a 25degC value but no temperature parameter
    # So MarChemSpec uses RGO81's 25degC value plus the WM13 temperature cxn
    thetar = theta_Ca_H_RGO81(T, P)[0]
    theta = thetar + 3.275e-4 * (T - 298.15)
    valid = (T >= 273.15) & (T <= 323.15)
    return theta, valid


def theta_H_Na_MarChemSpec25(T, P):
    """c-c': hydrogen sodium [MarChemSpec]."""
    theta = 0.036
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def theta_H_K_MarChemSpec25(T, P):
    """c-c': hydrogen potassium [MarChemSpec]."""
    theta = 0.005
    valid = np.isclose(T, 298.15, **temperature_tol)
    return theta, valid


def lambd_tris_tris_MarChemSpec25(T, P):
    """n-n: tris tris [MarChemSpec]."""
    # Temporary value from "MODEL PARAMETERS FOR TRIS Tests.docx" (2019-01-31)
    lambd = -0.006392
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def zeta_tris_Na_Cl_MarChemSpec25(T, P):
    """n-c-a: tris sodium chloride [MarChemSpec]."""
    # Temporary value from "MODEL PARAMETERS FOR TRIS Tests.docx" (2019-01-31)
    zeta = -0.003231
    valid = np.isclose(T, 298.15, **temperature_tol)
    return zeta, valid


def mu_tris_tris_tris_MarChemSpec25(T, P):
    """n-n-n: tris tris tris [MarChemSpec]."""
    # Temporary value from "MODEL PARAMETERS FOR TRIS Tests.docx" (2019-01-31)
    mu = 0.0009529
    valid = np.isclose(T, 298.15, **temperature_tol)
    return mu, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Clegg et al. (2021) ~~~~~
def bC_trisH_Cl_CHW21(T, P):
    """c-a: trisH+ chloride [CHW21]."""
    b0 = 0.03468
    b1 = 0.12802
    b2 = 0
    C0 = -9.366e-4
    C1 = 0.09269
    alph1 = 2
    alph2 = -9
    omega = 2.5
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_trisH_SO4_CHW21(T, P):
    """c-a: trisH+ sulfate [CHW21]."""
    b0 = 9.52294e-2
    b1 = 0.585908
    b2 = 0
    C0 = -1.59883e-3
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def lambd_tris_Ca_CHW21(T, P):
    """n-c: tris calcium [CHW21]."""
    lambd = -0.2686
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_tris_K_CHW21(T, P):
    """n-c: tris potassium [CHW21]."""
    lambd = 0.03394
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_tris_Mg_CHW21(T, P):
    """n-c: tris magnesium [CHW21]."""
    lambd = -0.1176
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_tris_Na_CHW21(T, P):
    """n-c: tris sodium [CHW21]."""
    lambd = 0.02632
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_tris_trisH_LTA21(T, P):
    """n-c: tris trisH+ [LTA21]."""
    lambd = -0.01241
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_tris_SO4_LTA21(T, P):
    """n-a: tris sulfate [LTA21]."""
    lambd = 0.08245
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def lambd_tris_tris_LTA21(T, P):
    """n-n: tris tris [LTA21]."""
    lambd = -0.0051635
    valid = np.isclose(T, 298.15, **temperature_tol)
    return lambd, valid


def mu_tris_tris_tris_LTA21(T, P):
    """n-n-n: tris tris tris [LTA21]."""
    mu = 0.000703
    valid = np.isclose(T, 298.15, **temperature_tol)
    return mu, valid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ JESS ~~~~~
# Ref. JESS = parameters obtained from http://jess.murdoch.edu.au/vewbel.shtml
def JESS_eq(T, P, j):
    Tr = 298.15
    Pr = 10
    return (
        j[0]
        + j[1] * (1 / T - 1 / Tr) * 1e3
        + j[2] * np.log(T / Tr)
        + j[3] * (P - Pr) * 0.0001
    )


def bC_H_Br_JESS(T, P):
    """c-a: hydrogen bromide [JESS]."""
    # Coefficients obtained online [2019-08-09]
    b0 = JESS_eq(
        T,
        P,
        [
            0.210904,
            -0.0111577,
            -0.0965749,
            -0.00485161,
        ],
    )
    b1 = JESS_eq(
        T,
        P,
        [
            0.324833,
            -0.141250,
            -0.287944,
            -0.0209720,
        ],
    )
    b2 = 0
    Cphi = JESS_eq(
        T,
        P,
        [
            0.00113432,
            -0.0110028,
            -0.0542243,
            0.000704627,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["H"] * i2c["Br"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)  # unknown validity
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_H_Cl_JESS(T, P):
    """c-a: hydrogen chloride [JESS]."""
    # Coefficients obtained online [2019-08-09]
    b0 = JESS_eq(
        T,
        P,
        [
            0.178627,
            0.209336,
            0.580887,
            0.000743741,
        ],
    )
    b1 = JESS_eq(
        T,
        P,
        [
            0.288278,
            -0.757685,
            -2.38033,
            -0.00662854,
        ],
    )
    b2 = 0
    Cphi = JESS_eq(
        T,
        P,
        [
            0.000259560,
            -0.0628809,
            -0.219810,
            1.46066e-5,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["H"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)  # unknown validity
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_H_ClO4_JESS(T, P):
    """c-a: hydrogen perchlorate [JESS]."""
    # Coefficients obtained online [2019-08-09]
    b0 = JESS_eq(
        T,
        P,
        [
            0.177990,
            -0.126144,
            -0.278470,
            -0.000346259,
        ],
    )
    b1 = JESS_eq(
        T,
        P,
        [
            0.282791,
            -0.469348,
            -0.990997,
            -0.0550796,
        ],
    )
    b2 = 0
    Cphi = JESS_eq(
        T,
        P,
        [
            0.00754498,
            0.0103387,
            0.000000,
            -0.00157682,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["H"] * i2c["ClO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)  # unknown validity
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_H_I_JESS(T, P):
    """c-a: hydrogen iodide [JESS]."""
    # Coefficients obtained online [2019-08-09]
    b0 = JESS_eq(
        T,
        P,
        [
            0.231988,
            -0.138301,
            -0.470802,
            -0.0212351,
        ],
    )
    b1 = JESS_eq(
        T,
        P,
        [
            0.419989,
            -0.344954,
            -0.859537,
            0.0477403,
        ],
    )
    b2 = 0
    Cphi = JESS_eq(
        T,
        P,
        [
            0.00286688,
            0.00703988,
            0.000000,
            0.00346557,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["H"] * i2c["I"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)  # unknown validity
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_H_NO3_JESS(T, P):
    """c-a: hydrogen nitrate [JESS]."""
    # Coefficients obtained online [2019-08-09]
    b0 = JESS_eq(
        T,
        P,
        [
            0.116523,
            -0.0848453,
            -0.216476,
            -0.0109287,
        ],
    )
    b1 = JESS_eq(
        T,
        P,
        [
            0.351089,
            -0.830710,
            -2.43129,
            0.0206574,
        ],
    )
    b2 = 0
    Cphi = JESS_eq(
        T,
        P,
        [
            -0.00531372,
            -0.0289539,
            -0.123911,
            0.00299856,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["H"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)  # unknown validity
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_Br_JESS(T, P):
    """c-a: potassium bromide [JESS]."""
    # Coefficients obtained online [2019-08-09]
    b0 = JESS_eq(
        T,
        P,
        [
            0.0547434,
            -0.420409,
            -1.13677,
            0.00766319,
        ],
    )
    b1 = JESS_eq(
        T,
        P,
        [
            0.238709,
            -0.160445,
            -0.235802,
            0.0172136,
        ],
    )
    b2 = 0
    Cphi = JESS_eq(
        T,
        P,
        [
            -0.00145317,
            0.0658735,
            0.185412,
            -0.000339080,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["Br"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)  # unknown validity
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_Cl_JESS(T, P):
    """c-a: potassium chloride [JESS]."""
    # Coefficients obtained online [2019-08-08]
    b0 = JESS_eq(
        T,
        P,
        [
            0.0478024,
            -0.359514,
            -1.00809,
            0.0104416,
        ],
    )
    b1 = JESS_eq(
        T,
        P,
        [
            0.220310,
            -0.158289,
            -0.228114,
            0.0198966,
        ],
    )
    b2 = 0
    Cphi = JESS_eq(
        T,
        P,
        [
            -0.000751880,
            0.0516663,
            0.149612,
            -0.000853398,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)  # unknown validity
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_I_JESS(T, P):
    """c-a: potassium iodide [JESS]."""
    # Coefficients obtained online [2019-08-09]
    b0 = JESS_eq(
        T,
        P,
        [
            0.0706420,
            -0.176246,
            -0.291178,
            0.0103160,
        ],
    )
    b1 = JESS_eq(
        T,
        P,
        [
            0.279022,
            -1.11921,
            -3.31506,
            -0.0125590,
        ],
    )
    b2 = 0
    Cphi = JESS_eq(
        T,
        P,
        [
            -0.00335865,
            0.000000,
            -0.0301254,
            -0.00160355,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["I"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)  # unknown validity
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_NO3_JESS(T, P):
    """c-a: potassium nitrate [JESS]."""
    # Coefficients obtained online [2019-08-09]
    b0 = JESS_eq(
        T,
        P,
        [
            -0.0763632,
            -0.582629,
            -1.49134,
            -0.000422651,
        ],
    )
    b1 = JESS_eq(
        T,
        P,
        [
            0.0435708,
            -1.90996,
            -4.95256,
            0.0795007,
        ],
    )
    b2 = 0
    Cphi = JESS_eq(
        T,
        P,
        [
            0.00526553,
            0.101479,
            0.274296,
            0.00254839,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)  # unknown validity
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_K_OH_JESS(T, P):
    """c-a: potassium hydroxide [JESS]."""
    # Coefficients obtained online [2019-08-09]
    b0 = JESS_eq(
        T,
        P,
        [
            0.153478,
            -0.519124,
            -1.69587,
            0.0274921,
        ],
    )
    b1 = JESS_eq(
        T,
        P,
        [
            0.184252,
            0.107673,
            0.539154,
            0.000813086,
        ],
    )
    b2 = 0
    Cphi = JESS_eq(
        T,
        P,
        [
            -0.000479227,
            0.0899943,
            0.295729,
            -0.00264638,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["K"] * i2c["OH"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)  # unknown validity
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Li_Br_JESS(T, P):
    """c-a: lithium bromide [JESS]."""
    # Coefficients obtained online [2019-08-09]
    b0 = JESS_eq(
        T,
        P,
        [
            0.173865,
            0.174090,
            0.523963,
            0.00576938,
        ],
    )
    b1 = JESS_eq(
        T,
        P,
        [
            0.287573,
            -0.908258,
            -2.72816,
            -0.0110614,
        ],
    )
    b2 = 0
    Cphi = JESS_eq(
        T,
        P,
        [
            0.00505045,
            -0.0474007,
            -0.165446,
            -0.00114107,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Li"] * i2c["Br"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)  # unknown validity
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Li_Cl_JESS(T, P):
    """c-a: lithium chloride [JESS]."""
    # Coefficients obtained online [2019-08-09]
    b0 = JESS_eq(
        T,
        P,
        [
            0.148491,
            0.0993795,
            0.288338,
            0.00616159,
        ],
    )
    b1 = JESS_eq(
        T,
        P,
        [
            0.305863,
            -0.534953,
            -1.57812,
            0.00327985,
        ],
    )
    b2 = 0
    Cphi = JESS_eq(
        T,
        P,
        [
            0.00363693,
            -0.0309516,
            -0.118649,
            -0.000747312,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Li"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)  # unknown validity
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Li_ClO4_JESS(T, P):
    """c-a: lithium perchlorate [JESS]."""
    # Coefficients obtained online [2019-08-09]
    b0 = JESS_eq(
        T,
        P,
        [
            0.194332,
            0.0519232,
            0.179439,
            -0.000735779,
        ],
    )
    b1 = JESS_eq(
        T,
        P,
        [
            0.435514,
            -0.694923,
            -2.09437,
            0.00524685,
        ],
    )
    b2 = 0
    Cphi = JESS_eq(
        T,
        P,
        [
            0.00160986,
            -0.0229881,
            -0.0977900,
            0.000703517,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Li"] * i2c["ClO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)  # unknown validity
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Li_I_JESS(T, P):
    """c-a: lithium iodide [JESS]."""
    # Coefficients obtained online [2019-08-09]
    b0 = JESS_eq(
        T,
        P,
        [
            0.176338,
            0.000000,
            -0.129076,
            -0.00291167,
        ],
    )
    b1 = JESS_eq(
        T,
        P,
        [
            0.523319,
            0.0858259,
            0.810061,
            -0.000530723,
        ],
    )
    b2 = 0
    Cphi = JESS_eq(
        T,
        P,
        [
            0.0109604,
            -0.0457470,
            -0.125706,
            0.000223867,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Li"] * i2c["I"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)  # unknown validity
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Li_NO3_JESS(T, P):
    """c-a: lithium nitrate [JESS]."""
    # Coefficients obtained online [2019-08-09]
    b0 = JESS_eq(
        T,
        P,
        [
            0.140963,
            -0.285745,
            -0.959602,
            -0.00269645,
        ],
    )
    b1 = JESS_eq(
        T,
        P,
        [
            0.286879,
            0.763311,
            2.87993,
            0.0225144,
        ],
    )
    b2 = 0
    Cphi = JESS_eq(
        T,
        P,
        [
            -0.00543304,
            0.00233974,
            0.000000,
            0.000892616,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Li"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)  # unknown validity
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Li_OH_JESS(T, P):
    """c-a: lithium hydroxide [JESS]."""
    # Coefficients obtained online [2019-08-09]
    b0 = JESS_eq(
        T,
        P,
        [
            0.0544861,
            -0.245358,
            -0.865967,
            0.00588145,
        ],
    )
    b1 = JESS_eq(
        T,
        P,
        [
            -0.139320,
            -0.725360,
            -2.44668,
            0.0583334,
        ],
    )
    b2 = 0
    Cphi = JESS_eq(
        T,
        P,
        [
            -0.00433424,
            0.0208725,
            0.0579693,
            0.00152371,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Li"] * i2c["OH"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)  # unknown validity
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_Br_JESS(T, P):
    """c-a: sodium bromide [JESS]."""
    # Coefficients obtained online [2019-08-09]
    b0 = JESS_eq(
        T,
        P,
        [
            0.0987329,
            -0.428710,
            -1.19150,
            0.00937076,
        ],
    )
    b1 = JESS_eq(
        T,
        P,
        [
            0.285551,
            -0.452514,
            -1.24784,
            0.000698698,
        ],
    )
    b2 = 0
    Cphi = JESS_eq(
        T,
        P,
        [
            0.000736067,
            0.0467849,
            0.124757,
            -0.00113162,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["Br"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)  # unknown validity
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_Cl_JESS(T, P):
    """c-a: sodium chloride [JESS]."""
    # Coefficients obtained online [2019-08-08]
    b0 = JESS_eq(
        T,
        P,
        [
            0.0779802,
            -0.429500,
            -1.21510,
            0.0108697,
        ],
    )
    b1 = JESS_eq(
        T,
        P,
        [
            0.269983,
            -0.379473,
            -1.03405,
            0.00877337,
        ],
    )
    b2 = 0
    Cphi = JESS_eq(
        T,
        P,
        [
            0.000918109,
            0.0518882,
            0.139812,
            -0.000952450,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["Cl"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)  # unknown validity
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_ClO4_JESS(T, P):
    """c-a: sodium perchlorate [JESS]."""
    # Coefficients obtained online [2019-08-09]
    b0 = JESS_eq(
        T,
        P,
        [
            0.0557192,
            -0.519285,
            -1.35911,
            0.00925914,
        ],
    )
    b1 = JESS_eq(
        T,
        P,
        [
            0.280349,
            -1.02206,
            -2.73189,
            0.0158837,
        ],
    )
    b2 = 0
    Cphi = JESS_eq(
        T,
        P,
        [
            -0.00126494,
            0.0944381,
            0.269364,
            -0.000962511,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["ClO4"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)  # unknown validity
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_I_JESS(T, P):
    """c-a: sodium iodide [JESS]."""
    # Coefficients obtained online [2019-08-09]
    b0 = JESS_eq(
        T,
        P,
        [
            0.124139,
            -0.149852,
            -0.225275,
            0.00691965,
        ],
    )
    b1 = JESS_eq(
        T,
        P,
        [
            0.317353,
            -1.47625,
            -4.67803,
            -0.0109580,
        ],
    )
    b2 = 0
    Cphi = JESS_eq(
        T,
        P,
        [
            0.000609691,
            -0.0485681,
            -0.197132,
            -0.000968575,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["I"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)  # unknown validity
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_NO3_JESS(T, P):
    """c-a: sodium nitrate [JESS]."""
    # Coefficients obtained online [2019-08-09]
    b0 = JESS_eq(
        T,
        P,
        [
            0.00500042,
            -0.708908,
            -1.99654,
            0.00501474,
        ],
    )
    b1 = JESS_eq(
        T,
        P,
        [
            0.200360,
            -0.879277,
            -2.23652,
            0.0625764,
        ],
    )
    b2 = 0
    Cphi = JESS_eq(
        T,
        P,
        [
            -0.000347888,
            0.153367,
            0.461828,
            0.000540176,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["NO3"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)  # unknown validity
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def bC_Na_OH_JESS(T, P):
    """c-a: sodium hydroxide [JESS]."""
    # Coefficients obtained online [2019-08-09]
    b0 = JESS_eq(
        T,
        P,
        [
            0.0857078,
            -0.618667,
            -1.90371,
            0.0278606,
        ],
    )
    b1 = JESS_eq(
        T,
        P,
        [
            0.276706,
            -0.443782,
            -1.30232,
            0.0242386,
        ],
    )
    b2 = 0
    Cphi = JESS_eq(
        T,
        P,
        [
            0.00418666,
            0.0982562,
            0.285252,
            -0.00224765,
        ],
    )
    C0 = Cphi / (2 * np.sqrt(np.abs(i2c["Na"] * i2c["OH"])))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = np.isclose(T, 298.15, **temperature_tol)  # unknown validity
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid


def psi_K_Mg_Cl_A15(T, P):
    """c-c'-a: potassium magnesium chloride [LS58 via A15]."""
    psi = -0.022 - 14.27 * (1 / T - 1 / 298.15)
    valid = (273.15 <= T) & (T <= 473.15)
    return psi, valid
