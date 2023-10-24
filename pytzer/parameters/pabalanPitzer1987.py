# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2023  Matthew P. Humphreys  (GNU GPLv3)
"""PP87ii: Pabalan & Pitzer (1987) Geochim. Cosmochim. Acta 51(9), 2429-2443.
https://doi.org/10.1016/0016-7037(87)90295-X
"""


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
