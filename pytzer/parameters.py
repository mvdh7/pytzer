# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
"""Evaluate Pitzer model interaction parameters."""
from autograd.numpy import array, float_, log, logical_and, sqrt
from autograd.numpy import abs as np_abs
from .constants import Tzero
from .tables import (P91_Ch3_T12, P91_Ch3_T13_I, P91_Ch3_T13_II,
    PM73_TableI, PM73_TableVI, PM73_TableVIII, PM73_TableIX)
from . import properties

# Note that variable T in this module is equivalent to tempK elsewhere (in K),
# and P is equivalent to pres (in dbar), for convenience

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Zero functions ~~~~~
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pitzer and Margoya (1973) ~~~~~
def bC_PM73(T, iset):
    # Experimental function - not production-ready
    zM, zX = properties.charges(array(iset.split('-')))[0]
    PM73_Tables = {
        -1: PM73_TableI,
        -2: PM73_TableVI,
        -3: PM73_TableVIII,
        -4: PM73_TableIX,
        -5: PM73_TableIX,
    }
    b0 = PM73_Tables[zM*zX][iset]['b0']
    b1 = PM73_Tables[zM*zX][iset]['b1']
    b2 = 0
    Cphi = PM73_Tables[zM*zX][iset]['Cphi']
    C0 = Cphi / (2 * sqrt(np_abs(zM * zX)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Sr_Br_PM73(T, P):
    """c-a: strontium bromide [PM73]."""
    # PM73 cite Robinson & Stokes (1965) Electrolyte Solutions, 2nd Ed.
    b0 = 0.4415 * 3/4
    b1 = 2.282 * 3/4
    b2 = 0
    Cphi = 0.00231 * 3/2**2.5
    zSr = +2
    zBr = -1
    C0 = Cphi / (2 * sqrt(np_abs(zSr * zBr)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Sr_Cl_PM73(T, P):
    """c-a: strontium chloride [PM73]."""
    # PM73 cite Robinson & Stokes (1965) Electrolyte Solutions, 2nd Ed.
    b0 = 0.3810 * 3/4
    b1 = 2.223 * 3/4
    b2 = 0
    Cphi = -0.00246 * 3/2**2.5
    zSr = +2
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(zSr * zCl)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_H2PO4_PM73(T, P):
    """c-a: potassium dihydrogen-phosphate [PM73]."""
    b0 = -0.0678
    b1 = -0.1042
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_SCN_PM73(T, P):
    """c-a: potassium thiocyanate [PM73]."""
    b0 = 0.0416
    b1 = 0.2302
    b2 = 0
    Cphi = -0.00252
    zK = +1
    zSCN = -1
    C0 = Cphi / (2 * sqrt(np_abs(zK * zSCN)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_SCN_PM73(T, P):
    """c-a: sodium thiocyanate [PM73]."""
    b0 = 0.1005
    b1 = 0.3582
    b2 = 0
    Cphi = -0.00303
    zNa = +1
    zSCN = -1
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zSCN)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Silvester and Pitzer (1978) ~~~~~
# General procedure:
#  - Inherit 298.15 K value from PM73;
#  - Add temperature derivative correction from SP78.
SP78_Tr = 298.15

def bC_Sr_Br_SP78(T, P):
    """c-a: strontium bromide [SP78]."""
    # SP78 cite Lange & Streeck (1930) Z Phys Chem Abt A 152
    b0r, b1r, b2, C0, C1, alph1, alph2, omega, _ = bC_Sr_Br_PM73(T, P)
    b0 = b0r + (-0.437e-3 * 3/4) * (T - SP78_Tr)
    b1 = b1r + (8.71e-3 * 3/4) * (T - SP78_Tr)
    # Validity range declared by MP98
    valid = logical_and(T >= 283.15,T <= 313.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Sr_Cl_SP78(T, P):
    """c-a: strontium chloride [SP78]."""
    # SP78 cite Lange & Streeck (1930) Z Phys Chem Abt A 152
    b0r, b1r, b2, C0, C1, alph1, alph2, omega, _ = bC_Sr_Br_PM73(T, P)
    b0 = b0r + (0.956e-3 * 3/4) * (T - SP78_Tr)
    b1 = b1r + (3.79e-3 * 3/4) * (T - SP78_Tr)
    # Validity range declared by MP98
    valid = logical_and(T >= 283.15,T <= 313.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_H2PO4_SP78(T, P):
    """c-a: potassium dihydrogen-phosphate [SP78]."""
    b0r, b1r, b2, C0r, C1, alph1, alph2, omega, _ = bC_K_H2PO4_PM73(T, P)
    b0 = b0r + (6.045e-4) * (T - SP78_Tr)
    b1 = b1r + (28.6e-4) * (T - SP78_Tr)
    zK = +1
    zH2PO4 = -1
    Cphi = C0r * (2 * sqrt(np_abs(zK * zH2PO4))) - 10.11e-5*(T - SP78_Tr)
    C0 = Cphi / (2 * sqrt(np_abs(zK * zH2PO4)))
    alph1 = 2
    alph2 = -9
    omega = -9
    # Validity range declared by MP98
    valid = logical_and(T >= 283.15,T <= 313.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_SCN_SP78(T, P):
    """c-a: potassium thiocyanate [SP78]."""
    b0r, b1r, b2, C0r, C1, alph1, alph2, omega, _ = bC_K_SCN_PM73(T, P)
    b0 = b0r + 6.87e-4*(T - SP78_Tr)
    b1 = b1r + 37e-4*(T - SP78_Tr)
    zK = +1
    zSCN = -1
    Cphi = C0r * (2 * sqrt(np_abs(zK * zSCN))) + 0.43e-5*(T - SP78_Tr)
    C0 = Cphi / (2 * sqrt(np_abs(zK * zSCN)))
    alph1 = 2
    alph2 = -9
    omega = -9
    # Validity range declared by MP98
    valid = logical_and(T >= 283.15,T <= 313.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_SCN_SP78(T, P):
    """c-a: sodium thiocyanate [SP78]."""
    b0r, b1r, b2, C0, C1, alph1, alph2, omega, _ = bC_Na_SCN_PM73(T, P)
    b0 = b0r + 0.00078*(T - SP78_Tr)
    b1 = b1r + 0.002*(T - SP78_Tr)
    alph1 = 2
    alph2 = -9
    omega = -9
    # Validity range declared by MP98
    valid = logical_and(T >= 283.15,T <= 313.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Roy et al. (1980) ~~~~~
def theta_H_Mg_RGB80(T, P):
    """c-c': hydrogen magnesium [RGB80]."""
    # RGB80 do provide theta values at 5, 15, 25, 35 and 45 degC, but no
    #  equation to interpolate between them.
    # This function just returns the 25 degC value.
    theta = 0.0620
    valid = T == 298.15
    return theta, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Rard and Miller (1981i) ~~~~~
def bC_Mg_SO4_RM81i(T, P):
    """c-a: magnesium sulfate [RM81i]."""
    b0 = 0.21499
    b1 = 3.3646
    b2 = -32.743
    Cphi = 0.02797
    zMg  = +2
    zSO4 = -2
    C0 = Cphi / (2 * sqrt(np_abs(zMg * zSO4)))
    C1 = 0
    alph1 = 1.4
    alph2 = 12
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Peiper and Pitzer (1982) ~~~~~
def PP82_eqMPH(T,q):
    """PP82 equation derived by MP Humphreys."""
    Tr = 298.15
    return q[0] + q[1] * (T - Tr) + q[2] * (T - Tr)**2 / 2

def bC_Na_CO3_PP82(T, P):
    """c-a: sodium carbonate [PP82]."""
    # I have no idea where MP98 got their T**2 parameters from
    #   or why they are so small.
    b0 = PP82_eqMPH(T, float_([
         0.0362,
         1.79e-3,
        -4.22e-5,
    ]))
    b1 = PP82_eqMPH(T, float_([
         1.51,
         2.05e-3,
        -16.8e-5,
    ]))
    b2 = 0
    Cphi = 0.0052
    zNa = +1
    zCO3 = -2
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zCO3)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_HCO3_PP82(T, P):
    """c-a: sodium bicarbonate [PP82]."""
    # I have no idea where MP98 got their T**2 parameters from
    #   or why they are so small.
    b0 = PP82_eqMPH(T, float_([
         0.028,
         1.00e-3,
        -2.6e-5,
    ]))
    b1 = PP82_eqMPH(T, float_([
         0.044,
         1.10e-3,
        -4.3e-5,
    ]))
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def theta_Cl_HCO3_PP82(T, P):
    """a-a': chloride bicarbonate [PP82]."""
    theta = 0.0359
    valid = T == 298.15
    return theta, valid

def theta_Cl_CO3_PP82(T, P):
    """a-a': chloride carbonate [PP82]."""
    theta = -0.053
    valid = T == 298.15
    return theta, valid

def psi_Na_Cl_HCO3_PP82(T, P):
    """c-a-a': sodium chloride bicarbonate [PP82]."""
    psi = -0.0143
    valid = T == 298.15
    return psi, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Roy et al. (1982) ~~~~~
def theta_Ca_H_RGO82(T, P):
    """c-c': calcium hydrogen [RGB80]."""
    theta = 0.0612
    valid = T == 298.15
    return theta, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ de Lima and Pitzer (1983) ~~~~~
def bC_Mg_Cl_dLP83(T, P):
    """c-a: magnesium chloride [dLP83]."""
    # dLP83 Eq. (11)
    b0 = 5.93915e-7*T**2 - 9.31654e-4*T + 0.576066
    b1 = 2.60169e-5*T**2 - 1.09438e-2*T + 2.60135
    b2 = 0
    Cphi = 3.01823e-7 * T**2 - 2.89125e-4 * T + 6.57867e-2
    zMg = +2
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(zMg * zCl)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 298.15, T <= 523.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Holmes and Mesmer (1983) ~~~~~
def HM83_eq25(T, a):
    """HM83 equation 25."""
    TR = 298.15
    return a[0] \
         + a[1] * (1/T - 1/TR) \
         + a[2] * log(T/TR) \
         + a[3] * (T - TR) \
         + a[4] * (T**2 - TR**2) \
         + a[5] * log(T - 260)

def bC_Cs_Cl_HM83(T, P):
    """c-a: caesium chloride [HM83]."""
    b0 = HM83_eq25(T, float_([
         0.03352,
        -1290.0,
        -8.4279,
         0.018502,
        -6.7942e-6,
         0,
    ]))
    b1 = HM83_eq25(T, float_([
         0.0429,
        -38.0,
         0,
         0.001306,
         0, 0,
    ]))
    b2 = 0
    Cphi = HM83_eq25(T, float_([
        -2.62e-4,
         157.13,
         1.0860,
        -0.0025242,
         9.840e-7,
         0,
    ]))
    zCs = +1
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(zCs * zCl)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 523.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_Cl_HM83(T, P):
    """c-a: potassium chloride [HM83]."""
    b0 = HM83_eq25(T, float_([
         0.04808,
        -758.48,
        -4.7062,
         0.010072,
        -3.7599e-6,
         0,
    ]))
    b1 = HM83_eq25(T, float_([
        0.0476,
        303.09,
        1.066,
        0, 0,
        0.0470,
    ]))
    b2 = 0
    Cphi = HM83_eq25(T, float_([
        -7.88e-4,
         91.270,
         0.58643,
        -0.0012980,
         4.9567e-7,
         0,
    ]))
    zK = +1
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(zK * zCl)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 523.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Li_Cl_HM83(T, P):
    """c-a: lithium chloride [HM83]."""
    b0 = HM83_eq25(T, float_([
         0.14847,
         0, 0,
        -1.546e-4,
         0, 0,
    ]))
    b1 = HM83_eq25(T, float_([
        0.307,
        0, 0,
        6.36e-4,
        0, 0,
    ]))
    b2 = 0
    Cphi = HM83_eq25(T, float_([
         0.003710,
         4.115,
         0, 0,
        -3.71e-9,
         0,
    ]))
    zLi = +1
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(zLi * zCl)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 523.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Roy et al. (1983) ~~~~~
def bC_K_HCO3_RGW83(T, P):
    """c-a: potassium bicarbonate [RGW83]."""
    b0 = -0.022
    b1 = 0.09
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Harvie et al. (1984) ~~~~~
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
    zNa = 1
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zCl)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_SO4_HMW84(T, P):
    """c-a: sodium sulfate [HMW84]."""
    b0 = 0.01958
    b1 = 1.113
    b2 = 0.0
    Cphi = 0.00497
    zNa = 1
    zSO4 = -2
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zSO4)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_HSO4_HMW84(T, P):
    """c-a: sodium bisulfate [HMW84]."""
    b0 = 0.0454
    b1 = 0.398
    b2 = 0.0
    Cphi = 0.0
    zNa = 1
    zHSO4 = -1
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zHSO4)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_OH_HMW84(T, P):
    """c-a: sodium hydroxide [HMW84]."""
    b0 = 0.0864
    b1 = 0.253
    b2 = 0.0
    Cphi = 0.0044
    zNa = 1
    zOH = -1
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zOH)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_HCO3_HMW84(T, P):
    """c-a: sodium bicarbonate [HMW84]."""
    b0 = 0.0277
    b1 = 0.0411
    b2 = 0.0
    Cphi = 0.0
    zNa = 1
    zHCO3 = -1
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zHCO3)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_CO3_HMW84(T, P):
    """c-a: sodium carbonate [HMW84]."""
    b0 = 0.0399
    b1 = 1.389
    b2 = 0.0
    Cphi = 0.0044
    zNa = 1
    zCO3 = -2
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zCO3)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_Cl_HMW84(T, P):
    """c-a: potassium chloride [HMW84]."""
    b0 = 0.04835
    b1 = 0.2122
    b2 = 0.0
    Cphi = -0.00084
    zK = 1
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(zK * zCl)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_SO4_HMW84(T, P):
    """c-a: potassium sulfate [HMW84]."""
    b0 = 0.04995
    b1 = 0.7793
    b2 = 0.0
    Cphi = 0.0
    zK = 1
    zSO4 = -2
    C0 = Cphi / (2 * sqrt(np_abs(zK * zSO4)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_HSO4_HMW84(T, P):
    """c-a: potassium bisulfate [HMW84]."""
    b0 = -0.0003
    b1 = 0.1735
    b2 = 0.0
    Cphi = 0.0
    zK = 1
    zHSO4 = -1
    C0 = Cphi / (2 * sqrt(np_abs(zK * zHSO4)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_OH_HMW84(T, P):
    """c-a: potassium hydroxide [HMW84]."""
    b0 = 0.1298
    b1 = 0.32
    b2 = 0.0
    Cphi = 0.0041
    zK = 1
    zOH = -1
    C0 = Cphi / (2 * sqrt(np_abs(zK * zOH)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_HCO3_HMW84(T, P):
    """c-a: potassium bicarbonate [HMW84]."""
    b0 = 0.0296
    b1 = -0.013
    b2 = 0.0
    Cphi = -0.008
    zK = 1
    zHCO3 = -1
    C0 = Cphi / (2 * sqrt(np_abs(zK * zHCO3)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_CO3_HMW84(T, P):
    """c-a: potassium carbonate [HMW84]."""
    b0 = 0.1488
    b1 = 1.43
    b2 = 0.0
    Cphi = -0.0015
    zK = 1
    zCO3 = -2
    C0 = Cphi / (2 * sqrt(np_abs(zK * zCO3)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Ca_Cl_HMW84(T, P):
    """c-a: calcium chloride [HMW84]."""
    b0 = 0.3159
    b1 = 1.614
    b2 = 0.0
    Cphi = -0.00034
    zCa = 2
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(zCa * zCl)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Ca_SO4_HMW84(T, P):
    """c-a: calcium sulfate [HMW84]."""
    b0 = 0.2
    b1 = 3.1973
    b2 = -54.24
    Cphi = 0.0
    zCa = 2
    zSO4 = -2
    C0 = Cphi / (2 * sqrt(np_abs(zCa * zSO4)))
    C1 = 0
    alph1 = 1.4
    alph2 = 12
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Ca_HSO4_HMW84(T, P):
    """c-a: calcium bisulfate [HMW84]."""
    b0 = 0.2145
    b1 = 2.53
    b2 = 0.0
    Cphi = 0.0
    zCa = 2
    zHSO4 = -1
    C0 = Cphi / (2 * sqrt(np_abs(zCa * zHSO4)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Ca_OH_HMW84(T, P):
    """c-a: calcium hydroxide [HMW84]."""
    b0 = -0.1747
    b1 = -0.2303
    b2 = -5.72
    Cphi = 0.0
    zCa = 2
    zOH = -1
    C0 = Cphi / (2 * sqrt(np_abs(zCa * zOH)))
    C1 = 0
    alph1 = 1.4
    alph2 = 12
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Ca_HCO3_HMW84(T, P):
    """c-a: calcium bicarbonate [HMW84]."""
    b0 = 0.4
    b1 = 2.977
    b2 = 0.0
    Cphi = 0.0
    zCa = 2
    zHCO3 = -1
    C0 = Cphi / (2 * sqrt(np_abs(zCa * zHCO3)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Ca_CO3_HMW84(T, P):
    """c-a: calcium carbonate [HMW84]."""
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    Cphi = 0.0
    zCa = 2
    zCO3 = -2
    C0 = Cphi / (2 * sqrt(np_abs(zCa * zCO3)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Mg_Cl_HMW84(T, P):
    """c-a: magnesium chloride [HMW84]."""
    b0 = 0.35235
    b1 = 1.6815
    b2 = 0.0
    Cphi = 0.00519
    zMg = 2
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(zMg * zCl)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Mg_SO4_HMW84(T, P):
    """c-a: magnesium sulfate [HMW84]."""
    b0 = 0.221
    b1 = 3.343
    b2 = -37.23
    Cphi = 0.025
    zMg = 2
    zSO4 = -2
    C0 = Cphi / (2 * sqrt(np_abs(zMg * zSO4)))
    C1 = 0
    alph1 = 1.4
    alph2 = 12
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Mg_HSO4_HMW84(T, P):
    """c-a: magnesium bisulfate [HMW84]."""
    b0 = 0.4746
    b1 = 1.729
    b2 = 0.0
    Cphi = 0.0
    zMg = 2
    zHSO4 = -1
    C0 = Cphi / (2 * sqrt(np_abs(zMg * zHSO4)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Mg_OH_HMW84(T, P):
    """c-a: magnesium hydroxide [HMW84]."""
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    Cphi = 0.0
    zMg = 2
    zOH = -1
    C0 = Cphi / (2 * sqrt(np_abs(zMg * zOH)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Mg_HCO3_HMW84(T, P):
    """c-a: magnesium bicarbonate [HMW84]."""
    b0 = 0.329
    b1 = 0.6072
    b2 = 0.0
    Cphi = 0.0
    zMg = 2
    zHCO3 = -1
    C0 = Cphi / (2 * sqrt(np_abs(zMg * zHCO3)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Mg_CO3_HMW84(T, P):
    """c-a: magnesium carbonate [HMW84]."""
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    Cphi = 0.0
    zMg = 2
    zCO3 = -2
    C0 = Cphi / (2 * sqrt(np_abs(zMg * zCO3)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_MgOH_Cl_HMW84(T, P):
    """c-a: magnesium-hydroxide chloride [HMW84]."""
    b0 = -0.1
    b1 = 1.658
    b2 = 0.0
    Cphi = 0.0
    zMgOH = 1
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(zMgOH * zCl)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_MgOH_SO4_HMW84(T, P):
    """c-a: magnesium-hydroxide sulfate [HMW84]."""
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    Cphi = 0.0
    zMgOH = 1
    zSO4 = -2
    C0 = Cphi / (2 * sqrt(np_abs(zMgOH * zSO4)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_MgOH_HSO4_HMW84(T, P):
    """c-a: magnesium-hydroxide bisulfate [HMW84]."""
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    Cphi = 0.0
    zMgOH = 1
    zHSO4 = -1
    C0 = Cphi / (2 * sqrt(np_abs(zMgOH * zHSO4)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_MgOH_OH_HMW84(T, P):
    """c-a: magnesium-hydroxide hydroxide [HMW84]."""
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    Cphi = 0.0
    zMgOH = 1
    zOH = -1
    C0 = Cphi / (2 * sqrt(np_abs(zMgOH * zOH)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_MgOH_HCO3_HMW84(T, P):
    """c-a: magnesium-hydroxide bicarbonate [HMW84]."""
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    Cphi = 0.0
    zMgOH = 1
    zHCO3 = -1
    C0 = Cphi / (2 * sqrt(np_abs(zMgOH * zHCO3)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_MgOH_CO3_HMW84(T, P):
    """c-a: magnesium-hydroxide carbonate [HMW84]."""
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    Cphi = 0.0
    zMgOH = 1
    zCO3 = -2
    C0 = Cphi / (2 * sqrt(np_abs(zMgOH * zCO3)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_H_Cl_HMW84(T, P):
    """c-a: hydrogen chloride [HMW84]."""
    b0 = 0.1775
    b1 = 0.2945
    b2 = 0.0
    Cphi = 0.0008
    zH = 1
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(zH * zCl)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_H_SO4_HMW84(T, P):
    """c-a: hydrogen sulfate [HMW84]."""
    b0 = 0.0298
    b1 = 0.0
    b2 = 0.0
    Cphi = 0.0438
    zH = 1
    zSO4 = -2
    C0 = Cphi / (2 * sqrt(np_abs(zH * zSO4)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_H_HSO4_HMW84(T, P):
    """c-a: hydrogen bisulfate [HMW84]."""
    b0 = 0.2065
    b1 = 0.5556
    b2 = 0.0
    Cphi = 0.0
    zH = 1
    zHSO4 = -1
    C0 = Cphi / (2 * sqrt(np_abs(zH * zHSO4)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_H_OH_HMW84(T, P):
    """c-a: hydrogen hydroxide [HMW84]."""
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    Cphi = 0.0
    zH = 1
    zOH = -1
    C0 = Cphi / (2 * sqrt(np_abs(zH * zOH)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_H_HCO3_HMW84(T, P):
    """c-a: hydrogen bicarbonate [HMW84]."""
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    Cphi = 0.0
    zH = 1
    zHCO3 = -1
    C0 = Cphi / (2 * sqrt(np_abs(zH * zHCO3)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_H_CO3_HMW84(T, P):
    """c-a: hydrogen carbonate [HMW84]."""
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    Cphi = 0.0
    zH = 1
    zCO3 = -2
    C0 = Cphi / (2 * sqrt(np_abs(zH * zCO3)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def theta_Cl_SO4_HMW84(T, P):
    """a-a': chloride sulfate [HMW84]."""
    theta = 0.02
    valid = T == 298.15
    return theta, valid

def psi_Na_Cl_SO4_HMW84(T, P):
    """c-a-a': sodium chloride sulfate [HMW84]."""
    psi = 0.0014
    valid = T == 298.15
    return psi, valid

def psi_K_Cl_SO4_HMW84(T, P):
    """c-a-a': potassium chloride sulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Ca_Cl_SO4_HMW84(T, P):
    """c-a-a': calcium chloride sulfate [HMW84]."""
    psi = -0.018
    valid = T == 298.15
    return psi, valid

def psi_Mg_Cl_SO4_HMW84(T, P):
    """c-a-a': magnesium chloride sulfate [HMW84]."""
    psi = -0.004
    valid = T == 298.15
    return psi, valid

def psi_MgOH_Cl_SO4_HMW84(T, P):
    """c-a-a': magnesium-hydroxide chloride sulfate [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_H_Cl_SO4_HMW84(T, P):
    """c-a-a': hydrogen chloride sulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_Cl_HSO4_HMW84(T, P):
    """a-a': chloride bisulfate [HMW84]."""
    theta = -0.006
    valid = T == 298.15
    return theta, valid

def psi_Na_Cl_HSO4_HMW84(T, P):
    """c-a-a': sodium chloride bisulfate [HMW84]."""
    psi = -0.006
    valid = T == 298.15
    return psi, valid

def psi_K_Cl_HSO4_HMW84(T, P):
    """c-a-a': potassium chloride bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Ca_Cl_HSO4_HMW84(T, P):
    """c-a-a': calcium chloride bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Mg_Cl_HSO4_HMW84(T, P):
    """c-a-a': magnesium chloride bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_MgOH_Cl_HSO4_HMW84(T, P):
    """c-a-a': magnesium-hydroxide chloride bisulfate [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_H_Cl_HSO4_HMW84(T, P):
    """c-a-a': hydrogen chloride bisulfate [HMW84]."""
    psi = 0.013
    valid = T == 298.15
    return psi, valid

def theta_Cl_OH_HMW84(T, P):
    """a-a': chloride hydroxide [HMW84]."""
    theta = -0.05
    valid = T == 298.15
    return theta, valid

def psi_Na_Cl_OH_HMW84(T, P):
    """c-a-a': sodium chloride hydroxide [HMW84]."""
    psi = -0.006
    valid = T == 298.15
    return psi, valid

def psi_K_Cl_OH_HMW84(T, P):
    """c-a-a': potassium chloride hydroxide [HMW84]."""
    psi = -0.006
    valid = T == 298.15
    return psi, valid

def psi_Ca_Cl_OH_HMW84(T, P):
    """c-a-a': calcium chloride hydroxide [HMW84]."""
    psi = -0.025
    valid = T == 298.15
    return psi, valid

def psi_Mg_Cl_OH_HMW84(T, P):
    """c-a-a': magnesium chloride hydroxide [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_MgOH_Cl_OH_HMW84(T, P):
    """c-a-a': magnesium-hydroxide chloride hydroxide [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_H_Cl_OH_HMW84(T, P):
    """c-a-a': hydrogen chloride hydroxide [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_Cl_HCO3_HMW84(T, P):
    """a-a': chloride bicarbonate [HMW84]."""
    theta = 0.03
    valid = T == 298.15
    return theta, valid

def psi_Na_Cl_HCO3_HMW84(T, P):
    """c-a-a': sodium chloride bicarbonate [HMW84]."""
    psi = -0.15
    valid = T == 298.15
    return psi, valid

def psi_K_Cl_HCO3_HMW84(T, P):
    """c-a-a': potassium chloride bicarbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Ca_Cl_HCO3_HMW84(T, P):
    """c-a-a': calcium chloride bicarbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Mg_Cl_HCO3_HMW84(T, P):
    """c-a-a': magnesium chloride bicarbonate [HMW84]."""
    psi = -0.096
    valid = T == 298.15
    return psi, valid

def psi_MgOH_Cl_HCO3_HMW84(T, P):
    """c-a-a': magnesium-hydroxide chloride bicarbonate [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_H_Cl_HCO3_HMW84(T, P):
    """c-a-a': hydrogen chloride bicarbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_CO3_Cl_HMW84(T, P):
    """a-a': carbonate chloride [HMW84]."""
    theta = -0.02
    valid = T == 298.15
    return theta, valid

def psi_Na_CO3_Cl_HMW84(T, P):
    """c-a-a': sodium carbonate chloride [HMW84]."""
    psi = 0.0085
    valid = T == 298.15
    return psi, valid

def psi_K_CO3_Cl_HMW84(T, P):
    """c-a-a': potassium carbonate chloride [HMW84]."""
    psi = 0.004
    valid = T == 298.15
    return psi, valid

def psi_Ca_CO3_Cl_HMW84(T, P):
    """c-a-a': calcium carbonate chloride [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Mg_CO3_Cl_HMW84(T, P):
    """c-a-a': magnesium carbonate chloride [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_MgOH_CO3_Cl_HMW84(T, P):
    """c-a-a': magnesium-hydroxide carbonate chloride [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_H_CO3_Cl_HMW84(T, P):
    """c-a-a': hydrogen carbonate chloride [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_HSO4_SO4_HMW84(T, P):
    """a-a': bisulfate sulfate [HMW84]."""
    theta = 0.0
    valid = T == 298.15
    return theta, valid

def psi_Na_HSO4_SO4_HMW84(T, P):
    """c-a-a': sodium bisulfate sulfate [HMW84]."""
    psi = -0.0094
    valid = T == 298.15
    return psi, valid

def psi_K_HSO4_SO4_HMW84(T, P):
    """c-a-a': potassium bisulfate sulfate [HMW84]."""
    psi = -0.0677
    valid = T == 298.15
    return psi, valid

def psi_Ca_HSO4_SO4_HMW84(T, P):
    """c-a-a': calcium bisulfate sulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Mg_HSO4_SO4_HMW84(T, P):
    """c-a-a': magnesium bisulfate sulfate [HMW84]."""
    psi = -0.0425
    valid = T == 298.15
    return psi, valid

def psi_MgOH_HSO4_SO4_HMW84(T, P):
    """c-a-a': magnesium-hydroxide bisulfate sulfate [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_H_HSO4_SO4_HMW84(T, P):
    """c-a-a': hydrogen bisulfate sulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_OH_SO4_HMW84(T, P):
    """a-a': hydroxide sulfate [HMW84]."""
    theta = -0.013
    valid = T == 298.15
    return theta, valid

def psi_Na_OH_SO4_HMW84(T, P):
    """c-a-a': sodium hydroxide sulfate [HMW84]."""
    psi = -0.009
    valid = T == 298.15
    return psi, valid

def psi_K_OH_SO4_HMW84(T, P):
    """c-a-a': potassium hydroxide sulfate [HMW84]."""
    psi = -0.05
    valid = T == 298.15
    return psi, valid

def psi_Ca_OH_SO4_HMW84(T, P):
    """c-a-a': calcium hydroxide sulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Mg_OH_SO4_HMW84(T, P):
    """c-a-a': magnesium hydroxide sulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_MgOH_OH_SO4_HMW84(T, P):
    """c-a-a': magnesium-hydroxide hydroxide sulfate [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_H_OH_SO4_HMW84(T, P):
    """c-a-a': hydrogen hydroxide sulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_HCO3_SO4_HMW84(T, P):
    """a-a': bicarbonate sulfate [HMW84]."""
    theta = 0.01
    valid = T == 298.15
    return theta, valid

def psi_Na_HCO3_SO4_HMW84(T, P):
    """c-a-a': sodium bicarbonate sulfate [HMW84]."""
    psi = -0.005
    valid = T == 298.15
    return psi, valid

def psi_K_HCO3_SO4_HMW84(T, P):
    """c-a-a': potassium bicarbonate sulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Ca_HCO3_SO4_HMW84(T, P):
    """c-a-a': calcium bicarbonate sulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Mg_HCO3_SO4_HMW84(T, P):
    """c-a-a': magnesium bicarbonate sulfate [HMW84]."""
    psi = -0.161
    valid = T == 298.15
    return psi, valid

def psi_MgOH_HCO3_SO4_HMW84(T, P):
    """c-a-a': magnesium-hydroxide bicarbonate sulfate [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_H_HCO3_SO4_HMW84(T, P):
    """c-a-a': hydrogen bicarbonate sulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_CO3_SO4_HMW84(T, P):
    """a-a': carbonate sulfate [HMW84]."""
    theta = 0.02
    valid = T == 298.15
    return theta, valid

def psi_Na_CO3_SO4_HMW84(T, P):
    """c-a-a': sodium carbonate sulfate [HMW84]."""
    psi = -0.005
    valid = T == 298.15
    return psi, valid

def psi_K_CO3_SO4_HMW84(T, P):
    """c-a-a': potassium carbonate sulfate [HMW84]."""
    psi = -0.009
    valid = T == 298.15
    return psi, valid

def psi_Ca_CO3_SO4_HMW84(T, P):
    """c-a-a': calcium carbonate sulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Mg_CO3_SO4_HMW84(T, P):
    """c-a-a': magnesium carbonate sulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_MgOH_CO3_SO4_HMW84(T, P):
    """c-a-a': magnesium-hydroxide carbonate sulfate [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_H_CO3_SO4_HMW84(T, P):
    """c-a-a': hydrogen carbonate sulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_HSO4_OH_HMW84(T, P):
    """a-a': bisulfate hydroxide [HMW84]."""
    theta = 0.0
    valid = T == 298.15
    return theta, valid

def psi_Na_HSO4_OH_HMW84(T, P):
    """c-a-a': sodium bisulfate hydroxide [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_K_HSO4_OH_HMW84(T, P):
    """c-a-a': potassium bisulfate hydroxide [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Ca_HSO4_OH_HMW84(T, P):
    """c-a-a': calcium bisulfate hydroxide [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Mg_HSO4_OH_HMW84(T, P):
    """c-a-a': magnesium bisulfate hydroxide [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_MgOH_HSO4_OH_HMW84(T, P):
    """c-a-a': magnesium-hydroxide bisulfate hydroxide [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_H_HSO4_OH_HMW84(T, P):
    """c-a-a': hydrogen bisulfate hydroxide [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_HCO3_HSO4_HMW84(T, P):
    """a-a': bicarbonate bisulfate [HMW84]."""
    theta = 0.0
    valid = T == 298.15
    return theta, valid

def psi_Na_HCO3_HSO4_HMW84(T, P):
    """c-a-a': sodium bicarbonate bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_K_HCO3_HSO4_HMW84(T, P):
    """c-a-a': potassium bicarbonate bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Ca_HCO3_HSO4_HMW84(T, P):
    """c-a-a': calcium bicarbonate bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Mg_HCO3_HSO4_HMW84(T, P):
    """c-a-a': magnesium bicarbonate bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_MgOH_HCO3_HSO4_HMW84(T, P):
    """c-a-a': magnesium-hydroxide bicarbonate bisulfate [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_H_HCO3_HSO4_HMW84(T, P):
    """c-a-a': hydrogen bicarbonate bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_CO3_HSO4_HMW84(T, P):
    """a-a': carbonate bisulfate [HMW84]."""
    theta = 0.0
    valid = T == 298.15
    return theta, valid

def psi_Na_CO3_HSO4_HMW84(T, P):
    """c-a-a': sodium carbonate bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_K_CO3_HSO4_HMW84(T, P):
    """c-a-a': potassium carbonate bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Ca_CO3_HSO4_HMW84(T, P):
    """c-a-a': calcium carbonate bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Mg_CO3_HSO4_HMW84(T, P):
    """c-a-a': magnesium carbonate bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_MgOH_CO3_HSO4_HMW84(T, P):
    """c-a-a': magnesium-hydroxide carbonate bisulfate [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_H_CO3_HSO4_HMW84(T, P):
    """c-a-a': hydrogen carbonate bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_HCO3_OH_HMW84(T, P):
    """a-a': bicarbonate hydroxide [HMW84]."""
    theta = 0.0
    valid = T == 298.15
    return theta, valid

def psi_Na_HCO3_OH_HMW84(T, P):
    """c-a-a': sodium bicarbonate hydroxide [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_K_HCO3_OH_HMW84(T, P):
    """c-a-a': potassium bicarbonate hydroxide [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Ca_HCO3_OH_HMW84(T, P):
    """c-a-a': calcium bicarbonate hydroxide [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Mg_HCO3_OH_HMW84(T, P):
    """c-a-a': magnesium bicarbonate hydroxide [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_MgOH_HCO3_OH_HMW84(T, P):
    """c-a-a': magnesium-hydroxide bicarbonate hydroxide [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_H_HCO3_OH_HMW84(T, P):
    """c-a-a': hydrogen bicarbonate hydroxide [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_CO3_OH_HMW84(T, P):
    """a-a': carbonate hydroxide [HMW84]."""
    theta = 0.1
    valid = T == 298.15
    return theta, valid

def psi_Na_CO3_OH_HMW84(T, P):
    """c-a-a': sodium carbonate hydroxide [HMW84]."""
    psi = -0.017
    valid = T == 298.15
    return psi, valid

def psi_K_CO3_OH_HMW84(T, P):
    """c-a-a': potassium carbonate hydroxide [HMW84]."""
    psi = -0.01
    valid = T == 298.15
    return psi, valid

def psi_Ca_CO3_OH_HMW84(T, P):
    """c-a-a': calcium carbonate hydroxide [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Mg_CO3_OH_HMW84(T, P):
    """c-a-a': magnesium carbonate hydroxide [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_MgOH_CO3_OH_HMW84(T, P):
    """c-a-a': magnesium-hydroxide carbonate hydroxide [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_H_CO3_OH_HMW84(T, P):
    """c-a-a': hydrogen carbonate hydroxide [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_CO3_HCO3_HMW84(T, P):
    """a-a': carbonate bicarbonate [HMW84]."""
    theta = -0.04
    valid = T == 298.15
    return theta, valid

def psi_Na_CO3_HCO3_HMW84(T, P):
    """c-a-a': sodium carbonate bicarbonate [HMW84]."""
    psi = 0.002
    valid = T == 298.15
    return psi, valid

def psi_K_CO3_HCO3_HMW84(T, P):
    """c-a-a': potassium carbonate bicarbonate [HMW84]."""
    psi = 0.012
    valid = T == 298.15
    return psi, valid

def psi_Ca_CO3_HCO3_HMW84(T, P):
    """c-a-a': calcium carbonate bicarbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Mg_CO3_HCO3_HMW84(T, P):
    """c-a-a': magnesium carbonate bicarbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_MgOH_CO3_HCO3_HMW84(T, P):
    """c-a-a': magnesium-hydroxide carbonate bicarbonate [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_H_CO3_HCO3_HMW84(T, P):
    """c-a-a': hydrogen carbonate bicarbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_K_Na_HMW84(T, P):
    """c-c': potassium sodium [HMW84]."""
    theta = -0.012
    valid = T == 298.15
    return theta, valid

def psi_K_Na_Cl_HMW84(T, P):
    """c-c'-a: potassium sodium chloride [HMW84]."""
    psi = -0.0018
    valid = T == 298.15
    return psi, valid

def psi_K_Na_SO4_HMW84(T, P):
    """c-c'-a: potassium sodium sulfate [HMW84]."""
    psi = -0.01
    valid = T == 298.15
    return psi, valid

def psi_K_Na_HSO4_HMW84(T, P):
    """c-c'-a: potassium sodium bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_K_Na_OH_HMW84(T, P):
    """c-c'-a: potassium sodium hydroxide [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_K_Na_HCO3_HMW84(T, P):
    """c-c'-a: potassium sodium bicarbonate [HMW84]."""
    psi = -0.003
    valid = T == 298.15
    return psi, valid

def psi_K_Na_CO3_HMW84(T, P):
    """c-c'-a: potassium sodium carbonate [HMW84]."""
    psi = 0.003
    valid = T == 298.15
    return psi, valid

def theta_Ca_Na_HMW84(T, P):
    """c-c': calcium sodium [HMW84]."""
    theta = 0.07
    valid = T == 298.15
    return theta, valid

def psi_Ca_Na_Cl_HMW84(T, P):
    """c-c'-a: calcium sodium chloride [HMW84]."""
    psi = -0.007
    valid = T == 298.15
    return psi, valid

def psi_Ca_Na_SO4_HMW84(T, P):
    """c-c'-a: calcium sodium sulfate [HMW84]."""
    psi = -0.055
    valid = T == 298.15
    return psi, valid

def psi_Ca_Na_HSO4_HMW84(T, P):
    """c-c'-a: calcium sodium bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Ca_Na_OH_HMW84(T, P):
    """c-c'-a: calcium sodium hydroxide [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_Ca_Na_HCO3_HMW84(T, P):
    """c-c'-a: calcium sodium bicarbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Ca_Na_CO3_HMW84(T, P):
    """c-c'-a: calcium sodium carbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_Mg_Na_HMW84(T, P):
    """c-c': magnesium sodium [HMW84]."""
    theta = 0.07
    valid = T == 298.15
    return theta, valid

def psi_Mg_Na_Cl_HMW84(T, P):
    """c-c'-a: magnesium sodium chloride [HMW84]."""
    psi = -0.012
    valid = T == 298.15
    return psi, valid

def psi_Mg_Na_SO4_HMW84(T, P):
    """c-c'-a: magnesium sodium sulfate [HMW84]."""
    psi = -0.015
    valid = T == 298.15
    return psi, valid

def psi_Mg_Na_HSO4_HMW84(T, P):
    """c-c'-a: magnesium sodium bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Mg_Na_OH_HMW84(T, P):
    """c-c'-a: magnesium sodium hydroxide [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_Mg_Na_HCO3_HMW84(T, P):
    """c-c'-a: magnesium sodium bicarbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Mg_Na_CO3_HMW84(T, P):
    """c-c'-a: magnesium sodium carbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_MgOH_Na_HMW84(T, P):
    """c-c': magnesium-hydroxide sodium [HMW84]."""
    theta = 0.0
    valid = T == 298.15
    return theta, valid

def psi_MgOH_Na_Cl_HMW84(T, P):
    """c-c'-a: magnesium-hydroxide sodium chloride [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_MgOH_Na_SO4_HMW84(T, P):
    """c-c'-a: magnesium-hydroxide sodium sulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_MgOH_Na_HSO4_HMW84(T, P):
    """c-c'-a: magnesium-hydroxide sodium bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_MgOH_Na_OH_HMW84(T, P):
    """c-c'-a: magnesium-hydroxide sodium hydroxide [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_MgOH_Na_HCO3_HMW84(T, P):
    """c-c'-a: magnesium-hydroxide sodium bicarbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_MgOH_Na_CO3_HMW84(T, P):
    """c-c'-a: magnesium-hydroxide sodium carbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_H_Na_HMW84(T, P):
    """c-c': hydrogen sodium [HMW84]."""
    theta = 0.036
    valid = T == 298.15
    return theta, valid

def psi_H_Na_Cl_HMW84(T, P):
    """c-c'-a: hydrogen sodium chloride [HMW84]."""
    psi = -0.004
    valid = T == 298.15
    return psi, valid

def psi_H_Na_SO4_HMW84(T, P):
    """c-c'-a: hydrogen sodium sulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_H_Na_HSO4_HMW84(T, P):
    """c-c'-a: hydrogen sodium bisulfate [HMW84]."""
    psi = -0.0129
    valid = T == 298.15
    return psi, valid

def psi_H_Na_OH_HMW84(T, P):
    """c-c'-a: hydrogen sodium hydroxide [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_H_Na_HCO3_HMW84(T, P):
    """c-c'-a: hydrogen sodium bicarbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_H_Na_CO3_HMW84(T, P):
    """c-c'-a: hydrogen sodium carbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_Ca_K_HMW84(T, P):
    """c-c': calcium potassium [HMW84]."""
    theta = 0.032
    valid = T == 298.15
    return theta, valid

def psi_Ca_K_Cl_HMW84(T, P):
    """c-c'-a: calcium potassium chloride [HMW84]."""
    psi = -0.025
    valid = T == 298.15
    return psi, valid

def psi_Ca_K_SO4_HMW84(T, P):
    """c-c'-a: calcium potassium sulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Ca_K_HSO4_HMW84(T, P):
    """c-c'-a: calcium potassium bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Ca_K_OH_HMW84(T, P):
    """c-c'-a: calcium potassium hydroxide [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_Ca_K_HCO3_HMW84(T, P):
    """c-c'-a: calcium potassium bicarbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Ca_K_CO3_HMW84(T, P):
    """c-c'-a: calcium potassium carbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_K_Mg_HMW84(T, P):
    """c-c': potassium magnesium [HMW84]."""
    theta = 0.0
    valid = T == 298.15
    return theta, valid

def psi_K_Mg_Cl_HMW84(T, P):
    """c-c'-a: potassium magnesium chloride [HMW84]."""
    psi = -0.022
    valid = T == 298.15
    return psi, valid

def psi_K_Mg_SO4_HMW84(T, P):
    """c-c'-a: potassium magnesium sulfate [HMW84]."""
    psi = -0.048
    valid = T == 298.15
    return psi, valid

def psi_K_Mg_HSO4_HMW84(T, P):
    """c-c'-a: potassium magnesium bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_K_Mg_OH_HMW84(T, P):
    """c-c'-a: potassium magnesium hydroxide [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_K_Mg_HCO3_HMW84(T, P):
    """c-c'-a: potassium magnesium bicarbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_K_Mg_CO3_HMW84(T, P):
    """c-c'-a: potassium magnesium carbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_K_MgOH_HMW84(T, P):
    """c-c': potassium magnesium-hydroxide [HMW84]."""
    theta = 0.0
    valid = T == 298.15
    return theta, valid

def psi_K_MgOH_Cl_HMW84(T, P):
    """c-c'-a: potassium magnesium-hydroxide chloride [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_K_MgOH_SO4_HMW84(T, P):
    """c-c'-a: potassium magnesium-hydroxide sulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_K_MgOH_HSO4_HMW84(T, P):
    """c-c'-a: potassium magnesium-hydroxide bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_K_MgOH_OH_HMW84(T, P):
    """c-c'-a: potassium magnesium-hydroxide hydroxide [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_K_MgOH_HCO3_HMW84(T, P):
    """c-c'-a: potassium magnesium-hydroxide bicarbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_K_MgOH_CO3_HMW84(T, P):
    """c-c'-a: potassium magnesium-hydroxide carbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_H_K_HMW84(T, P):
    """c-c': hydrogen potassium [HMW84]."""
    theta = 0.005
    valid = T == 298.15
    return theta, valid

def psi_H_K_Cl_HMW84(T, P):
    """c-c'-a: hydrogen potassium chloride [HMW84]."""
    psi = -0.011
    valid = T == 298.15
    return psi, valid

def psi_H_K_SO4_HMW84(T, P):
    """c-c'-a: hydrogen potassium sulfate [HMW84]."""
    psi = 0.197
    valid = T == 298.15
    return psi, valid

def psi_H_K_HSO4_HMW84(T, P):
    """c-c'-a: hydrogen potassium bisulfate [HMW84]."""
    psi = -0.0265
    valid = T == 298.15
    return psi, valid

def psi_H_K_OH_HMW84(T, P):
    """c-c'-a: hydrogen potassium hydroxide [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_H_K_HCO3_HMW84(T, P):
    """c-c'-a: hydrogen potassium bicarbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_H_K_CO3_HMW84(T, P):
    """c-c'-a: hydrogen potassium carbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_Ca_Mg_HMW84(T, P):
    """c-c': calcium magnesium [HMW84]."""
    theta = 0.007
    valid = T == 298.15
    return theta, valid

def psi_Ca_Mg_Cl_HMW84(T, P):
    """c-c'-a: calcium magnesium chloride [HMW84]."""
    psi = -0.012
    valid = T == 298.15
    return psi, valid

def psi_Ca_Mg_SO4_HMW84(T, P):
    """c-c'-a: calcium magnesium sulfate [HMW84]."""
    psi = 0.024
    valid = T == 298.15
    return psi, valid

def psi_Ca_Mg_HSO4_HMW84(T, P):
    """c-c'-a: calcium magnesium bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Ca_Mg_OH_HMW84(T, P):
    """c-c'-a: calcium magnesium hydroxide [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_Ca_Mg_HCO3_HMW84(T, P):
    """c-c'-a: calcium magnesium bicarbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Ca_Mg_CO3_HMW84(T, P):
    """c-c'-a: calcium magnesium carbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_Ca_MgOH_HMW84(T, P):
    """c-c': calcium magnesium-hydroxide [HMW84]."""
    theta = 0.0
    valid = T == 298.15
    return theta, valid

def psi_Ca_MgOH_Cl_HMW84(T, P):
    """c-c'-a: calcium magnesium-hydroxide chloride [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Ca_MgOH_SO4_HMW84(T, P):
    """c-c'-a: calcium magnesium-hydroxide sulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Ca_MgOH_HSO4_HMW84(T, P):
    """c-c'-a: calcium magnesium-hydroxide bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Ca_MgOH_OH_HMW84(T, P):
    """c-c'-a: calcium magnesium-hydroxide hydroxide [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_Ca_MgOH_HCO3_HMW84(T, P):
    """c-c'-a: calcium magnesium-hydroxide bicarbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Ca_MgOH_CO3_HMW84(T, P):
    """c-c'-a: calcium magnesium-hydroxide carbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_Ca_H_HMW84(T, P):
    """c-c': calcium hydrogen [HMW84]."""
    theta = 0.092
    valid = T == 298.15
    return theta, valid

def psi_Ca_H_Cl_HMW84(T, P):
    """c-c'-a: calcium hydrogen chloride [HMW84]."""
    psi = -0.015
    valid = T == 298.15
    return psi, valid

def psi_Ca_H_SO4_HMW84(T, P):
    """c-c'-a: calcium hydrogen sulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Ca_H_HSO4_HMW84(T, P):
    """c-c'-a: calcium hydrogen bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Ca_H_OH_HMW84(T, P):
    """c-c'-a: calcium hydrogen hydroxide [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_Ca_H_HCO3_HMW84(T, P):
    """c-c'-a: calcium hydrogen bicarbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Ca_H_CO3_HMW84(T, P):
    """c-c'-a: calcium hydrogen carbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_Mg_MgOH_HMW84(T, P):
    """c-c': magnesium magnesium-hydroxide [HMW84]."""
    theta = 0.0
    valid = T == 298.15
    return theta, valid

def psi_Mg_MgOH_Cl_HMW84(T, P):
    """c-c'-a: magnesium magnesium-hydroxide chloride [HMW84]."""
    psi = 0.028
    valid = T == 298.15
    return psi, valid

def psi_Mg_MgOH_SO4_HMW84(T, P):
    """c-c'-a: magnesium magnesium-hydroxide sulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Mg_MgOH_HSO4_HMW84(T, P):
    """c-c'-a: magnesium magnesium-hydroxide bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Mg_MgOH_OH_HMW84(T, P):
    """c-c'-a: magnesium magnesium-hydroxide hydroxide [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_Mg_MgOH_HCO3_HMW84(T, P):
    """c-c'-a: magnesium magnesium-hydroxide bicarbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_Mg_MgOH_CO3_HMW84(T, P):
    """c-c'-a: magnesium magnesium-hydroxide carbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_H_Mg_HMW84(T, P):
    """c-c': hydrogen magnesium [HMW84]."""
    theta = 0.1
    valid = T == 298.15
    return theta, valid

def psi_H_Mg_Cl_HMW84(T, P):
    """c-c'-a: hydrogen magnesium chloride [HMW84]."""
    psi = -0.011
    valid = T == 298.15
    return psi, valid

def psi_H_Mg_SO4_HMW84(T, P):
    """c-c'-a: hydrogen magnesium sulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_H_Mg_HSO4_HMW84(T, P):
    """c-c'-a: hydrogen magnesium bisulfate [HMW84]."""
    psi = -0.0178
    valid = T == 298.15
    return psi, valid

def psi_H_Mg_OH_HMW84(T, P):
    """c-c'-a: hydrogen magnesium hydroxide [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_H_Mg_HCO3_HMW84(T, P):
    """c-c'-a: hydrogen magnesium bicarbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_H_Mg_CO3_HMW84(T, P):
    """c-c'-a: hydrogen magnesium carbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def theta_H_MgOH_HMW84(T, P):
    """c-c': hydrogen magnesium-hydroxide [HMW84]."""
    theta = 0.0
    valid = T == 298.15
    return theta, valid

def psi_H_MgOH_Cl_HMW84(T, P):
    """c-c'-a: hydrogen magnesium-hydroxide chloride [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_H_MgOH_SO4_HMW84(T, P):
    """c-c'-a: hydrogen magnesium-hydroxide sulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_H_MgOH_HSO4_HMW84(T, P):
    """c-c'-a: hydrogen magnesium-hydroxide bisulfate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_H_MgOH_OH_HMW84(T, P):
    """c-c'-a: hydrogen magnesium-hydroxide hydroxide [HMW84]."""
    psi = 0
    valid = T == 298.15
    return psi, valid

def psi_H_MgOH_HCO3_HMW84(T, P):
    """c-c'-a: hydrogen magnesium-hydroxide bicarbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def psi_H_MgOH_CO3_HMW84(T, P):
    """c-c'-a: hydrogen magnesium-hydroxide carbonate [HMW84]."""
    psi = 0.0
    valid = T == 298.15
    return psi, valid

def lambd_CO2_H_HMW84(T, P):
    """n-c: carbon-dioxide hydrogen [HMW84]."""
    lambd = 0.0
    valid = T == 298.15
    return lambd, valid

def lambd_CO2_Na_HMW84(T, P):
    """n-c: carbon-dioxide sodium [HMW84]."""
    lambd = 0.1
    valid = T == 298.15
    return lambd, valid

def lambd_CO2_K_HMW84(T, P):
    """n-c: carbon-dioxide potassium [HMW84]."""
    lambd = 0.051
    valid = T == 298.15
    return lambd, valid

def lambd_CO2_Ca_HMW84(T, P):
    """n-c: carbon-dioxide calcium [HMW84]."""
    lambd = 0.183
    valid = T == 298.15
    return lambd, valid

def lambd_CO2_Mg_HMW84(T, P):
    """n-c: carbon-dioxide magnesium [HMW84]."""
    lambd = 0.183
    valid = T == 298.15
    return lambd, valid

def lambd_CO2_Cl_HMW84(T, P):
    """n-a: carbon-dioxide chloride [HMW84]."""
    lambd = -0.005
    valid = T == 298.15
    return lambd, valid

def lambd_CO2_SO4_HMW84(T, P):
    """n-a: carbon-dioxide sulfate [HMW84]."""
    lambd = 0.097
    valid = T == 298.15
    return lambd, valid

def lambd_CO2_HSO4_HMW84(T, P):
    """n-c: carbon-dioxide bisulfate [HMW84]."""
    lambd = -0.003
    valid = T == 298.15
    return lambd, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Phutela and Pitzer (1986) ~~~~~
PP86ii_Tr = 298.15

def PP86ii_eq28(T, q):
    """PP86ii equation 28."""
    Tr = PP86ii_Tr
    return ((T**2 - Tr**2) * q[0] / 2 \
          + (T**3 - Tr**3) * q[1] / 3 \
          + (T**4 - Tr**4) * q[2] / 4 \
          + (T**5 - Tr**5) * q[3] / 5 \
          +         Tr**2  * q[4]) / T**2

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
    return q[0] * (T   / 2 + Tr**2/(2*T) - Tr     ) \
         + q[1] * (T**2/ 6 + Tr**3/(3*T) - Tr**2/2) \
         + q[2] * (T**3/12 + Tr**4/(4*T) - Tr**3/3) \
         + q[3] * (t**5 + 4 - 5*t) * Tr**5 / (20 * T) \
         + q[4] * (Tr - Tr**2/T) \
         + q[5]

def bC_Mg_SO4_PP86ii(T, P):
    """c-a: magnesium sulfate [PP86ii]."""
    b0r, b1r, b2r, C0r, C1, alph1, alph2, omega, _ = bC_Mg_SO4_RM81i(T, P)
    b0 = PP86ii_eq29(T, float_([
        -1.0282,
         8.4790e-3,
        -2.3366e-5,
         2.1575e-8,
         6.8402e-4,
         b0r,
    ]))
    b1 = PP86ii_eq29(T, float_([
        -2.9596e-1,
         9.4564e-4,
         0, 0,
         1.1028e-2,
         b1r,
    ]))
    b2 = PP86ii_eq29(T, float_([
        -1.3764e+1,
         1.2121e-1,
        -2.7642e-4,
         0,
        -2.1515e-1,
         b2r,
    ]))
    C0 = PP86ii_eq29(T, float_([
         1.0541e-1,
        -8.9316e-4,
         2.5100e-6,
        -2.3436e-9,
        -8.7899e-5,
         C0r,
    ]))
    valid = T <= 473
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Holmes and Mesmer (1986) ~~~~~
# Note that HM86 use alph1 of 1.4 even where there is no beta2 term (p. 502)
# Also HM86 contains functions for caesium and lithium sulfates, not yet coded
def HM86_eq8(T, a):
    """HM86 equation 8."""
    TR = 298.15
    # Typo in a[5] term in HM86 has been corrected here
    return a[0] \
         + a[1] * (TR - TR**2/T) \
         + a[2] * (T**2 + 2*TR**3/T -3*TR**2) \
         + a[3] * (T + TR**2/T - 2*TR) \
         + a[4] * (log(T/TR) + TR/T - 1) \
         + a[5] * (1/(T - 263) + (263*T - TR**2) / (T*(TR - 263)**2)) \
         + a[6] * (1/(680 - T) + (TR**2 - 680*T) / (T*(680 - TR)**2))

def bC_K_SO4_HM86(T, P):
    """c-a: potassium sulfate [HM86]."""
    b0 = HM86_eq8(T, float_([
         0,
         7.476e-4,
         0,
         4.265e-3,
        -3.088,
         0, 0,
    ]))
    b1 = HM86_eq8(T, float_([
         0.6179,
         6.85e-3,
         5.576e-5,
        -5.841e-2,
         0,
        -0.90,
         0,
    ]))
    b2 = 0
    Cphi = HM86_eq8(T, float_([
         9.15467e-3,
         0, 0,
        -1.81e-4,
         0, 0, 0,
    ]))
    zK = +1
    zSO4 = -2
    C0 = Cphi / (2 * sqrt(np_abs(zK * zSO4)))
    C1 = 0
    alph1 = 1.4
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 298.15, T <= 523.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_SO4_HM86(T, P):
    """c-a: sodium sulfate [HM86]."""
    b0 = HM86_eq8(T, float_([
        -1.727e-2,
         1.7828e-3,
         9.133e-6,
         0,
        -6.552,
         0,
        -96.90,
    ]))
    b1 = HM86_eq8(T, float_([
         0.7534,
         5.61e-3,
        -5.7513e-4,
         1.11068,
        -378.82,
         0,
         1861.3,
    ]))
    b2 = 0
    Cphi = HM86_eq8(T, float_([
         1.1745e-2,
        -3.3038e-4,
         1.85794e-5,
        -3.9200e-2,
         14.2130,
         0,
        -24.950,
    ]))
    zNa = +1
    zSO4 = -2
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zSO4)))
    C1 = 0
    alph1 = 1.4
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 298.15, T <= 523.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pabalan and Pitzer (1987i) ~~~~~
# Note that there are two Pabalan & Pitzer (1987)'s: one compiling a suite of
#  electrolytes (PP87ii), and one just for NaOH (PP87i).
# There are also a bunch of Phutela & Pitzer papers in similar years, so take
#  care with reference codes!
def PP87i_eqNaOH(T, P, a):
    """PP87i equation for sodium hydroxide, with pressure in bar."""
    return a[ 0] \
         + a[ 1] * P \
         + a[ 2] / T \
         + a[ 3] * P / T \
         + a[ 4] * log(T) \
         + a[ 5] * T \
         + a[ 6] * T * P \
         + a[ 7] * T**2 \
         + a[ 8] * T**2 * P \
         + a[ 9] / (T - 227) \
         + a[10] / (647 - T) \
         + a[11] * P / (647 - T)

def bC_Na_OH_PP87i(T, P):
    """c-a: sodium hydroxide [PP87i]."""
    P_bar = P / 10 # Convert dbar to bar
    b0 = PP87i_eqNaOH(T, P_bar, [
         2.7682478e+2,
        -2.8131778e-3,
        -7.3755443e+3,
         3.7012540e-1,
        -4.9359970e+1,
         1.0945106e-1,
         7.1788733e-6,
        -4.0218506e-5,
        -5.8847404e-9,
         1.1931122e+1,
         2.4824963e00,
        -4.8217410e-3,
    ])
    b1 = PP87i_eqNaOH(T, P_bar, [
         4.6286977e+2,
         0,
        -1.0294181e+4,
         0,
        -8.5960581e+1,
         2.3905969e-1,
         0,
        -1.0795894e-4,
         0, 0, 0, 0,
    ])
    b2 = 0
    Cphi = PP87i_eqNaOH(T, P_bar, [
        -1.6686897e+01,
         4.0534778e-04,
         4.5364961e+02,
        -5.1714017e-02,
         2.9680772e000,
        -6.5161667e-03,
        -1.0553037e-06,
         2.3765786e-06,
         8.9893405e-10,
        -6.8923899e-01,
        -8.1156286e-02,
         0,
    ])
    zNa = +1
    zOH = -1
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zOH)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 298.15, T <= 523.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Mg_Cl_PP87i(T, P):
    """c-a: magnesium chloride [PP87i]."""
    b0, b1, b2, _, C1, alph1, alph2, omega, _ = bC_Mg_Cl_dLP83(T, P)
    Cphi = 2.41831e-7 * T**2 - 2.49949e-4 * T + 5.95320e-2
    zMg = +2
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(zMg * zCl)))
    valid = logical_and(T >= 298.15, T <= 473.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Simonson et al. (1987a) ~~~~~
def SRRJ87_eq7(T, a):
    """SRRJ87 equation 7."""
    Tr = 298.15
    return a[0] + a[1]*1e-3*(T - Tr) + a[2]*1e-5*(T - Tr)**2

def bC_K_Cl_SRRJ87(T, P):
    """c-a: potassium chloride [SRRJ87]."""
    # Parameters from SRRJ87 Table III
    b0 = SRRJ87_eq7(T, float_([
         0.0481,
         0.592,
        -0.562,
    ]))
    b1 = SRRJ87_eq7(T, float_([
         0.2188,
         1.500,
        -1.085,
    ]))
    b2 = 0
    Cphi = SRRJ87_eq7(T, float_([
        -0.790,
        -0.639,
         0.613,
    ]))
    zK = +1
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(zK * zCl)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 278.15, T <= 328.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_Cl_SRRJ87(T, P):
    """c-a: sodium chloride [SRRJ87]."""
    # Parameters from SRRJ87 Table III
    b0 = SRRJ87_eq7(T, float_([
         0.0754,
         0.792,
        -0.935,
    ]))
    b1 = SRRJ87_eq7(T, float_([
         0.2770,
         1.006,
        -0.756,
    ]))
    b2 = 0
    Cphi = SRRJ87_eq7(T, float_([
         1.40,
        -1.20,
         1.15,
    ]))
    zNa = +1
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zCl)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 278.15, T <= 328.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_BOH4_SRRJ87(T, P):
    """c-a: potassium borate [SRRJ87]."""
    # Parameters from SRRJ87 Table III
    b0 = SRRJ87_eq7(T, float_([
        0.1469,
        2.881,
        0,
    ]))
    b1 = SRRJ87_eq7(T, float_([
        -0.0989,
        -6.876,
         0,
    ]))
    b2 = 0
    Cphi = SRRJ87_eq7(T, float_([
        -56.43,
        -9.56,
         0,
    ]))
    zK = +1
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(zK * zCl)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 278.15, T <= 328.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_BOH4_SRRJ87(T, P):
    """c-a: sodium borate [SRRJ87]."""
    # Parameters from SRRJ87 Table III
    b0 = SRRJ87_eq7(T, float_([
        -0.0510,
         5.264,
         0,
    ]))
    b1 = SRRJ87_eq7(T, float_([
         0.0961,
        -10.68,
         0,
    ]))
    b2 = 0
    Cphi = SRRJ87_eq7(T, float_([
         14.98,
        -15.7,
         0,
    ]))
    zNa = +1
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zCl)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 278.15, T <= 328.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def theta_BOH4_Cl_SRRJ87(T, P):
    """a-a': borate chloride [SRRJ87]."""
    # Parameter from SRRJ87 Table III
    theta = -0.056
    valid = logical_and(T >= 278.15, T <= 328.15)
    return theta, valid

def psi_K_BOH4_Cl_SRRJ87(T, P):
    """c-a-a': potassium borate chloride [SRRJ87]."""
    psi = 0
    valid = logical_and(T >= 278.15, T <= 328.15)
    return psi, valid

def psi_Na_BOH4_Cl_SRRJ87(T, P):
    """c-a-a': sodium borate chloride [SRRJ87]."""
    # Parameter from SRRJ87 Table III
    psi = -0.019
    valid = logical_and(T >= 278.15, T <= 328.15)
    return psi, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Simonson et al. (1987b) ~~~~~
def SRM87_eqTableIII(T, abc):
    """SRM87 equation from Table III."""
    return abc[0] \
         + abc[1] * 1e-3 * (T - 298.15) \
         + abc[2] * 1e-3 * (T -303.15)**2

def bC_Mg_BOH4_SRM87(T, P):
    """c-a: magnesium borate [SRM87]."""
    b0 = SRM87_eqTableIII(T, float_([
        -0.6230,
         6.496,
         0,
    ]))
    b1 = SRM87_eqTableIII(T, float_([
         0.2515,
        -17.13,
         0,
    ]))
    b2 = SRM87_eqTableIII(T, float_([
        -11.47,
         0,
        -3.240]))
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 278.15, T <= 528.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Ca_BOH4_SRM87(T, P):
    """c-a: calcium borate [SRM87]."""
    b0 = SRM87_eqTableIII(T, float_([
        -0.4462,
         5.393,
         0,
    ]))
    b1 = SRM87_eqTableIII(T, float_([
        -0.8680,
        -18.20,
         0,
    ]))
    b2 = SRM87_eqTableIII(T, float_([
        -15.88,
         0,
        -2.858,
    ]))
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 278.15, T <= 528.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Hershey et al. (1988) ~~~~~
def bC_Na_HS_HPM88(T, P):
    """c-a: sodium bisulfide [HPM88]."""
    b0 = 3.66e-1 - 6.75e+1 / T
    b1 = 0
    b2 = 0
    Cphi = -1.27e-2
    zNa = +1
    zHS = -1
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zHS)))
    C1 = 0
    alph1 = -9
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 278.15,T <= 318.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_HS_HPM88(T, P):
    """c-a: potassium bisulfide [HPM88]."""
    b0 = 6.37e-1 - 1.40e+2 / T
    b1 = 0
    b2 = 0
    Cphi = -1.94e-1
    zK = +1
    zHS = -1
    C0 = Cphi / (2 * sqrt(np_abs(zK * zHS)))
    C1 = 0
    alph1 = -9
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 278.15,T <= 298.15)
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
    valid = T == 298.15
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
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Møller (1988) ~~~~~
def M88_eq13(T, a):
    """M88 equation 13."""
    return a[0] + a[1]*T + a[2]/T + a[3]*log(T) + a[4]/(T - 263) \
        + a[5]*T**2 + a[6]/(680 - T) + a[7]/(T - 227)

def b0_Ca_Cl_M88(T, P):
    """beta0: calcium chloride [M88]."""
    return M88_eq13(T, float_([
        -9.41895832e+1,
        -4.04750026e-2,
         2.34550368e+3,
         1.70912300e+1,
        -9.22885841e-1,
         1.51488122e-5,
        -1.39082000e00,
         0,
    ]))

def b1_Ca_Cl_M88(T, P):
    """beta1: calcium chloride [M88]."""
    return M88_eq13(T, float_([
         3.47870000e00,
        -1.54170000e-2,
         0, 0, 0,
         3.17910000e-5,
         0, 0,
    ]))

def Cphi_Ca_Cl_M88(T, P):
    """Cphi: calcium chloride [M88]."""
    return M88_eq13(T, float_([
        -3.03578731e+1,
        -1.36264728e-2,
         7.64582238e+2,
         5.50458061e00,
        -3.27377782e-1,
         5.69405869e-6,
        -5.36231106e-1,
         0,
    ]))

def bC_Ca_Cl_M88(T, P):
    """c-a: calcium chloride [M88]."""
    b0 = b0_Ca_Cl_M88(T, P)
    b1 = b1_Ca_Cl_M88(T, P)
    b2 = 0
    Cphi = Cphi_Ca_Cl_M88(T, P)
    zCa = +2
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(zCa * zCl)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 298.15, T <= 523.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Ca_SO4_M88(T, P):
    """c-a: calcium sulfate [M88]."""
    b0 = 0.15
    b1 = 3.00
    b2 = M88_eq13(T, float_([
        -1.29399287e+2,
         4.00431027e-1,
         0, 0, 0, 0, 0, 0,
    ]))
    C0 = 0
    C1 = 0
    alph1 = 1.4
    alph2 = 12
    omega = -9
    valid = logical_and(T >= 298.15, T <= 523.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_Cl_M88(T, P):
    """c-a: sodium chloride [M88]."""
    b0 = M88_eq13(T, float_([
         1.43783204e+1,
         5.60767406e-3,
        -4.22185236e+2,
        -2.51226677e00,
         0,
        -2.61718135e-6,
         4.43854508e00,
        -1.70502337e00,
    ]))
    b1 = M88_eq13(T, float_([
        -4.83060685e-1,
         1.40677479e-3,
         1.19311989e+2,
         0, 0, 0, 0,
        -4.23433299e00,
    ]))
    b2 = 0
    Cphi = M88_eq13(T, float_([
        -1.00588714e-1,
        -1.80529413e-5,
         8.61185543e00,
         1.24880954e-2,
         0,
         3.41172108e-8,
         6.83040995e-2,
         2.93922611e-1,
    ]))
    zNa = +1
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zCl)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 573.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_SO4_M88(T, P):
    """c-a: sodium sulfate [M88]."""
    b0 = M88_eq13(T, float_([
         8.16920027e+1,
         3.01104957e-2,
        -2.32193726e+3,
        -1.43780207e+1,
        -6.66496111e-1,
        -1.03923656e-5,
         0, 0,
    ]))
    b1 = M88_eq13(T, float_([
         1.00463018e+3,
         5.77453682e-1,
        -2.18434467e+4,
        -1.89110656e+2,
        -2.03550548e-1,
        -3.23949532e-4,
         1.46772243e+3,
         0,
    ]))
    b2 = 0
    Cphi = M88_eq13(T, float_([
        -8.07816886e+1,
        -3.54521126e-2,
         2.02438830e+3,
         1.46197730e+1,
        -9.16974740e-2,
         1.43946005e-5,
        -2.42272049e00,
         0,
    ]))
    zNa = +1
    zSO4 = -2
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zSO4)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 573.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def theta_Ca_Na_M88(T, P):
    """c-c': calcium sodium [M88]."""
    theta = 0.05
    valid = logical_and(T >= 298.15, T <= 523.15)
    return theta, valid

def theta_Cl_SO4_M88(T, P):
    """a-a': chloride sulfate [M88]."""
    theta = 0.07
    valid = logical_and(T >= 298.15, T <= 423.15)
    return theta, valid

def psi_Ca_Na_Cl_M88(T, P):
    """c-c'-a: calcium sodium chloride [M88]."""
    psi = -0.003
    valid = logical_and(T >= 298.15, T <= 523.15)
    return psi, valid

def psi_Ca_Na_SO4_M88(T, P):
    """c-c'-a: calcium sodium sulfate [M88]."""
    psi = -0.012
    valid = logical_and(T >= 298.15, T <= 523.15)
    return psi, valid

def psi_Ca_Cl_SO4_M88(T, P):
    """c-a-a': calcium chloride sulfate [M88]."""
    psi = -0.018
    valid = logical_and(T >= 298.15, T <= 523.15)
    return psi, valid

def psi_Na_Cl_SO4_M88(T, P):
    """c-a-a': sodium chloride sulfate [M88]."""
    psi = -0.009
    valid = logical_and(T >= 298.15, T <= 423.15)
    return psi, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Greenberg and Møller (1989) ~~~~~
def GM89_eq3(T, a):
    """GM89 equation 3."""
    return M88_eq13(T, a)

def Cphi_Ca_Cl_GM89(T, P):
    """Cphi: calcium chloride [GM89]."""
    return GM89_eq3(T, float_([
         1.93056024e+1,
         9.77090932e-3,
        -4.28383748e+2,
        -3.57996343e00,
         8.82068538e-2,
        -4.62270238e-6,
         9.91113465e00,
         0,
    ]))

def bC_Ca_Cl_GM89(T, P):
    """c-a: calcium chloride [GM89]."""
    b0, b1, b2, _, C1, alph1, alph2, omega, valid = bC_Ca_Cl_M88(T, P)
    Cphi = Cphi_Ca_Cl_GM89(T, P)
    zCa = +2
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(zCa * zCl)))
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_Cl_GM89(T, P):
    """c-a: potassium chloride [GM89]."""
    b0 = GM89_eq3(T, float_([
         2.67375563e+1,
         1.00721050e-2,
        -7.58485453e+2,
        -4.70624175e00,
         0,
        -3.75994338e-6,
         0, 0,
    ]))
    b1 = GM89_eq3(T, float_([
        -7.41559626e00,
         0,
         3.22892989e+2,
         1.16438557e00,
         0, 0, 0,
        -5.94578140e00,
    ]))
    b2 = 0
    Cphi = GM89_eq3(T, float_([
        -3.30531334e00,
        -1.29807848e-3,
         9.12712100e+1,
         5.86450181e-1,
         0,
         4.95713573e-7,
         0, 0,
    ]))
    zK = +1
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(zK * zCl)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 523.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_SO4_GM89(T, P):
    """c-a: potassium sulfate [GM89]."""
    b0 = GM89_eq3(T, float_([
         4.07908797e+1,
         8.26906675e-3,
        -1.41842998e+3,
        -6.74728848e00,
         0, 0, 0, 0,
    ]))
    b1 = GM89_eq3(T, float_([
        -1.31669651e+1,
         2.35793239e-2,
         2.06712594e+3,
         0, 0, 0, 0, 0,
    ]))
    b2 = 0
    Cphi = -0.0188
    zK = +1
    zSO4 = -2
    C0 = Cphi / (2 * sqrt(np_abs(zK * zSO4)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 523.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def theta_Ca_K_GM89(T, P):
    """c-c': calcium potassium [GM89]."""
    theta = 0.1156
    valid = logical_and(T >= 273.15, T <= 523.15)
    return theta, valid

def theta_K_Na_GM89(T, P):
    """c-c': potassium sodium [GM89]."""
    theta = GM89_eq3(T, float_([
        -5.02312111e-2,
         0,
         1.40213141e+1,
         0, 0, 0, 0, 0,
    ]))
    valid = logical_and(T >= 273.15, T <= 523.15)
    return theta, valid

def psi_Ca_K_Cl_GM89(T, P):
    """c-c'-a: calcium potassium chloride [GM89]."""
    psi = GM89_eq3(T, float_([
         4.76278977e-2,
         0,
        -2.70770507e+1,
         0, 0, 0, 0, 0,
    ]))
    valid = logical_and(T >= 273.15, T <= 523.15)
    return psi, valid

def psi_Ca_K_SO4_GM89(T, P):
    """c-c'-a: calcium potassium sulfate [GM89]."""
    psi = 0
    valid = logical_and(T >= 273.15, T <= 523.15)
    return psi, valid

def psi_K_Na_Cl_GM89(T, P):
    """c-c'-a: potassium sodium chloride [GM89]."""
    psi = GM89_eq3(T, float_([
         1.34211308e-2,
         0,
        -5.10212917e00,
         0, 0, 0, 0, 0,
    ]))
    valid = logical_and(T >= 273.15, T <= 523.15)
    return psi, valid

def psi_K_Na_SO4_GM89(T, P):
    """c-c'-a: potassium sodium sulfate [GM89]."""
    psi = GM89_eq3(T, float_([
         3.48115174e-2,
         0,
        -8.21656777e00,
         0, 0, 0, 0, 0,
    ]))
    valid = logical_and(T >= 273.15, T <= 423.15)
    return psi, valid

def psi_K_Cl_SO4_GM89(T, P):
    """c-a-a': potassium chloride sulfate [GM89]."""
    psi = GM89_eq3(T, float_([
        -2.12481475e-1,
         2.84698333e-4,
         3.75619614e+1,
         0, 0, 0, 0, 0,
    ]))
    valid = logical_and(T >= 273.15, T <= 523.15)
    return psi, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Millero et al. (1989) ~~~~~
def bC_Na_SO3_MHJZ89(T, P):
    """c-a: sodium sulfite [MHJZ89]."""
    b0 = 5.88444 - 1730.55/T # Eq. (36)
    b1 = -19.4549 + 6153.78/T # Eq. (37)
    b2 = 0
    Cphi = -1.2355 + 367.07/T # Eq. (38)
    zNa = +1
    zSO3 = -2
    C0 = Cphi/(2*sqrt(np_abs(zNa*zSO3)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_HSO3_MHJZ89(T, P):
    """c-a: sodium hydrogen-sulfite [MHJZ89]."""
    b0 = 4.3407 - 1248.66/T # Eq. (29)
    b1 = -13.146 + 4014.80/T # Eq. (30)
    b2 = 0
    Cphi = 0.9565 + 277.85/T # Eq. (31), note difference from MP98 Table A3
    zNa = +1
    zHSO3 = -1
    C0 = Cphi/(2*sqrt(np_abs(zNa*zHSO3)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def theta_Cl_SO3_MHJZ89(T, P):
    """a-a': chloride sulfite [MHJZ89]."""
    theta = 0.099 # +/- 0.004
    valid = logical_and(T >= 273.15, T <= 323.15)
    return theta, valid

def psi_Na_Cl_SO3_MHJZ89(T, P):
    """c-a-a': sodium chloride sulfite [MHJZ89]."""
    psi = -0.0156 # +/- 0.001
    valid = logical_and(T >= 273.15, T <= 323.15)
    return psi, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Archer (1992) ~~~~~
def A92ii_eq36(T, P, a):
    """A92ii equation 36, with pressure in MPa."""
    # a[5] and a[6] multipliers are corrected for typos in A92ii
    return (a[ 0]
          + a[ 1] * 10**-3*T
          + a[ 2] * 4e-6*T**2
          + a[ 3] * 1 / (T - 200)
          + a[ 4] * 1 / T
          + a[ 5] * 100 / (T - 200)**2
          + a[ 6] * 200 / T**2
          + a[ 7] * 8e-9*T**3
          + a[ 8] * 1 / (650 - T)**0.5
          + a[ 9] * 10**-5*P
          + a[10] * 2e-4*P / (T - 225)
          + a[11] * 100*P / (650 - T)**3
          + a[12] * 2e-8*P*T
          + a[13] * 2e-4*P / (650 - T)
          + a[14] * 10**-7*P**2
          + a[15] * 2e-6*P**2 / (T - 225)
          + a[16] * P**2 / (650 - T)**3
          + a[17] * 2e-10*P**2*T
          + a[18] * 4e-13*P**2*T**2
          + a[19] * 0.04*P / (T - 225)**2
          + a[20] * 4e-11*P*T**2
          + a[21] * 2e-8*P**3 / (T - 225)
          + a[22] * 0.01*P**3 / (650 - T)**3
          + a[23] * 200 / (650 - T)**3)

def bC_Na_Cl_A92ii(T, P):
    """c-a: sodium chloride [A92ii]."""
    P_MPa = P / 100 # Convert dbar to MPa
    # Parameters from A92ii Table 2, with noted corrections
    b0 = A92ii_eq36(T, P_MPa, [
          0.242408292826506,
          0,
         -0.162683350691532,
          1.38092472558595,
          0, 0,
        -67.2829389568145,
          0,
          0.625057580755179,
        -21.2229227815693,
         81.8424235648693,
         -1.59406444547912,
          0, 0,
         28.6950512789644,
        -44.3370250373270,
          1.92540008303069,
        -32.7614200872551,
          0, 0,
         30.9810098813807,
          2.46955572958185,
         -0.725462987197141,
         10.1525038212526,
    ])
    b1 = A92ii_eq36(T, P_MPa, [
        - 1.90196616618343,
          5.45706235080812,
          0,
        -40.5376417191367,
          0, 0,
          4.85065273169753  * 1e2,
        -0.661657744698137,
          0, 0,
          2.42206192927009  * 1e2,
          0,
        -99.0388993875343,
          0, 0,
        -59.5815563506284,
          0, 0, 0, 0, 0, 0, 0, 0,
    ])
    b2 = 0
    C0 = A92ii_eq36(T, P_MPa, [
          0,
         -0.0412678780636594,
          0.0193288071168756,
         -0.338020294958017, # typo in A92ii
          0,
          0.0426735015911910,
          4.14522615601883,
         -0.00296587329276653,
          0,
          1.39697497853107,
         -3.80140519885645,
          0.06622025084, # typo in A92ii - "Rard's letter"
          0,
        -16.8888941636379,
         -2.49300473562086,
          3.14339757137651,
          0,
          2.79586652877114,
          0, 0, 0, 0, 0,
         -0.502708980699711,
    ])
    C1 = A92ii_eq36(T, P_MPa, [ \
          0.788987974218570,
         -3.67121085194744,
          1.12604294979204,
          0, 0,
        -10.1089172644722,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         16.6503495528290,
    ])
    alph1 = 2
    alph2 = -9
    omega = 2.5
    valid = logical_and(T >= 250, T <= 600)
    valid = logical_and(valid, P_MPa <= 100)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Campbell et al. (1993) ~~~~~
def CMR93_eq31(T, a):
    """CMR93 equation 31."""
    return M88_eq13(T, a)

def bC_H_Cl_CMR93(T, P):
    """c-a: hydrogen chloride [CMR93]."""
    # b0 a[1] term corrected here for typo, following WM13
    b0 = CMR93_eq31(T, float_([
         1.2859,
        -2.1197e-3,
        -142.5877,
         0, 0, 0, 0, 0,
    ]))
    b1 = CMR93_eq31(T, float_([
        -4.4474,
         8.425698e-3,
         665.7882,
         0, 0, 0, 0, 0,
    ]))
    b2 = 0
    Cphi = CMR93_eq31(T, float_([
        -0.305156,
         5.16e-4,
         45.52154,
         0, 0, 0, 0, 0,
    ]))
    zH = +1
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(zH * zCl)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 328.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def theta_H_K_CMR93(T, P):
    """c-c': hydrogen potassium [CMR93]."""
    # assuming CMR93's lowercase t means temperature in degC
    theta = 0.005 -0.0002275 * (T - Tzero)
    valid = logical_and(T >= 273.15, T <= 328.15)
    return theta, valid

def theta_H_Na_CMR93(T, P):
    """c-c': hydrogen sodium [CMR93]."""
    # assuming CMR93's lowercase t means temperature in degC
    theta = 0.0342 -0.000209 * (T - Tzero)
    valid = logical_and(T >= 273.15, T <= 328.15)
    return theta, valid

def psi_H_K_Cl_CMR93(T, P):
    """c-c'-a: hydrogen potassium chloride [CMR93]."""
    psi = 0
    valid = logical_and(T >= 273.15, T <= 523.15)
    return psi, valid

def psi_H_Na_Cl_CMR93(T, P):
    """c-c'-a: hydrogen sodium chloride [CMR93]."""
    psi = 0
    valid = logical_and(T >= 273.15, T <= 523.15)
    return psi, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Hovey et al. (1993) ~~~~~
def HPR93_eq36(T, a):
    """HPR93 equation 36."""
    Tref = 298.15
    return a[0] + a[1]*(1/T - 1/Tref) + a[2]*log(T/Tref)

def bC_Na_SO4_HPR93(T, P):
    """c-a: sodium sulfate [HPR93]."""
    b0 = HPR93_eq36(T, float_([
         0.006536438,
        -30.197349,
        -0.20084955,
    ]))
    b1 = HPR93_eq36(T, float_([
         0.87426420,
        -70.014123,
         0.2962095,
    ]))
    b2 = 0
    Cphi = HPR93_eq36(T, float_([
        0.007693706,
        4.5879201,
        0.019471746,
    ]))
    zNa = +1
    zSO4 = -2
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zSO4)))
    C1 = 0
    alph1 = 1.7
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273., T <= 373.)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_HSO4_HPR93(T, P):
    """c-a: sodium bisulfate, low ionic strengths [HPR93]."""
    # Parameters from HPR93 Table 3 for low ionic strengths
    b0 = 0.0670967
    b1 = 0.3826401
    b2 = 0
    Cphi = -0.0039056
    zNa = +1
    zHSO4 = -1
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zHSO4)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Clegg et al. (1994) ~~~~~
CRP94_Tr = 328.15 # K

def CRP94_eq24(T,q):
    return q[0] + 1e-3 * \
        ( (T-CRP94_Tr)    * q[1] \
        + (T-CRP94_Tr)**2 * q[2] / 2 \
        + (T-CRP94_Tr)**3 * q[3] / 6)

def bC_H_HSO4_CRP94(T, P):
    """c-a: hydrogen bisulfate [CRP94]."""
    # Parameters from CRP94 Table 6
    b0 = CRP94_eq24(T, float_([
         0.227784933,
        -3.78667718,
        -0.124645729,
        -0.00235747806,
    ]))
    b1 = CRP94_eq24(T, float_([
        0.372293409,
        1.50,
        0.207494846,
        0.00448526492,
    ]))
    b2 = 0
    C0 = CRP94_eq24(T, float_([
        -0.00280032520,
         0.216200279,
         0.0101500824,
         0.000208682230,
    ]))
    C1 = CRP94_eq24(T, float_([
        -0.025,
         18.1728946,
         0.382383535,
         0.0025,
    ]))
    alph1 = 2
    alph2 = -9
    omega = 2.5
    valid = logical_and(T >= 273.15, T <= 328.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_H_SO4_CRP94(T, P):
    """c-a: hydrogen sulfate [CRP94]."""
    # Evaluate parameters from CRP94 Table 6
    b0 = CRP94_eq24(T, float_([
        0.0348925351,
        4.97207803,
        0.317555182,
        0.00822580341,
    ]))
    b1 = CRP94_eq24(T, float_([
        -1.06641231,
        -74.6840429,
        -2.26268944,
        -0.0352968547,
    ]))
    b2 = 0
    C0 = CRP94_eq24(T, float_([
         0.00764778951,
        -0.314698817,
        -0.0211926525,
        -0.000586708222,
    ]))
    C1 = CRP94_eq24(T, float_([
         0,
        -0.176776695,
        -0.731035345,
         0,
    ]))
    alph1 = 2 - 1842.843 * (1/T - 1/298.15)
    alph2 = -9
    omega = 2.5
    valid = logical_and(T >= 273.15, T <= 328.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def theta_HSO4_SO4_CRP94(T, P):
    """a-a': bisulfate sulfate [CRP94]."""
    theta = 0
    valid = logical_and(T >= 273.15, T <= 328.15)
    return theta, valid

def psi_H_HSO4_SO4_CRP94(T, P):
    """c-a-a': hydrogen bisulfate sulfate [CRP94]."""
    psi = 0
    valid = logical_and(T >= 273.15, T <= 328.15)
    return psi, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Millero and Pierrot (1998) ~~~~~
def MP98_eq15(T,q):
    # q[0] = PR
    # q[1] = PJ  * 1e5
    # q[2] = PRL * 1e4
    Tr = 298.15
    return (q[0] + q[1]*1e-5 * (Tr**3/3 - Tr**2 * q[2]*1e-4)*(1/T - 1/Tr)
        + q[1]*1e-5*(T**2 - Tr**2)/6)

def bC_Na_I_MP98(T, P):
    """c-a: sodium iodide [MP98]."""
    b0 = MP98_eq15(T, float_([
         0.1195,
        -1.01,
         8.355,
    ]))
    b1 = MP98_eq15(T, float_([
         0.3439,
        -2.54,
         8.28,
    ]))
    b2 = 0
    Cphi = MP98_eq15(T, float_([
         0.0018,
         0,
        -0.835,
    ]))
    zNa = +1
    zI = -1
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zI)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_Br_MP98(T, P):
    """c-a: sodium bromide [MP98]."""
    b0 = MP98_eq15(T, float_([
         0.0973,
        -1.3,
         7.692,
    ]))
    b1 = MP98_eq15(T, float_([
         0.2791,
        -1.06,
         10.79,
    ]))
    b2 = 0
    Cphi = MP98_eq15(T, float_([
         0.00116,
         0.16405,
        -0.93,
    ]))
    zNa = +1
    zBr = -1
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zBr)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_F_MP98(T, P):
    """c-a: sodium fluoride [MP98]."""
    b0 = MP98_eq15(T, float_([
         0.215,
        -2.37,
         5.361e-4,
    ]))
    b1 = MP98_eq15(T, float_([
        0.2107,
        0,
        8.7e-4,
    ]))
    b2 = 0
    C0 = 0
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_Br_MP98(T, P):
    """c-a: potassium bromide [MP98]."""
    b0 = MP98_eq15(T, float_([
         0.0569,
        -1.43,
         7.39,
    ]))
    b1 = MP98_eq15(T, float_([
         0.2122,
        -0.762,
         1.74,
    ]))
    b2 = 0
    Cphi = MP98_eq15(T, float_([
        -0.0018,
         0.216,
        -0.7004,
    ]))
    zK = +1
    zBr = -1
    C0 = Cphi / (2 * sqrt(np_abs(zK * zBr)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_F_MP98(T, P):
    """c-a: potassium fluoride [MP98]."""
    b0 = MP98_eq15(T, float_([
         0.08089,
        -1.39,
         2.14,
    ]))
    b1 = MP98_eq15(T, float_([
        0.2021,
        0,
        5.44,
    ]))
    b2 = 0
    Cphi = MP98_eq15(T, float_([
        0.00093,
        0,
        0.595,
    ]))
    zK = +1
    zF = -1
    C0 = Cphi / (2 * sqrt(np_abs(zK * zF)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_OH_MP98(T, P):
    """c-a: potassium hydroxide [MP98]."""
    b0 = MP98_eq15(T, float_([
         0.1298,
        -0.946,
         9.914,
    ])) # copy of KI
    b1 = MP98_eq15(T, float_([
         0.32,
        -2.59,
         11.86,
    ])) # copy of KI
    b2 = 0
    Cphi = MP98_eq15(T, float_([
        -0.0041,
         0.0638,
        -0.944,
    ])) # copy of KI
    zK = +1
    zOH = -1
    C0 = Cphi / (2 * sqrt(np_abs(zK * zOH)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_I_MP98(T, P):
    """c-a: potassium iodide [MP98]."""
    b0 = MP98_eq15(T, float_([
         0.0746,
        -0.748,
         9.914,
    ]))
    b1 = MP98_eq15(T, float_([
         0.2517,
        -1.8,
         11.86,
    ]))
    b2 = 0
    Cphi = MP98_eq15(T, float_([
        -0.00414,
         0,
        -0.944,
    ]))
    zK = +1
    zI = -1
    C0 = Cphi / (2 * sqrt(np_abs(zK * zI)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_ClO3_MP98(T, P):
    """c-a: sodium chlorate [MP98]."""
    b0 = MP98_eq15(T, float_([
         0.0249,
        -1.56,
         10.35,
    ]))
    b1 = MP98_eq15(T, float_([
         0.2455,
        -2.69,
         19.07,
    ]))
    b2 = 0
    Cphi = MP98_eq15(T, float_([
         0.0004,
         0.222,
         9.29,
    ]))
    zNa = +1
    zClO3 = -1
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zClO3)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_ClO3_MP98(T, P):
    """c-a: potassium chlorate [MP98]."""
    b0 = MP98_eq15(T, float_([
        -0.096,
         15.1,
         19.87,
    ]))
    b1 = MP98_eq15(T, float_([
         0.2841,
        -27,
         31.8,
    ]))
    b2 = 0
    Cphi = MP98_eq15(T, float_([
         0,
        -19.1,
         0,
    ]))
    zK = +1
    zClO3 = -1
    C0 = Cphi / (2 * sqrt(np_abs(zK * zClO3)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_ClO4_MP98(T, P):
    """c-a: sodium perchlorate [MP98]."""
    b0 = MP98_eq15(T, float_([
         0.0554,
        -0.611,
         12.96,
    ]))
    b1 = MP98_eq15(T, float_([
         0.2755,
        -6.35,
         22.97,
    ]))
    b2 = 0
    Cphi = MP98_eq15(T, float_([
        -0.00118,
         0.0562,
        -1.623,
    ]))
    zNa = +1
    zClO4 = -1
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zClO4)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_BrO3_MP98(T, P):
    """c-a: sodium bromate [MP98]."""
    b0 = MP98_eq15(T, float_([
        -0.0205,
        -6.5,
         5.59,
    ]))
    b1 = MP98_eq15(T, float_([
        0.191,
        5.45,
        34.37,
    ]))
    b2 = 0
    Cphi = MP98_eq15(T, float_([
        0.0059,
        2.5,
        0,
    ]))
    zNa = +1
    zBrO3 = -1
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zBrO3)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_BrO3_MP98(T, P):
    """c-a: potassium bromate [MP98]."""
    b0 = MP98_eq15(T, float_([
        -0.129,
         9.17,
         5.59,
    ]))
    b1 = MP98_eq15(T, float_([
         0.2565,
        -20.2,
         34.37,
    ]))
    b2 = 0
    Cphi = MP98_eq15(T, float_([
         0,
        -26.6,
         0,
    ]))
    zK = +1
    zBrO3 = -1
    C0 = Cphi / (2 * sqrt(np_abs(zK * zBrO3)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_NO3_MP98(T, P):
    """c-a: sodium nitrate [MP98]."""
    b0 = MP98_eq15(T, float_([
         0.0068,
        -2.24,
         12.66,
    ]))
    b1 = MP98_eq15(T, float_([
         0.1783,
        -2.96,
         20.6,
    ]))
    b2 = 0
    Cphi = MP98_eq15(T, float_([
        -0.00072,
         0.594,
        -2.316,
    ]))
    zNa = +1
    zNO3 = -1
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zNO3)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_NO3_MP98(T, P):
    """c-a: potassium nitrate [MP98]."""
    b0 = MP98_eq15(T, float_([
        -0.0816,
        -0.785,
         2.06,
    ]))
    b1 = MP98_eq15(T, float_([
         0.0494,
        -8.26,
         64.5,
    ]))
    b2 = 0
    Cphi = MP98_eq15(T, float_([
        0.0066,
        0,
        3.97,
    ]))
    zK = +1
    zNO3 = -1
    C0 = Cphi / (2 * sqrt(np_abs(zK * zNO3)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Mg_NO3_MP98(T, P):
    """c-a: magnesium nitrate [MP98]."""
    b0 = MP98_eq15(T, float_([
         0.367125,
        -1.2322,
         5.15,
    ]))
    b1 = MP98_eq15(T, float_([
        1.58475,
        4.0492,
        44.925,
    ]))
    b2 = 0
    Cphi = MP98_eq15(T, float_([
        -0.020625,
         0, 0,
    ]))
    zMg = +2
    zNO3 = -1
    C0 = Cphi / (2 * sqrt(np_abs(zMg * zNO3)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Ca_NO3_MP98(T, P):
    """c-a: calcium nitrate [MP98]."""
    b0 = MP98_eq15(T, float_([
        0.210825,
        4.0248,
        5.295,
    ]))
    b1 = MP98_eq15(T, float_([
         1.40925,
        -13.289,
         91.875,
    ]))
    b2 = 0
    Cphi = MP98_eq15(T, float_([
        -0.020142,
        -15.435,
         0,
    ]))
    zCa = +2
    zNO3 = -1
    C0 = Cphi / (2 * sqrt(np_abs(zCa * zNO3)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_H_Br_MP98(T, P):
    """c-a: hydrogen bromide [MP98]."""
    b0 = MP98_eq15(T, float_([
         0.196,
        -0.357,
        -2.049,
    ]))
    b1 = MP98_eq15(T, float_([
         0.3564,
        -0.913,
         4.467,
    ]))
    b2 = 0
    Cphi = MP98_eq15(T, float_([
         0.00827,
         0.01272,
        -0.5685,
    ]))
    zH = +1
    zBr = -1
    C0 = Cphi / (2 * sqrt(np_abs(zH * zBr)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Sr_Cl_MP98(T, P):
    """c-a: strontium chloride [MP98]."""
    b0 = MP98_eq15(T, float_([
         0.28575,
        -0.18367,
         7.1,
    ]))
    b1 = MP98_eq15(T, float_([
        1.66725,
        0,
        28.425,
    ]))
    b2 = 0
    Cphi = MP98_eq15(T, float_([
        -0.0013,
         0, 0,
    ]))
    zSr = +2
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(zSr * zCl)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_NH4_Cl_MP98(T, P):
    """c-a: ammonium chloride [MP98]."""
    b0 = MP98_eq15(T, float_([
         0.0522,
        -0.597,
         0.779,
    ]))
    b1 = MP98_eq15(T, float_([
        0.1918,
        0.444,
        12.58,
    ]))
    b2 = 0
    Cphi = MP98_eq15(T, float_([
        -0.00301,
         0.0578,
         0.21,
    ]))
    zNH4 = +1
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(zNH4 * zCl)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_NH4_Br_MP98(T, P):
    """c-a: ammonium bromide [MP98]."""
    b0 = MP98_eq15(T, float_([
         0.0624,
        -0.597,
         0.779,
    ]))
    b1 = MP98_eq15(T, float_([
        0.1947,
        0,
        12.58,
    ]))
    b2 = 0
    Cphi = MP98_eq15(T, float_([
        -0.00436,
         0,
         0.21,
    ]))
    zNH4 = +1
    zBr = -1
    C0 = Cphi / (2 * sqrt(np_abs(zNH4 * zBr)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_NH4_F_MP98(T, P):
    """c-a: ammonium fluoride [MP98]."""
    b0 = MP98_eq15(T, float_([
        0.1306,
        1.09,
        0.95,
    ]))
    b1 = MP98_eq15(T, float_([
        0.257,
        0,
        5.97,
    ]))
    b2 = 0
    Cphi = MP98_eq15(T, float_([
        -0.0043,
         0, 0,
    ]))
    zNH4 = +1
    zF = -1
    C0 = Cphi / (2 * sqrt(np_abs(zNH4 * zF)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def MP98_eqTableA3(T,abc):
    """MP98 equation for Table A3."""
    Tr = 298.15
    return abc[0] + abc[1] * (T - Tr) + abc[2] * (T - Tr)**2

def bC_Na_HSO4_MP98(T, P):
    """c-a: sodium bisulfate [MP98]."""
    # MP98 cite Pierrot et al. (1997) J Solution Chem 26(1),
    #  but their equations look quite different, and there is no Cphi there.
    # This equation is therefore directly from MP98.
    b0 = MP98_eqTableA3(T, float_([
         0.544,
        -1.8478e-3,
         5.3937e-5,
    ]))
    b1 = MP98_eqTableA3(T, float_([
         0.3826401,
        -1.8431e-2,
         0,
    ]))
    b2 = 0
    Cphi = 0.003905
    zNa = +1
    zHSO4 = -1
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zHSO4)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 273.15, T <= 323.15)
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Archer (1999) ~~~~~
def A99_eq22(T, a):
    """A99 equation 22."""
    Tref  = 298.15
    return   a[0] \
           + a[1] * (T - Tref)    * 1e-2 \
           + a[2] * (T - Tref)**2 * 1e-5 \
           + a[3] * 1e2 / (T - 225) \
           + a[4] * 1e3 /  T \
           + a[5] * 1e6 / (T - 225)**3

def bC_K_Cl_A99(T, P):
    """c-a: potassium chloride [A99]."""
    # KCl T parameters from A99 Table 4
    b0 = A99_eq22(T, float_([
         0.413229483398493,
        -0.0870121476114027,
         0.101413736179231,
        -0.0199822538522801,
        -0.0998120581680816,
         0,
    ]))
    b1 = A99_eq22(T, float_([
         0.206691413598171,
         0.102544606022162,
         0,
         0,
         0,
        -0.00188349608000903,
    ]))
    b2 = 0
    C0 = A99_eq22(T, float_([
        -0.00133515934994478,
         0,
         0,
         0.00234117693834228,
        -0.00075896583546707,
         0,
    ]))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = logical_and(T >= 260, T <= 420)
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Rard and Clegg (1999) ~~~~~
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
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def psi_H_Mg_HSO4_RC99(T, P):
    """c-c'-a: hydrogen magnesium bisulfate [RC99]."""
    # RC99 Table 6, left column
    psi = -0.027079
    valid = T == 298.15
    return psi, valid

def psi_H_Mg_SO4_RC99(T, P):
    """c-c'-a: hydrogen magnesium sulfate [RC99]."""
    # RC99 Table 6, left column
    psi = -0.047368
    valid = T == 298.15
    return psi, valid

def psi_Mg_HSO4_SO4_RC99(T, P):
    """c-a-a': magnesium bisulfate sulfate [RC99]."""
    # RC99 Table 6, left column
    psi = -0.078418
    valid = T == 298.15
    return psi, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Waters and Millero (2013) ~~~~~
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
    b1 = b1 + (T - TR) * P91_Ch3_T13_II['Ca-SO4']['b1']
    b2 = b2 + (T - TR) * P91_Ch3_T13_II['Ca-SO4']['b2']
    # The C0 temperature correction in P91 is zero
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Ca_HSO4_WM13(T, P):
    """c-a: calcium bisulfate [WM13]."""
    TR = 298.15
    b0, b1, b2, C0, C1, alph1, alph2, omega, valid = bC_Ca_HSO4_HMW84(T, P)
    # WM13 use temperature derivatives for Ca-ClO4 from P91, but with typos
    b0 = b0 + (T - TR) * P91_Ch3_T13_I['Ca-ClO4']['b0']
    b1 = b1 + (T - TR) * P91_Ch3_T13_I['Ca-ClO4']['b1']
    C0 = C0 + (T - TR) * P91_Ch3_T13_I['Ca-ClO4']['C0']
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_HSO4_WM13(T, P):
    """c-a: potassium bisulfate [WM13]."""
    TR = 298.15
    b0, b1, b2, C0, C1, alph1, alph2, omega, valid = bC_K_HSO4_HMW84(T, P)
    # WM13 use temperature derivatives for K-ClO4 from P91
    b0 = b0 + (T - TR) * P91_Ch3_T12['K-ClO4']['b0']
    b1 = b1 + (T - TR) * P91_Ch3_T12['K-ClO4']['b1']
    # The Cphi temperature correction in P91 is zero
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_HSO4_HPR93viaWM13(T, P):
    """c-a: sodium sulfate [HPR93 via WM13]."""
    # WM13 Table A1 - can't find where HPR93 state this
    return bC_none(T, P)

def theta_HSO4_SO4_WM13(T, P):
    """a-a': bisulfate sulfate [WM13]."""
    return theta_none(T, P) # WM13 Table A7

def psi_H_Cl_SO4_WM13(T, P):
    """c-a-a': hydrogen chloride sulfate [WM13]."""
    return psi_none(T, P) # WM13 Table A8

def psi_H_Cl_OH_WM13(T, P):
    """c-a-a': hydrogen chloride hydroxide [WM13]."""
    return psi_none(T, P) # WM13 Table A8

def psi_Mg_Cl_OH_WM13(T, P):
    """c-a-a': magnesium chloride hydroxide [WM13]."""
    return psi_none(T, P) # WM13 Table A8

def psi_Ca_HSO4_SO4_WM13(T, P):
    """c-a-a': calcium bisulfate sulfate [WM13]."""
    return psi_none(T, P) # WM13 Table A8

def psi_H_OH_SO4_WM13(T, P):
    """c-a-a': hydrogen hydroxide sulfate [WM13]."""
    return psi_none(T, P) # WM13 Table A8

def psi_Mg_OH_SO4_WM13(T, P):
    """c-a-a': magnesium hydroxide sulfate [WM13]."""
    return psi_none(T, P) # WM13 Table A8

def psi_Ca_OH_SO4_WM13(T, P):
    """c-a-a': calcium hydroxide sulfate [WM13]."""
    return psi_none(T, P) # WM13 Table A8

def psi_H_Na_SO4_WM13(T, P):
    """c-c'-a: hydrogen sodium sulfate [WM13]."""
    return psi_none(T, P) # WM13 Table A9

def psi_Ca_H_SO4_WM13(T, P):
    """c-c'-a: calcium hydrogen sulfate [WM13]."""
    return psi_none(T, P) # WM13 Table A9

def psi_Ca_H_HSO4_WM13(T, P):
    """c-c'-a: calcium hydrogen bisulfate [WM13]."""
    return psi_none(T, P) # WM13 Table A9

def psi_Mg_Na_HSO4_WM13(T, P):
    """c-c'-a: magnesium sodium bisulfate [WM13]."""
    return psi_none(T, P) # WM13 Table A9

def psi_Ca_Na_HSO4_WM13(T, P):
    """c-c'-a: calcium sodium bisulfate [WM13]."""
    return psi_none(T, P) # WM13 Table A9

def psi_K_Na_HSO4_WM13(T, P):
    """c-c'-a: potassium sodium bisulfate [WM13]."""
    return psi_none(T, P) # WM13 Table A9

def psi_Ca_Mg_HSO4_WM13(T, P):
    """c-c'-a: calcium magnesium bisulfate [WM13]."""
    return psi_none(T, P) # WM13 Table A9

def psi_K_Mg_HSO4_WM13(T, P):
    """c-c'-a: potassium magnesium bisulfate [WM13]."""
    return psi_none(T, P) # WM13 Table A9

def psi_Ca_K_SO4_WM13(T, P):
    """c-c'-a: calcium potassium sulfate [WM13]."""
    return psi_none(T, P) # WM13 Table A9

def psi_Ca_K_HSO4_WM13(T, P):
    """c-c'-a: calcium potassium bisulfate [WM13]."""
    return psi_none(T, P) # WM13 Table A9

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Gallego-Urrea and Turner (2017) ~~~~~
# From G17 Supp. Info. Table S6, 'simultaneous optimisation'.
def bC_Na_Cl_GT17simopt(T, P):
    """c-a: sodium chloride [GT17simopt]."""
    b0 = 0.07722
    b1 = 0.26768
    b2 = 0
    Cphi = 0.001628
    zNa = +1
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(zNa * zCl)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_trisH_Cl_GT17simopt(T, P):
    """c-a: trisH+ chloride [GT17simopt]."""
    b0 = 0.04181
    b1 = 0.16024
    b2 = 0
    Cphi = -0.00132
    ztrisH = +1
    zCl = -1
    C0 = Cphi / (2 * sqrt(np_abs(ztrisH * zCl)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_trisH_SO4_GT17simopt(T, P):
    """c-a: trisH+ sulfate [GT17simopt]."""
    b0 = 0.09746
    b1 = 0.52936
    b2 = 0
    Cphi = -0.004957
    ztrisH = +1
    zSO4   = -2
    C0 = Cphi / (2 * sqrt(np_abs(ztrisH * zSO4)))
    C1 = 0
    alph1 = 2
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def theta_H_trisH_GT17simopt(T, P):
    """c-c': hydrogen trisH [GT17simopt]."""
    theta = -0.00575
    valid = T == 298.15
    return theta, valid

def psi_H_trisH_Cl_GT17simopt(T, P):
    """c-c'-a: hydrogen trisH chloride [GT17simopt]."""
    psi = -0.00700
    valid = T == 298.15
    return psi, valid

def lambd_tris_trisH_GT17simopt(T, P):
    """n-c: tris trisH [GT17simopt]."""
    lambd = 0.06306
    valid = T == 298.15
    return lambd, valid

def lambd_tris_Na_GT17simopt(T, P):
    """n-c: tris sodium [GT17simopt]."""
    lambd = 0.01580
    valid = T == 298.15
    return lambd, valid

def lambd_tris_K_GT17simopt(T, P):
    """n-c: tris potassium [GT17simopt]."""
    lambd = 0.02895
    valid = T == 298.15
    return lambd, valid

def lambd_tris_Mg_GT17simopt(T, P):
    """n-c: tris magnesium [GT17simopt]."""
    lambd = -0.14505
    valid = T == 298.15
    return lambd, valid

def lambd_tris_Ca_GT17simopt(T, P):
    """n-c: tris calcium [GT17simopt]."""
    lambd = -0.31081
    valid = T == 298.15
    return lambd, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Zezin and Driesner (2017) ~~~~~
def ZD17_eq8(T, P, b):
    """ZD17 equation 8, pressure in MPa."""
    return (b[ 0]
          + b[ 1] * T/1000
          + b[ 2] * (T/500)**2
          + b[ 3] / (T - 215)
          + b[ 4] * 1e4 / (T - 215)**3
          + b[ 5] * 1e2 / (T - 215)**2
          + b[ 6] * 2e2 /  T**2
          + b[ 7] * (T/500)**3
          + b[ 8] / (650 - T)**0.5
          + b[ 9] * 1e-5 * P
          + b[10] * 2e-4 * P / (T - 225)
          + b[11] * 1e2  * P / (650 - T)**3
          + b[12] * 1e-5 * P *  T/500
          + b[13] * 2e-4 * P / (650 - T)
          + b[14] * 1e-7 * P**2
          + b[15] * 2e-6 * P**2 / (T - 225)
          + b[16] * P**2 / (650 - T)**3
          + b[17] * 1e-7 * P**2 *  T/500
          + b[18] * 1e-7 * P**2 * (T/500)**2
          + b[19] * 4e-2 * P / (T - 225)**2
          + b[20] * 1e-5 * P * (T/500)**2
          + b[21] * 2e-8 * P**3 / (T - 225)
          + b[22] * 1e-2 * P**3 / (650 - T)**3
          + b[23] * 2e2  / (650 - T)**3)

def bC_K_Cl_ZD17(T, P):
    """c-a: potassium chloride [ZD17]."""
    P_MPa = P / 100 # Convert dbar to MPa
    # KCl T and P parameters from ZD17 Table 2
    b0 = ZD17_eq8(T, P_MPa, [
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
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ])
    b1 = ZD17_eq8(T, P_MPa, [
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
         0, 0, 0, 0,
         4.0457998,
         0, 0,
        -162.81428,
         296.7078,
         0,
        -0.7343191,
         46.340392,
    ])
    b2 = 0
    C0 = ZD17_eq8(T, P_MPa, [
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
         0, 0, 0, 0, 0,
        -5.4129002,
         0, 0, 0, 0,
    ])
    C1 = ZD17_eq8(T, P_MPa, [
         0,
         1.0025605,
         0, 0,
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
         0, 0,
        -4.7090241,
         0, 0,
         542.1083,
         0, 0,
         1.6548655,
         59.165704,
    ])
    alph1 = 2
    alph2 = -9
    omega = 2.5
    valid = T <= 600
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MarChemSpec project ~~~~~
def theta_Ca_H_MarChemSpec(T, P):
    """c-c': calcium hydrogen [MarChemSpec]."""
    # 1. WM13 cite the wrong reference for this (they say RXX80)
    # 2. The equation given by WM13 doesn't match RGO82
    # 3. RGO82 give a 25degC value but no temperature parameter
    # So MarChemSpec uses RGO82's 25degC value plus the WM13 temperature cxn
    thetar = theta_Ca_H_RGO82(T, P)[0]
    theta = thetar + 3.275e-4*(T - 298.15)
    valid = logical_and(T >= 273.15, T <= 323.15)
    return theta, valid

def theta_H_Na_MarChemSpec25(T, P):
    """c-c': hydrogen sodium [MarChemSpec]."""
    theta = 0.036
    valid = T == 298.15
    return theta, valid

def theta_H_K_MarChemSpec25(T, P):
    """c-c': hydrogen potassium [MarChemSpec]."""
    theta = 0.005
    valid = T == 298.15
    return theta, valid

def lambd_tris_tris_MarChemSpec25(T, P):
    """n-n: tris tris [MarChemSpec]."""
    # Temporary value from "MODEL PARAMETERS FOR TRIS Tests.docx" (2019-01-31)
    lambd = -0.006392
    valid = T == 298.15
    return lambd, valid

def zeta_tris_Na_Cl_MarChemSpec25(T, P):
    """n-c-a: tris sodium chloride [MarChemSpec]."""
    # Temporary value from "MODEL PARAMETERS FOR TRIS Tests.docx" (2019-01-31)
    zeta = -0.003231
    valid = T == 298.15
    return zeta, valid

def mu_tris_tris_tris_MarChemSpec25(T, P):
    """n-n-n: tris tris tris [MarChemSpec]."""
    # Temporary value from "MODEL PARAMETERS FOR TRIS Tests.docx" (2019-01-31)
    mu = 0.0009529
    valid = T == 298.15
    return mu, valid
