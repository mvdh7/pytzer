# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
"""Evaluate thermodynamic equilibrium constants."""
from autograd.numpy import log, log10

def HSO4_CRP94_extra(tempK, pres):
    """Bisulfate dissociation: CRP94 Eq. (21) with extra digits on constant
    term (S.L. Clegg, pers. comm, 7 Feb 2019).
    """
    log10kHSO4 = (562.694864456 - 102.5154*log(tempK) - 1.117033e-4*tempK**2
        + 0.2477538*tempK - 13273.75/tempK)
    lnkHSO4 = log10kHSO4*log(10)
    return lnkHSO4

def HSO4_CRP94(tempK, pres):
    """CRP94 Eq. (21) without extra digits on constant term."""
    # Matches Clegg's model [2019-07-02]
    log10kHSO4 = (562.69486 - 102.5154*log(tempK) - 1.117033e-4*tempK**2
        + 0.2477538*tempK - 13273.75/tempK)
    lnkHSO4 = log10kHSO4*log(10)
    return lnkHSO4

def trisH_BH64(tempK, pres):
    """TrisH+ dissociation following BH64 Eq. (3)."""
    # Matches Clegg's model [2019-07-02]
    log10ktrisH = -(2981.4/tempK - 3.5888 + 0.005571*tempK)
    lnktrisH = log10ktrisH*log(10)
    return lnktrisH

def rhow_K75(tempK, pres):
    """Water density in g/cm**3 following Kell (1975) J. Chem. Eng. Data 20(1),
    97-105.
    """
    # Matches Clegg's model [2019-07-02]
    tempC = tempK - 273.15
    return (0.99983952 + 16.945176e-3*tempC - 7.9870401e-6*tempC**2
        - 46.170461e-9*tempC**3 + 105.56302e-12*tempC**4
        - 280.54253e-15*tempC**5) / (1 + 16.879850e-3*tempC)

def H2O_M79(tempK, pres):
    """ Water dissociation following M79."""
    # MP98 says this is HO58 refitted by M79
    return 148.9802 - 13847.26/tempK - 23.6521*log(tempK)

def H2O_M88(tempK, pres):
    """Water dissociation following M88."""
    return (1.04031130e+3 + 4.86092851e-1*tempK - 3.26224352e+4/tempK
        - 1.90877133e+2*log(tempK) - 5.35204850e-1/(tempK - 263)
        - 2.32009393e-4*tempK**2 + 5.20549183e+1/(680 - tempK))

def H2O_MF(tempK, pres):
    """Marshall and Frank, J. Phys. Chem. Ref. Data 10, 295-304."""
    # Matches Clegg's model [2019-07-02]
    log10kH2O = (-4.098 - 3.2452e3/tempK + 2.2362e5/tempK**2 - 3.984e7/tempK**3
        + (1.3957e1 - 1.2623e3/tempK + 8.5641e5/tempK**2)
        * log10(rhow_K75(tempK, pres)))
    lnkH2O = log10kH2O*log(10)
    return lnkH2O

def Mg_CW91_ln(tempK, pres):
    """MgOH+ formation following CW91 Eq. (244) [p392]."""
    return 8.9108 - 1155/tempK

def Mg_CW91(tempK, pres):
    """MgOH+ formation following CW91 in log10 and then converted."""
    # Matches Clegg's model [2019-07-02]
    log10kMg = 3.87 - 501.5/tempK
    lnkMg = log10kMg*log(10)
    return lnkMg
