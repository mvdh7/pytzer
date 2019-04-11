from autograd.numpy import exp, pi, sqrt
import autograd.numpy as np

NA = 6.0221367e+23 # Avogadro's constant in 1 / mol

def gm1drho(tempK, pres):
    """AW90 Eq. (3): (g - 1) * rho."""
    # Produces values like in AW90 Fig. 1
    # tempK in K, pres in MPa
    # AW90 Table 2:
    b = [
        -4.044525e-2,
        103.6180,
        75.32165,
        -23.23778,
        -3.548184,
        -1246.311,
        263307.7,
        -6.928953e-1,
        -204.4473,
    ]
    # AW90 Eq. (3):
    gm1drho = \
        b[0] * pres/tempK + \
        b[1] / sqrt(tempK) + \
        b[2] / (tempK - 215) + \
        b[3] / sqrt(tempK - 215) + \
        b[4] / (tempK - 215)**0.25 + \
        exp( \
            b[5] / tempK + \
            b[6] / tempK**2 + \
            b[7] * pres/tempK + \
            b[8] * pres/tempK**2 \
    ) 
    return gm1drho

def g(tempK, pres, rho):
    """Calculate g given density."""
    return gm1drho(tempK, pres)*rho + 1

def D(tempK, pres, rho):
    """Dielectric constant following Archer's DIEL()."""
    # Note that Archer's code uses different values from AW90 just in this
    # subroutine (so also different from in Aosm calculation below)
    Mw = 18.0153
    al = 1.444e-24
    k  = 1.380658e-16
    mu = 1.84e-18
    A = (al + g(tempK, pres, rho)*mu**2 / (3*k*tempK)) * 4*pi*NA*rho/(3*Mw)    
    return (1 + 9*A + 3*sqrt(9*A**2 + 2*A + 1)) / 4

def Aosm(tempK, pres, rho):
    """D-H limiting slope for osmotic coefficient, following dhll.for."""
    # Constants from Table 1 footnote:
    e  = 1.6021773e-19 # charge on an electron in C
    E0 = 8.8541878e-12 # permittivity of vacuum in C**2 / (J * m)
    # Constants from Table 2 footnote:
    # al = 18.1458392e-30 # molecular polarisability in m**3
    # mu = 6.1375776e-30 # dipole moment with no electric field in C * m
    k  = 1.380658e-23 # Boltzmann constant in J / K
    # Mw = 0.0180153 # molecular mass of water in kg / mol
    return sqrt(2e-3*pi*rho*NA) * (100*e**2 / \
        (4*pi * D(tempK, pres, rho) * E0*k*tempK))**1.5 / 3

def rhoWP93(tempK):
    """Density of the saturated liquid", Eq. (2) from WP93."""  
    # output units in kg / m**3
    # Temperature conversion from WP93 Section 1 on Nomenclature
    tau = 1 - tempK/647.096
    # Coefficients from WP93 Section 4.1:
    b = [
        1,
        1.99274064,
        1.09965342,
        -0.510839303,
        -1.75493479,
        -45.5170352,
        -6.74694450e5,
    ] 
    # WP93 Eq. (2):
    rhodrhoc = \
        b[0] + \
        b[1] * tau**(  1/3) + \
        b[2] * tau**(  2/3) + \
        b[3] * tau**(  5/3) + \
        b[4] * tau**( 16/3) + \
        b[5] * tau**( 43/3) + \
        b[6] * tau**(110/3)
    return rhodrhoc * 322

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
#tempK = array([298.15, 473.15, 623.15, 773.15])
#pres = 0.101325 # 1 atm in MPa
    
tempK = np.arange(263.15, 314.15, 5)
pres = np.full_like(tempK, 70.0)

import pytzer as pz

#rho = rhoWP93(tempK) * 1e-3
rho = pz.teos10.rho(tempK, pres*1e6) * 1e-3
testgm1drho = gm1drho(tempK, pres)
testg = g(tempK, pres, rho)
testD = D(tempK, pres, rho)
testAosm = Aosm(tempK, pres, rho)

#print(teos.rho(tempK, pres))

#%% Compare at 298.15 vs pytzer
from matplotlib import pyplot as plt
pzAosm = pz.coeffs.Aosm_MarChemSpec(tempK)[0]

fig, ax = plt.subplots(1, 1)
ax.plot(tempK, testAosm)
ax.plot(tempK, pzAosm)
