from autograd.numpy import array, exp, pi, sqrt

NA = 6.0221367e+23 # Avogadro's constant in 1 / mol

# AW90 Eq. (3): (g - 1) * rho
def gm1drho(tempK, pres):
    
    # Produces values like in AW90 Fig. 1
    
    # tempK in K
    # pres in MPa
        
    # AW90 Table 2
    b = [-4.044525e-2, #0
         103.6180, #1
         75.32165, #2
         -23.23778, #3
         -3.548184, #4
         -1246.311, #5
         263307.7, #6
         -6.928953e-1, #7
         -204.4473] #8
    
    # AW90 Eq. (3)
    gm1drho = \
        b[0] * pres / tempK + \
        b[1] / sqrt(tempK) + \
        b[2] / (tempK - 215) + \
        b[3] / sqrt(tempK - 215) + \
        b[4] / (tempK - 215)**0.25 + \
        exp( \
            b[5] / tempK + \
            b[6] / tempK**2 + \
            b[7] * pres / tempK + \
            b[8] * pres / tempK**2 \
        )
    
    return gm1drho


# Calculate g given density too (rho)
def g(tempK, pres, rho):
    
    return gm1drho(tempK, pres) * rho + 1


# Dielectric constant following Archer's DIEL()
def D(tempK, pres, rho):
    
    Mw = 18.0153
    al = 1.444e-24
    k  = 1.380658e-16
    mu = 1.84e-18
    
    A = (al + g(tempK, pres, rho) * mu**2 / (3 * k * tempK)) * \
        4 * pi * NA * rho / (3 * Mw)
        
    return (1 + 9 * A + 3 * sqrt(9 * A**2 + 2 * A + 1)) / 4
   

# Debye-Hueckel limiting slope for osmotic coefficient following dhll.for
def Aosm(tempK, pres, rho):
    
    # Constants from Table 1 footnote
    e  = 1.6021773e-19 # charge on an electron in C
    E0 = 8.8541878e-12 # permittivity of vacuum in C**2 / (J * m)
    
    # Constants from Table 2 footnote
    # al = 18.1458392e-30 # molecular polarisability in m**3
    # mu = 6.1375776e-30 # dipole moment with no electric field in C * m
    k  = 1.380658e-23 # Boltzmann constant in J / K
    # Mw = 0.0180153 # molecular mass of water in kg / mol

    return sqrt(2e-3 * pi * rho * NA) * (e**2 * 100 / \
        (4 * pi * D(tempK, pres, rho) * E0 * k * tempK))**1.5 / 3


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
tempK = array([298.15, 473.15, 623.15, 773.15])
pres = 0.101325 # 1 atm in MPa
rho = 1

testgm1drho = gm1drho(tempK, pres)

testg = g(tempK, pres, rho)

testD = D(tempK, pres, rho)

testAosm = Aosm(tempK, pres, rho)


import pytzer as pz
pzAosm = pz.coeffs.Aosm_CRP94(tempK)[0][0]
