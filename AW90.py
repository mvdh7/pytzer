from autograd.numpy import array, exp, sqrt

def gm1drho(tempK, pres):
    
    # tempK in K
    # pres in MPa
    
    # AW90 Table 2
    b = [-4.044525e-2,
         103.6180,
         75.32165,
         -23.23778,
         -3.548184,
         -1246.311,
         263307.7,
         -6.928953e-1,
         -204.4473]
    
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

tempK = array([473.15, 623.15, 773.15])
    
test = gm1drho(tempK, 1)
