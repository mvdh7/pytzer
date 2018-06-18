import autograd.numpy as np

def Aosm_CRP94(t): # CRP94 Appendix II

    # Transform temperature
    X = (2 * t - 373.15 - 234.15) / (373.15 - 234.15)

    # Set coefficients - CRP94 Table 11
    a_Aosm = np.array( \
             [ 0.797256081240 / 2,
               0.573389669896e-1,
               0.977632177788e-3,
               0.489973732417e-2,
              -0.313151784342e-2,
               0.179145971002e-2,
              -0.920584241844e-3,
               0.443862726879e-3,
              -0.203661129991e-3,
               0.900924147948e-4,
              -0.388189392385e-4,
               0.164245088592e-4,
              -0.686031972567e-5,
               0.283455806377e-5,
              -0.115641433004e-5,
               0.461489672579e-6,
              -0.177069754948e-6,
               0.612464488231e-7,
              -0.175689013085e-7])

    # Set up T matrix - CRP94 Eq. (AII2)
    Tmx = np.full((np.size(t),np.size(a_Aosm)),1.)
    Tmx[:,1] = X
    for C in range(2,np.size(a_Aosm)):
        Tmx[:,C] = 2 * X * Tmx[:,C-1] - Tmx[:,C-2]

    # Solve for Aosm (CRP94 E.AII1)
    Aosm = np.matmul(Tmx,a_Aosm)

    # Validity range
    valid = np.logical_and(t >= 234.15,t <= 373.15)

    return Aosm, valid
