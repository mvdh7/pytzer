def bC_Na_Cl_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.0765, dtype='float64')
    b1   = np.full_like(T,0.2644, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.00127, dtype='float64')
    zNa = np.float_(1)
    zCl = np.float_(-1)
    C0 = Cphi / (2 * np.sqrt(np.abs(zNa*zCl)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_SO4_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.01958, dtype='float64')
    b1   = np.full_like(T,1.113, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.00497, dtype='float64')
    zNa = np.float_(1)
    zSO4 = np.float_(-2)
    C0 = Cphi / (2 * np.sqrt(np.abs(zNa*zSO4)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_HSO4_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.0454, dtype='float64')
    b1   = np.full_like(T,0.398, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0, dtype='float64')
    zNa = np.float_(1)
    zHSO4 = np.float_(-1)
    C0 = Cphi / (2 * np.sqrt(np.abs(zNa*zHSO4)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_OH_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.0864, dtype='float64')
    b1   = np.full_like(T,0.253, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0044, dtype='float64')
    zNa = np.float_(1)
    zOH = np.float_(-1)
    C0 = Cphi / (2 * np.sqrt(np.abs(zNa*zOH)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_HCO3_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.0277, dtype='float64')
    b1   = np.full_like(T,0.0411, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0, dtype='float64')
    zNa = np.float_(1)
    zHCO3 = np.float_(-1)
    C0 = Cphi / (2 * np.sqrt(np.abs(zNa*zHCO3)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Na_CO3_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.0399, dtype='float64')
    b1   = np.full_like(T,1.389, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0044, dtype='float64')
    zNa = np.float_(1)
    zCO3 = np.float_(-2)
    C0 = Cphi / (2 * np.sqrt(np.abs(zNa*zCO3)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_Cl_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.04835, dtype='float64')
    b1   = np.full_like(T,0.2122, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,-0.00084, dtype='float64')
    zK = np.float_(1)
    zCl = np.float_(-1)
    C0 = Cphi / (2 * np.sqrt(np.abs(zK*zCl)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_SO4_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.04995, dtype='float64')
    b1   = np.full_like(T,0.7793, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0, dtype='float64')
    zK = np.float_(1)
    zSO4 = np.float_(-2)
    C0 = Cphi / (2 * np.sqrt(np.abs(zK*zSO4)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_HSO4_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,-0.0003, dtype='float64')
    b1   = np.full_like(T,0.1735, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0, dtype='float64')
    zK = np.float_(1)
    zHSO4 = np.float_(-1)
    C0 = Cphi / (2 * np.sqrt(np.abs(zK*zHSO4)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_OH_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.1298, dtype='float64')
    b1   = np.full_like(T,0.32, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0041, dtype='float64')
    zK = np.float_(1)
    zOH = np.float_(-1)
    C0 = Cphi / (2 * np.sqrt(np.abs(zK*zOH)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_HCO3_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.0296, dtype='float64')
    b1   = np.full_like(T,-0.013, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,-0.008, dtype='float64')
    zK = np.float_(1)
    zHCO3 = np.float_(-1)
    C0 = Cphi / (2 * np.sqrt(np.abs(zK*zHCO3)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_K_CO3_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.1488, dtype='float64')
    b1   = np.full_like(T,1.43, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,-0.0015, dtype='float64')
    zK = np.float_(1)
    zCO3 = np.float_(-2)
    C0 = Cphi / (2 * np.sqrt(np.abs(zK*zCO3)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Ca_Cl_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.3159, dtype='float64')
    b1   = np.full_like(T,1.614, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,-0.00034, dtype='float64')
    zCa = np.float_(2)
    zCl = np.float_(-1)
    C0 = Cphi / (2 * np.sqrt(np.abs(zCa*zCl)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Ca_SO4_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.2, dtype='float64')
    b1   = np.full_like(T,3.1973, dtype='float64')
    b2   = np.full_like(T,-54.24, dtype='float64')
    Cphi = np.full_like(T,0.0, dtype='float64')
    zCa = np.float_(2)
    zSO4 = np.float_(-2)
    C0 = Cphi / (2 * np.sqrt(np.abs(zCa*zSO4)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(1.4)
    alph2 = np.float_(12)
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Ca_HSO4_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.2145, dtype='float64')
    b1   = np.full_like(T,2.53, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0, dtype='float64')
    zCa = np.float_(2)
    zHSO4 = np.float_(-1)
    C0 = Cphi / (2 * np.sqrt(np.abs(zCa*zHSO4)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Ca_OH_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,-0.1747, dtype='float64')
    b1   = np.full_like(T,-0.2303, dtype='float64')
    b2   = np.full_like(T,-5.72, dtype='float64')
    Cphi = np.full_like(T,0.0, dtype='float64')
    zCa = np.float_(2)
    zOH = np.float_(-1)
    C0 = Cphi / (2 * np.sqrt(np.abs(zCa*zOH)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(1.4)
    alph2 = np.float_(12)
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Ca_HCO3_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.4, dtype='float64')
    b1   = np.full_like(T,2.977, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0, dtype='float64')
    zCa = np.float_(2)
    zHCO3 = np.float_(-1)
    C0 = Cphi / (2 * np.sqrt(np.abs(zCa*zHCO3)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Ca_CO3_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.0, dtype='float64')
    b1   = np.full_like(T,0.0, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0, dtype='float64')
    zCa = np.float_(2)
    zCO3 = np.float_(-2)
    C0 = Cphi / (2 * np.sqrt(np.abs(zCa*zCO3)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Mg_Cl_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.35235, dtype='float64')
    b1   = np.full_like(T,1.6815, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.00519, dtype='float64')
    zMg = np.float_(2)
    zCl = np.float_(-1)
    C0 = Cphi / (2 * np.sqrt(np.abs(zMg*zCl)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Mg_SO4_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.221, dtype='float64')
    b1   = np.full_like(T,3.343, dtype='float64')
    b2   = np.full_like(T,-37.23, dtype='float64')
    Cphi = np.full_like(T,0.025, dtype='float64')
    zMg = np.float_(2)
    zSO4 = np.float_(-2)
    C0 = Cphi / (2 * np.sqrt(np.abs(zMg*zSO4)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(1.4)
    alph2 = np.float_(12)
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Mg_HSO4_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.4746, dtype='float64')
    b1   = np.full_like(T,1.729, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0, dtype='float64')
    zMg = np.float_(2)
    zHSO4 = np.float_(-1)
    C0 = Cphi / (2 * np.sqrt(np.abs(zMg*zHSO4)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Mg_OH_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.0, dtype='float64')
    b1   = np.full_like(T,0.0, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0, dtype='float64')
    zMg = np.float_(2)
    zOH = np.float_(-1)
    C0 = Cphi / (2 * np.sqrt(np.abs(zMg*zOH)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Mg_HCO3_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.329, dtype='float64')
    b1   = np.full_like(T,0.6072, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0, dtype='float64')
    zMg = np.float_(2)
    zHCO3 = np.float_(-1)
    C0 = Cphi / (2 * np.sqrt(np.abs(zMg*zHCO3)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_Mg_CO3_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.0, dtype='float64')
    b1   = np.full_like(T,0.0, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0, dtype='float64')
    zMg = np.float_(2)
    zCO3 = np.float_(-2)
    C0 = Cphi / (2 * np.sqrt(np.abs(zMg*zCO3)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_MgOH_Cl_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,-0.1, dtype='float64')
    b1   = np.full_like(T,1.658, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0, dtype='float64')
    zMgOH = np.float_(1)
    zCl = np.float_(-1)
    C0 = Cphi / (2 * np.sqrt(np.abs(zMgOH*zCl)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_MgOH_SO4_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.0, dtype='float64')
    b1   = np.full_like(T,0.0, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0, dtype='float64')
    zMgOH = np.float_(1)
    zSO4 = np.float_(-2)
    C0 = Cphi / (2 * np.sqrt(np.abs(zMgOH*zSO4)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_MgOH_HSO4_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.0, dtype='float64')
    b1   = np.full_like(T,0.0, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0, dtype='float64')
    zMgOH = np.float_(1)
    zHSO4 = np.float_(-1)
    C0 = Cphi / (2 * np.sqrt(np.abs(zMgOH*zHSO4)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_MgOH_OH_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.0, dtype='float64')
    b1   = np.full_like(T,0.0, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0, dtype='float64')
    zMgOH = np.float_(1)
    zOH = np.float_(-1)
    C0 = Cphi / (2 * np.sqrt(np.abs(zMgOH*zOH)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_MgOH_HCO3_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.0, dtype='float64')
    b1   = np.full_like(T,0.0, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0, dtype='float64')
    zMgOH = np.float_(1)
    zHCO3 = np.float_(-1)
    C0 = Cphi / (2 * np.sqrt(np.abs(zMgOH*zHCO3)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_MgOH_CO3_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.0, dtype='float64')
    b1   = np.full_like(T,0.0, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0, dtype='float64')
    zMgOH = np.float_(1)
    zCO3 = np.float_(-2)
    C0 = Cphi / (2 * np.sqrt(np.abs(zMgOH*zCO3)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_H_Cl_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.1775, dtype='float64')
    b1   = np.full_like(T,0.2945, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0008, dtype='float64')
    zH = np.float_(1)
    zCl = np.float_(-1)
    C0 = Cphi / (2 * np.sqrt(np.abs(zH*zCl)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_H_SO4_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.0298, dtype='float64')
    b1   = np.full_like(T,0.0, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0438, dtype='float64')
    zH = np.float_(1)
    zSO4 = np.float_(-2)
    C0 = Cphi / (2 * np.sqrt(np.abs(zH*zSO4)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_H_HSO4_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.2065, dtype='float64')
    b1   = np.full_like(T,0.5556, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0, dtype='float64')
    zH = np.float_(1)
    zHSO4 = np.float_(-1)
    C0 = Cphi / (2 * np.sqrt(np.abs(zH*zHSO4)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_H_OH_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.0, dtype='float64')
    b1   = np.full_like(T,0.0, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0, dtype='float64')
    zH = np.float_(1)
    zOH = np.float_(-1)
    C0 = Cphi / (2 * np.sqrt(np.abs(zH*zOH)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_H_HCO3_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.0, dtype='float64')
    b1   = np.full_like(T,0.0, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0, dtype='float64')
    zH = np.float_(1)
    zHCO3 = np.float_(-1)
    C0 = Cphi / (2 * np.sqrt(np.abs(zH*zHCO3)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

def bC_H_CO3_HMW84(T):
# Coefficients from HMW84 Table 1
    b0   = np.full_like(T,0.0, dtype='float64')
    b1   = np.full_like(T,0.0, dtype='float64')
    b2   = np.full_like(T,0.0, dtype='float64')
    Cphi = np.full_like(T,0.0, dtype='float64')
    zH = np.float_(1)
    zCO3 = np.float_(-2)
    C0 = Cphi / (2 * np.sqrt(np.abs(zH*zCO3)))
    C1 = np.zeros_like(T)
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    valid = T == 298.15
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

