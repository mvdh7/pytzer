# --- bC: ----------------------------- 

def bC_H_Cl_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.1775 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.2945 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0008 * 1.0, dtype='float64')

    zH = np.float_(1)
    zCl = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zH*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_H_Br_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.196 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.3564 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.00827 * 1.0, dtype='float64')

    zH = np.float_(1)
    zBr = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zH*zBr)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_H_I_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.2362 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.392 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0011 * 1.0, dtype='float64')

    zH = np.float_(1)
    zI = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zH*zI)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_H_ClO4_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.1747 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.2931 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.00819 * 1.0, dtype='float64')

    zH = np.float_(1)
    zClO4 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zH*zClO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_H_NO3_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.1119 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.3206 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.001 * 1.0, dtype='float64')

    zH = np.float_(1)
    zNO3 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zH*zNO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Li_Cl_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.1494 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.3074 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.00359 * 1.0, dtype='float64')

    zLi = np.float_(1)
    zCl = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zLi*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Li_Br_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.1748 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.2547 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0053 * 1.0, dtype='float64')

    zLi = np.float_(1)
    zBr = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zLi*zBr)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Li_I_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.2104 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.373 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0 * 1.0, dtype='float64')

    zLi = np.float_(1)
    zI = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zLi*zI)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Li_OH_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.015 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.14 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0 * 1.0, dtype='float64')

    zLi = np.float_(1)
    zOH = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zLi*zOH)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Li_ClO4_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.1973 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.3996 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0008 * 1.0, dtype='float64')

    zLi = np.float_(1)
    zClO4 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zLi*zClO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Li_NO2_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.1336 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.325 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.0053 * 1.0, dtype='float64')

    zLi = np.float_(1)
    zNO2 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zLi*zNO2)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Li_NO3_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.142 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.278 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00551 * 1.0, dtype='float64')

    zLi = np.float_(1)
    zNO3 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zLi*zNO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_F_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.0215 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.2107 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0 * 1.0, dtype='float64')

    zNa = np.float_(1)
    zF = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zF)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_Cl_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.0765 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.2664 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.00127 * 1.0, dtype='float64')

    zNa = np.float_(1)
    zCl = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_Br_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.0973 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.2791 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.00116 * 1.0, dtype='float64')

    zNa = np.float_(1)
    zBr = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zBr)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_I_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.1195 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.3439 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0018 * 1.0, dtype='float64')

    zNa = np.float_(1)
    zI = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zI)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_OH_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.0864 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.253 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0044 * 1.0, dtype='float64')

    zNa = np.float_(1)
    zOH = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zOH)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_ClO3_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.0249 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.2455 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0004 * 1.0, dtype='float64')

    zNa = np.float_(1)
    zClO3 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zClO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_ClO4_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.0554 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.2755 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00118 * 1.0, dtype='float64')

    zNa = np.float_(1)
    zClO4 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zClO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_BrO3_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,-0.0205 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.191 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0059 * 1.0, dtype='float64')

    zNa = np.float_(1)
    zBrO3 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zBrO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_CNS_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.1005 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.3582 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00303 * 1.0, dtype='float64')

    zNa = np.float_(1)
    zCNS = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zCNS)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_NO2_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.0641 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.1015 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.0049 * 1.0, dtype='float64')

    zNa = np.float_(1)
    zNO2 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zNO2)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_NO3_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.0068 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.1783 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00072 * 1.0, dtype='float64')

    zNa = np.float_(1)
    zNO3 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zNO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_H2PO4_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,-0.0533 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.0396 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.00795 * 1.0, dtype='float64')

    zNa = np.float_(1)
    zH2PO4 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zH2PO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_H2AsO4_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,-0.0442 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.2895 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0 * 1.0, dtype='float64')

    zNa = np.float_(1)
    zH2AsO4 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zH2AsO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_BO2_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,-0.0526 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.1104 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0154 * 1.0, dtype='float64')

    zNa = np.float_(1)
    zBO2 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zBO2)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_BF4_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,-0.0252 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.1824 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0021 * 1.0, dtype='float64')

    zNa = np.float_(1)
    zBF4 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zBF4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_F_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.08089 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.2021 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.00093 * 1.0, dtype='float64')

    zK = np.float_(1)
    zF = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zF)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_Cl_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.04835 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.2122 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00084 * 1.0, dtype='float64')

    zK = np.float_(1)
    zCl = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_Br_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.0569 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.2212 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.0018 * 1.0, dtype='float64')

    zK = np.float_(1)
    zBr = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zBr)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_I_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.0746 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.2517 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00414 * 1.0, dtype='float64')

    zK = np.float_(1)
    zI = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zI)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_OH_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.1298 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.32 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0041 * 1.0, dtype='float64')

    zK = np.float_(1)
    zOH = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zOH)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_ClO3_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,-0.096 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.2481 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0 * 1.0, dtype='float64')

    zK = np.float_(1)
    zClO3 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zClO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_BrO3_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,-0.129 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.2565 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0 * 1.0, dtype='float64')

    zK = np.float_(1)
    zBrO3 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zBrO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_CNS_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.0416 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.2302 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00252 * 1.0, dtype='float64')

    zK = np.float_(1)
    zCNS = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zCNS)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_NO2_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.0151 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.015 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0007 * 1.0, dtype='float64')

    zK = np.float_(1)
    zNO2 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zNO2)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_NO3_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,-0.0816 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.0494 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0066 * 1.0, dtype='float64')

    zK = np.float_(1)
    zNO3 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zNO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_H2PO4_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,-0.0678 * 1.0, dtype='float64')
    b1   = np.full_like(T,-0.1042 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0 * 1.0, dtype='float64')

    zK = np.float_(1)
    zH2PO4 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zH2PO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_H2AsO4_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,-0.0584 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.0626 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0 * 1.0, dtype='float64')

    zK = np.float_(1)
    zH2AsO4 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zH2AsO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_PtF6_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,-0.163 * 1.0, dtype='float64')
    b1   = np.full_like(T,-0.282 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0 * 1.0, dtype='float64')

    zK = np.float_(1)
    zPtF6 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zPtF6)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Rb_F_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.1141 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.2842 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.0105 * 1.0, dtype='float64')

    zRb = np.float_(1)
    zF = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zRb*zF)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Rb_Cl_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.0441 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.1483 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00101 * 1.0, dtype='float64')

    zRb = np.float_(1)
    zCl = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zRb*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Rb_Br_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.0396 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.153 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00144 * 1.0, dtype='float64')

    zRb = np.float_(1)
    zBr = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zRb*zBr)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Rb_I_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.0397 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.133 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00108 * 1.0, dtype='float64')

    zRb = np.float_(1)
    zI = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zRb*zI)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Rb_NO2_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.0269 * 1.0, dtype='float64')
    b1   = np.full_like(T,-0.1553 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00366 * 1.0, dtype='float64')

    zRb = np.float_(1)
    zNO2 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zRb*zNO2)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Rb_NO3_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,-0.0789 * 1.0, dtype='float64')
    b1   = np.full_like(T,-0.0172 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.00529 * 1.0, dtype='float64')

    zRb = np.float_(1)
    zNO3 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zRb*zNO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Cs_F_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.1306 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.257 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.0043 * 1.0, dtype='float64')

    zCs = np.float_(1)
    zF = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCs*zF)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Cs_Cl_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.03 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.0558 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.00038 * 1.0, dtype='float64')

    zCs = np.float_(1)
    zCl = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCs*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Cs_Br_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.0279 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.0139 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,4e-05 * 1.0, dtype='float64')

    zCs = np.float_(1)
    zBr = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCs*zBr)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Cs_I_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.0244 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.0262 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00365 * 1.0, dtype='float64')

    zCs = np.float_(1)
    zI = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCs*zI)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Cs_OH_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.15 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.3 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0 * 1.0, dtype='float64')

    zCs = np.float_(1)
    zOH = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCs*zOH)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Cs_NO3_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,-0.0758 * 1.0, dtype='float64')
    b1   = np.full_like(T,-0.0669 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0 * 1.0, dtype='float64')

    zCs = np.float_(1)
    zNO3 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCs*zNO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Cs_NO2_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.0427 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.06 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.0051 * 1.0, dtype='float64')

    zCs = np.float_(1)
    zNO2 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCs*zNO2)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Ag_NO3_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,-0.0856 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.0025 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.00591 * 1.0, dtype='float64')

    zAg = np.float_(1)
    zNO3 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zAg*zNO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Tl_ClO4_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,-0.087 * 1.0, dtype='float64')
    b1   = np.full_like(T,-0.023 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0 * 1.0, dtype='float64')

    zTl = np.float_(1)
    zClO4 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zTl*zClO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Tl_NO3_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,-0.105 * 1.0, dtype='float64')
    b1   = np.full_like(T,-0.378 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0 * 1.0, dtype='float64')

    zTl = np.float_(1)
    zNO3 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zTl*zNO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_NH4_Cl_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.0522 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.1918 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00301 * 1.0, dtype='float64')

    zNH4 = np.float_(1)
    zCl = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNH4*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_NH4_Br_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,0.0624 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.1947 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00436 * 1.0, dtype='float64')

    zNH4 = np.float_(1)
    zBr = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNH4*zBr)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_NH4_ClO4_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,-0.0103 * 1.0, dtype='float64')
    b1   = np.full_like(T,-0.0194 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0 * 1.0, dtype='float64')

    zNH4 = np.float_(1)
    zClO4 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNH4*zClO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_NH4_NO3_PM73(T):

    # Coefficients from PM73 Table I

    b0   = np.full_like(T,-0.0154 * 1.0, dtype='float64')
    b1   = np.full_like(T,0.112 * 1.0, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-3e-05 * 1.0, dtype='float64')

    zNH4 = np.float_(1)
    zNO3 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNH4*zNO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Mg_Cl_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.4698 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.242 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.00979 * 0.5303300858899106, dtype='float64')

    zMg = np.float_(2)
    zCl = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zMg*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Mg_Br_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.5769 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.337 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.00589 * 0.5303300858899106, dtype='float64')

    zMg = np.float_(2)
    zBr = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zMg*zBr)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Mg_I_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.6536 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.4055 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.01496 * 0.5303300858899106, dtype='float64')

    zMg = np.float_(2)
    zI = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zMg*zI)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Mg_ClO4_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.6615 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.678 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.01806 * 0.5303300858899106, dtype='float64')

    zMg = np.float_(2)
    zClO4 = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zMg*zClO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Mg_NO3_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.4895 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.113 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.03889 * 0.5303300858899106, dtype='float64')

    zMg = np.float_(2)
    zNO3 = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zMg*zNO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Ca_Cl_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.4212 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.152 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00064 * 0.5303300858899106, dtype='float64')

    zCa = np.float_(2)
    zCl = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCa*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Ca_Br_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.5088 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.151 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00485 * 0.5303300858899106, dtype='float64')

    zCa = np.float_(2)
    zBr = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCa*zBr)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Ca_I_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.5839 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.409 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00158 * 0.5303300858899106, dtype='float64')

    zCa = np.float_(2)
    zI = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCa*zI)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Ca_ClO4_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.6015 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.342 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00943 * 0.5303300858899106, dtype='float64')

    zCa = np.float_(2)
    zClO4 = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCa*zClO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Ca_NO3_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.2811 * 0.75, dtype='float64')
    b1   = np.full_like(T,1.879 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.03798 * 0.5303300858899106, dtype='float64')

    zCa = np.float_(2)
    zNO3 = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCa*zNO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Sr_Cl_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.381 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.223 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00246 * 0.5303300858899106, dtype='float64')

    zSr = np.float_(2)
    zCl = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zSr*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Sr_Br_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.4415 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.282 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.00231 * 0.5303300858899106, dtype='float64')

    zSr = np.float_(2)
    zBr = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zSr*zBr)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Sr_I_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.535 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.48 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.00501 * 0.5303300858899106, dtype='float64')

    zSr = np.float_(2)
    zI = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zSr*zI)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Sr_ClO4_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.5692 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.089 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.02472 * 0.5303300858899106, dtype='float64')

    zSr = np.float_(2)
    zClO4 = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zSr*zClO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Sr_NO3_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.1795 * 0.75, dtype='float64')
    b1   = np.full_like(T,1.84 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.03757 * 0.5303300858899106, dtype='float64')

    zSr = np.float_(2)
    zNO3 = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zSr*zNO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Ba_Cl_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.3504 * 0.75, dtype='float64')
    b1   = np.full_like(T,1.995 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.03654 * 0.5303300858899106, dtype='float64')

    zBa = np.float_(2)
    zCl = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zBa*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Ba_Br_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.4194 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.093 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.03009 * 0.5303300858899106, dtype='float64')

    zBa = np.float_(2)
    zBr = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zBa*zBr)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Ba_I_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.5625 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.249 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.03286 * 0.5303300858899106, dtype='float64')

    zBa = np.float_(2)
    zI = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zBa*zI)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Ba_OH_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.229 * 0.75, dtype='float64')
    b1   = np.full_like(T,1.6 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0 * 0.5303300858899106, dtype='float64')

    zBa = np.float_(2)
    zOH = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zBa*zOH)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Ba_ClO4_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.4819 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.101 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.05894 * 0.5303300858899106, dtype='float64')

    zBa = np.float_(2)
    zClO4 = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zBa*zClO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Ba_NO3_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,-0.043 * 0.75, dtype='float64')
    b1   = np.full_like(T,1.07 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0 * 0.5303300858899106, dtype='float64')

    zBa = np.float_(2)
    zNO3 = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zBa*zNO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Mn_Cl_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.1363 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.067 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.03865 * 0.5303300858899106, dtype='float64')

    zMn = np.float_(2)
    zCl = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zMn*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Fe_Cl_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.4479 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.043 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.01623 * 0.5303300858899106, dtype='float64')

    zFe = np.float_(2)
    zCl = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zFe*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Co_Cl_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.4857 * 0.75, dtype='float64')
    b1   = np.full_like(T,1.936 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.02869 * 0.5303300858899106, dtype='float64')

    zCo = np.float_(2)
    zCl = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCo*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Co_Br_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.5693 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.213 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00127 * 0.5303300858899106, dtype='float64')

    zCo = np.float_(2)
    zBr = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCo*zBr)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Co_I_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.695 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.23 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.0088 * 0.5303300858899106, dtype='float64')

    zCo = np.float_(2)
    zI = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCo*zI)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Co_NO3_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.4159 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.254 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.01436 * 0.5303300858899106, dtype='float64')

    zCo = np.float_(2)
    zNO3 = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCo*zNO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Ni_Cl_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.4639 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.108 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00702 * 0.5303300858899106, dtype='float64')

    zNi = np.float_(2)
    zCl = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNi*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Cu_Cl_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.4107 * 0.75, dtype='float64')
    b1   = np.full_like(T,1.835 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.07624 * 0.5303300858899106, dtype='float64')

    zCu = np.float_(2)
    zCl = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCu*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Cu_NO3_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.4224 * 0.75, dtype='float64')
    b1   = np.full_like(T,1.907 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.04136 * 0.5303300858899106, dtype='float64')

    zCu = np.float_(2)
    zNO3 = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCu*zNO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Zn_Cl_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.3469 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.19 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.1659 * 0.5303300858899106, dtype='float64')

    zZn = np.float_(2)
    zCl = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zZn*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Zn_Br_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.6213 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.179 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.2035 * 0.5303300858899106, dtype='float64')

    zZn = np.float_(2)
    zBr = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zZn*zBr)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Zn_I_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.6428 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.594 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.0269 * 0.5303300858899106, dtype='float64')

    zZn = np.float_(2)
    zI = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zZn*zI)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Zn_ClO4_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.6747 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.396 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.02134 * 0.5303300858899106, dtype='float64')

    zZn = np.float_(2)
    zClO4 = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zZn*zClO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Zn_NO3_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.4641 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.255 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.02955 * 0.5303300858899106, dtype='float64')

    zZn = np.float_(2)
    zNO3 = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zZn*zNO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Cd_NO3_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.382 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.224 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.04836 * 0.5303300858899106, dtype='float64')

    zCd = np.float_(2)
    zNO3 = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCd*zNO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Pb_ClO4_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.4443 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.296 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.01667 * 0.5303300858899106, dtype='float64')

    zPb = np.float_(2)
    zClO4 = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zPb*zClO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Pb_NO3_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,-0.0482 * 0.75, dtype='float64')
    b1   = np.full_like(T,0.38 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.01005 * 0.5303300858899106, dtype='float64')

    zPb = np.float_(2)
    zNO3 = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zPb*zNO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_UO2_Cl_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.5698 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.192 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.06951 * 0.5303300858899106, dtype='float64')

    zUO2 = np.float_(2)
    zCl = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zUO2*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_UO2_ClO4_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.8151 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.859 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.04089 * 0.5303300858899106, dtype='float64')

    zUO2 = np.float_(2)
    zClO4 = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zUO2*zClO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_UO2_NO3_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.6143 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.151 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.05948 * 0.5303300858899106, dtype='float64')

    zUO2 = np.float_(2)
    zNO3 = np.float_(1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zUO2*zNO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Li_SO4_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.1817 * 0.75, dtype='float64')
    b1   = np.full_like(T,1.694 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00753 * 0.5303300858899106, dtype='float64')

    zLi = np.float_(1)
    zSO4 = np.float_(-2)
    C0  = Cphi / (2 * np.sqrt(np.abs(zLi*zSO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_SO4_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.0261 * 0.75, dtype='float64')
    b1   = np.full_like(T,1.484 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.00938 * 0.5303300858899106, dtype='float64')

    zNa = np.float_(1)
    zSO4 = np.float_(-2)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zSO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_S2O3_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.0882 * 0.75, dtype='float64')
    b1   = np.full_like(T,1.701 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.00705 * 0.5303300858899106, dtype='float64')

    zNa = np.float_(1)
    zS2O3 = np.float_(-2)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zS2O3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_CrO4_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.125 * 0.75, dtype='float64')
    b1   = np.full_like(T,1.826 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00407 * 0.5303300858899106, dtype='float64')

    zNa = np.float_(1)
    zCrO4 = np.float_(-2)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zCrO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_CO3_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.253 * 0.75, dtype='float64')
    b1   = np.full_like(T,1.128 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.09057 * 0.5303300858899106, dtype='float64')

    zNa = np.float_(1)
    zCO3 = np.float_(-2)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zCO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_HPO4_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,-0.0777 * 0.75, dtype='float64')
    b1   = np.full_like(T,1.954 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0554 * 0.5303300858899106, dtype='float64')

    zNa = np.float_(1)
    zHPO4 = np.float_(-2)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zHPO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_HAsO4_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.0407 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.173 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0034 * 0.5303300858899106, dtype='float64')

    zNa = np.float_(1)
    zHAsO4 = np.float_(-2)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zHAsO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_SO4_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.0666 * 0.75, dtype='float64')
    b1   = np.full_like(T,1.039 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0 * 0.5303300858899106, dtype='float64')

    zK = np.float_(1)
    zSO4 = np.float_(-2)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zSO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_CrO4_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.1011 * 0.75, dtype='float64')
    b1   = np.full_like(T,1.652 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00147 * 0.5303300858899106, dtype='float64')

    zK = np.float_(1)
    zCrO4 = np.float_(-2)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zCrO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_PtCN4_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.0881 * 0.75, dtype='float64')
    b1   = np.full_like(T,3.164 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0247 * 0.5303300858899106, dtype='float64')

    zK = np.float_(1)
    zPtCN4 = np.float_(-2)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zPtCN4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_HPO4_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.033 * 0.75, dtype='float64')
    b1   = np.full_like(T,1.699 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0309 * 0.5303300858899106, dtype='float64')

    zK = np.float_(1)
    zHPO4 = np.float_(-2)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zHPO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_HAsO4_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.1728 * 0.75, dtype='float64')
    b1   = np.full_like(T,2.198 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.0336 * 0.5303300858899106, dtype='float64')

    zK = np.float_(1)
    zHAsO4 = np.float_(-2)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zHAsO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Rb_SO4_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.0772 * 0.75, dtype='float64')
    b1   = np.full_like(T,1.481 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00019 * 0.5303300858899106, dtype='float64')

    zRb = np.float_(1)
    zSO4 = np.float_(-2)
    C0  = Cphi / (2 * np.sqrt(np.abs(zRb*zSO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Cs_SO4_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.1184 * 0.75, dtype='float64')
    b1   = np.full_like(T,1.481 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.01131 * 0.5303300858899106, dtype='float64')

    zCs = np.float_(1)
    zSO4 = np.float_(-2)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCs*zSO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_NH4_SO4_PM73(T):

    # Coefficients from PM73 Table VI

    b0   = np.full_like(T,0.0545 * 0.75, dtype='float64')
    b1   = np.full_like(T,0.878 * 0.75, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.00219 * 0.5303300858899106, dtype='float64')

    zNH4 = np.float_(1)
    zSO4 = np.float_(-2)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNH4*zSO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Al_Cl_PM73(T):

    # Coefficients from PM73 Table VIII

    b0   = np.full_like(T,1.049 * 0.6666666666666666, dtype='float64')
    b1   = np.full_like(T,8.767 * 0.6666666666666666, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0071 * 0.3849001794597505, dtype='float64')

    zAl = np.float_(3)
    zCl = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zAl*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Sr_Cl_PM73(T):

    # Coefficients from PM73 Table VIII

    b0   = np.full_like(T,1.05 * 0.6666666666666666, dtype='float64')
    b1   = np.full_like(T,7.978 * 0.6666666666666666, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.084 * 0.3849001794597505, dtype='float64')

    zSr = np.float_(3)
    zCl = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zSr*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Y_Cl_PM73(T):

    # Coefficients from PM73 Table VIII

    b0   = np.full_like(T,0.9599 * 0.6666666666666666, dtype='float64')
    b1   = np.full_like(T,8.166 * 0.6666666666666666, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.0587 * 0.3849001794597505, dtype='float64')

    zY = np.float_(3)
    zCl = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zY*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_La_Cl_PM73(T):

    # Coefficients from PM73 Table VIII

    b0   = np.full_like(T,0.9158 * 0.6666666666666666, dtype='float64')
    b1   = np.full_like(T,8.231 * 0.6666666666666666, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.0831 * 0.3849001794597505, dtype='float64')

    zLa = np.float_(3)
    zCl = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zLa*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Ce_Cl_PM73(T):

    # Coefficients from PM73 Table VIII

    b0   = np.full_like(T,0.9187 * 0.6666666666666666, dtype='float64')
    b1   = np.full_like(T,8.227 * 0.6666666666666666, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.0809 * 0.3849001794597505, dtype='float64')

    zCe = np.float_(3)
    zCl = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCe*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Pr_Cl_PM73(T):

    # Coefficients from PM73 Table VIII

    b0   = np.full_like(T,0.903 * 0.6666666666666666, dtype='float64')
    b1   = np.full_like(T,8.181 * 0.6666666666666666, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.0727 * 0.3849001794597505, dtype='float64')

    zPr = np.float_(3)
    zCl = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zPr*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Nd_Cl_PM73(T):

    # Coefficients from PM73 Table VIII

    b0   = np.full_like(T,0.9175 * 0.6666666666666666, dtype='float64')
    b1   = np.full_like(T,8.104 * 0.6666666666666666, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.0737 * 0.3849001794597505, dtype='float64')

    zNd = np.float_(3)
    zCl = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNd*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Sm_Cl_PM73(T):

    # Coefficients from PM73 Table VIII

    b0   = np.full_like(T,0.933 * 0.6666666666666666, dtype='float64')
    b1   = np.full_like(T,8.273 * 0.6666666666666666, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.0728 * 0.3849001794597505, dtype='float64')

    zSm = np.float_(3)
    zCl = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zSm*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Eu_Cl_PM73(T):

    # Coefficients from PM73 Table VIII

    b0   = np.full_like(T,0.937 * 0.6666666666666666, dtype='float64')
    b1   = np.full_like(T,8.385 * 0.6666666666666666, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.0687 * 0.3849001794597505, dtype='float64')

    zEu = np.float_(3)
    zCl = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zEu*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Cr_Cl_PM73(T):

    # Coefficients from PM73 Table VIII

    b0   = np.full_like(T,1.1046 * 0.6666666666666666, dtype='float64')
    b1   = np.full_like(T,7.883 * 0.6666666666666666, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.1172 * 0.3849001794597505, dtype='float64')

    zCr = np.float_(3)
    zCl = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCr*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Cr_NO3_PM73(T):

    # Coefficients from PM73 Table VIII

    b0   = np.full_like(T,1.056 * 0.6666666666666666, dtype='float64')
    b1   = np.full_like(T,7.777 * 0.6666666666666666, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.1533 * 0.3849001794597505, dtype='float64')

    zCr = np.float_(3)
    zNO3 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCr*zNO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Ga_ClO4_PM73(T):

    # Coefficients from PM73 Table VIII

    b0   = np.full_like(T,1.2381 * 0.6666666666666666, dtype='float64')
    b1   = np.full_like(T,9.794 * 0.6666666666666666, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0904 * 0.3849001794597505, dtype='float64')

    zGa = np.float_(3)
    zClO4 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zGa*zClO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_In_Cl_PM73(T):

    # Coefficients from PM73 Table VIII

    b0   = np.full_like(T,-1.68 * 0.6666666666666666, dtype='float64')
    b1   = np.full_like(T,-3.85 * 0.6666666666666666, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0 * 0.3849001794597505, dtype='float64')

    zIn = np.float_(3)
    zCl = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zIn*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_PO4_PM73(T):

    # Coefficients from PM73 Table VIII

    b0   = np.full_like(T,0.2672 * 0.6666666666666666, dtype='float64')
    b1   = np.full_like(T,5.777 * 0.6666666666666666, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.1339 * 0.3849001794597505, dtype='float64')

    zNa = np.float_(1)
    zPO4 = np.float_(-3)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zPO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_AsO4_PM73(T):

    # Coefficients from PM73 Table VIII

    b0   = np.full_like(T,0.3582 * 0.6666666666666666, dtype='float64')
    b1   = np.full_like(T,5.895 * 0.6666666666666666, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.124 * 0.3849001794597505, dtype='float64')

    zNa = np.float_(1)
    zAsO4 = np.float_(-3)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zAsO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_PO4_PM73(T):

    # Coefficients from PM73 Table VIII

    b0   = np.full_like(T,0.5594 * 0.6666666666666666, dtype='float64')
    b1   = np.full_like(T,5.958 * 0.6666666666666666, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.2255 * 0.3849001794597505, dtype='float64')

    zK = np.float_(1)
    zPO4 = np.float_(-3)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zPO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_P3O9_PM73(T):

    # Coefficients from PM73 Table VIII

    b0   = np.full_like(T,0.4867 * 0.6666666666666666, dtype='float64')
    b1   = np.full_like(T,8.349 * 0.6666666666666666, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.0886 * 0.3849001794597505, dtype='float64')

    zK = np.float_(1)
    zP3O9 = np.float_(-3)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zP3O9)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_AsO4_PM73(T):

    # Coefficients from PM73 Table VIII

    b0   = np.full_like(T,0.7491 * 0.6666666666666666, dtype='float64')
    b1   = np.full_like(T,6.511 * 0.6666666666666666, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.3376 * 0.3849001794597505, dtype='float64')

    zK = np.float_(1)
    zAsO4 = np.float_(-3)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zAsO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_FeCN6_PM73(T):

    # Coefficients from PM73 Table VIII

    b0   = np.full_like(T,0.5035 * 0.6666666666666666, dtype='float64')
    b1   = np.full_like(T,7.121 * 0.6666666666666666, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.1176 * 0.3849001794597505, dtype='float64')

    zK = np.float_(1)
    zFeCN6 = np.float_(-3)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zFeCN6)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_CoCN6_PM73(T):

    # Coefficients from PM73 Table VIII

    b0   = np.full_like(T,0.5603 * 0.6666666666666666, dtype='float64')
    b1   = np.full_like(T,5.815 * 0.6666666666666666, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.1603 * 0.3849001794597505, dtype='float64')

    zK = np.float_(1)
    zCoCN6 = np.float_(-3)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zCoCN6)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Coen3_Cl_PM73(T):

    # Coefficients from PM73 Table VIII

    b0   = np.full_like(T,0.2603 * 0.6666666666666666, dtype='float64')
    b1   = np.full_like(T,3.563 * 0.6666666666666666, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.0916 * 0.3849001794597505, dtype='float64')

    zCoen3 = np.float_(1)
    zCl = np.float_(-3)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCoen3*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Coen3_NO3_PM73(T):

    # Coefficients from PM73 Table VIII

    b0   = np.full_like(T,0.1882 * 0.6666666666666666, dtype='float64')
    b1   = np.full_like(T,3.935 * 0.6666666666666666, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0 * 0.3849001794597505, dtype='float64')

    zCoen3 = np.float_(1)
    zNO3 = np.float_(-3)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCoen3*zNO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Coen3_ClO4_PM73(T):

    # Coefficients from PM73 Table VIII

    b0   = np.full_like(T,0.1619 * 0.6666666666666666, dtype='float64')
    b1   = np.full_like(T,5.395 * 0.6666666666666666, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0 * 0.3849001794597505, dtype='float64')

    zCoen3 = np.float_(1)
    zClO4 = np.float_(-3)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCoen3*zClO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Copn3_ClO4_PM73(T):

    # Coefficients from PM73 Table VIII

    b0   = np.full_like(T,0.2022 * 0.6666666666666666, dtype='float64')
    b1   = np.full_like(T,3.976 * 0.6666666666666666, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0 * 0.3849001794597505, dtype='float64')

    zCopn3 = np.float_(1)
    zClO4 = np.float_(-3)
    C0  = Cphi / (2 * np.sqrt(np.abs(zCopn3*zClO4)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Th_Cl_PM73(T):

    # Coefficients from PM73 Table IX

    b0   = np.full_like(T,1.622 * 0.625, dtype='float64')
    b1   = np.full_like(T,21.33 * 0.625, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.3309 * 0.3125, dtype='float64')

    zTh = np.float_(4)
    zCl = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zTh*zCl)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Th_NO3_PM73(T):

    # Coefficients from PM73 Table IX

    b0   = np.full_like(T,1.546 * 0.625, dtype='float64')
    b1   = np.full_like(T,18.22 * 0.625, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.5906 * 0.3125, dtype='float64')

    zTh = np.float_(4)
    zNO3 = np.float_(-1)
    C0  = Cphi / (2 * np.sqrt(np.abs(zTh*zNO3)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_P2O7_PM73(T):

    # Coefficients from PM73 Table IX

    b0   = np.full_like(T,0.699 * 0.625, dtype='float64')
    b1   = np.full_like(T,17.16 * 0.625, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,0.0 * 0.3125, dtype='float64')

    zNa = np.float_(1)
    zP2O7 = np.float_(-4)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zP2O7)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_P2O7_PM73(T):

    # Coefficients from PM73 Table IX

    b0   = np.full_like(T,0.977 * 0.625, dtype='float64')
    b1   = np.full_like(T,17.88 * 0.625, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.2418 * 0.3125, dtype='float64')

    zK = np.float_(1)
    zP2O7 = np.float_(-4)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zP2O7)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_FeCN6_PM73(T):

    # Coefficients from PM73 Table IX

    b0   = np.full_like(T,1.021 * 0.625, dtype='float64')
    b1   = np.full_like(T,16.23 * 0.625, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.5579 * 0.3125, dtype='float64')

    zK = np.float_(1)
    zFeCN6 = np.float_(-4)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zFeCN6)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_MoCN8_PM73(T):

    # Coefficients from PM73 Table IX

    b0   = np.full_like(T,0.854 * 0.625, dtype='float64')
    b1   = np.full_like(T,18.53 * 0.625, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.3499 * 0.3125, dtype='float64')

    zK = np.float_(1)
    zMoCN8 = np.float_(-4)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zMoCN8)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_WCN8_PM73(T):

    # Coefficients from PM73 Table IX

    b0   = np.full_like(T,1.032 * 0.625, dtype='float64')
    b1   = np.full_like(T,18.49 * 0.625, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.4937 * 0.3125, dtype='float64')

    zK = np.float_(1)
    zWCN8 = np.float_(-4)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zWCN8)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_MeN_MoCN8_PM73(T):

    # Coefficients from PM73 Table IX

    b0   = np.full_like(T,0.938 * 0.625, dtype='float64')
    b1   = np.full_like(T,15.91 * 0.625, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.333 * 0.3125, dtype='float64')

    zMeN = np.float_(1)
    zMoCN8 = np.float_(-4)
    C0  = Cphi / (2 * np.sqrt(np.abs(zMeN*zMoCN8)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_Na_P3O10_PM73(T):

    # Coefficients from PM73 Table IX

    b0   = np.full_like(T,1.869 * 0.6, dtype='float64')
    b1   = np.full_like(T,36.1 * 0.6, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.163 * 0.26832815729997483, dtype='float64')

    zNa = np.float_(1)
    zP3O10 = np.float_(-5)
    C0  = Cphi / (2 * np.sqrt(np.abs(zNa*zP3O10)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: ----------------------------- 

def bC_K_P3O10_PM73(T):

    # Coefficients from PM73 Table IX

    b0   = np.full_like(T,1.939 * 0.6, dtype='float64')
    b1   = np.full_like(T,39.64 * 0.6, dtype='float64')
    b2   = np.zeros_like(T)
    Cphi = np.full_like(T,-0.1055 * 0.26832815729997483, dtype='float64')

    zK = np.float_(1)
    zP3O10 = np.float_(-5)
    C0  = Cphi / (2 * np.sqrt(np.abs(zK*zP3O10)))
    C1   = np.zeros_like(T)

    alph1 = np.float_(2)
    alph2 = -9
    omega = -9

    valid = T == 298.15

    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

