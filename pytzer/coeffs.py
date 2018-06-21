import autograd.numpy as np

###############################################################################
# === MOLLER 1988 =============================================================

def M88_eq13(T,a):
    
    return a[0]              \
         + a[1]  * T         \
         + a[2]  / T         \
         + a[3]  * np.log(T) \
         + a[4]  / (T-263.)  \
         + a[5]  * T**2      \
         + a[6]  / (680.-T)  \
         + a[7]  / (T-227.)

# --- Debye-Hueckel slope -----------------------------------------------------

def Aosm_M88(T):
    
    Aosm  = M88_eq13(T,
                     np.float_([ 3.36901532e-1,
                                -6.32100430e-4,
                                 9.14252359e00,
                                -1.35143986e-2,
                                 2.26089488e-3,
                                 1.92118597e-6,
                                 4.52586464e+1,
                                 0            ]))
    
    valid = np.logical_and(T >= 273.15, T <= 573.15)
    
    return Aosm, valid

# --- bC: calcium chloride ----------------------------------------------------

def CaCl_M88(T):
    
    b0    = M88_eq13(T,
                     np.float_([-9.41895832e+1,
                                -4.04750026e-2,
                                 2.34550368e+3,
                                 1.70912300e+1,
                                -9.22885841e-1,
                                 1.51488122e-5,
                                -1.39082000e00,
                                 0            ]))
    
    b1    = M88_eq13(T,
                     np.float_([ 3.47870000e00,
                                -1.54170000e-2,
                                 0            ,
                                 0            ,
                                 0            ,
                                 3.17910000e-5,
                                 0            ,
                                 0            ]))
    
    b2    = np.zeros_like(T)
    
    Cphi  = M88_eq13(T,
                     np.float_([-3.03578731e+1,
                                -1.36264728e-2,
                                 7.64582238e+2,
                                 5.50458061e00,
                                -3.27377782e-1,
                                 5.69405869e-6,
                                -5.36231106e-1,
                                 0            ]))
    
    zCa   = np.float_(+2)
    zCl   = np.float_(-1)
    C0    = Cphi / (2 * np.sqrt(np.abs(zCa*zCl)))
    
    C1    = np.zeros_like(T)
    
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    
    valid = np.logical_and(T >= 298.15, T <= 523.15)
    
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: calcium sulfate -----------------------------------------------------

def CaSO4_M88(T):
    
    b0    = np.full_like(T,0.15, dtype='float64')
    
    b1    = np.full_like(T,3.00, dtype='float64')
    
    b2    = M88_eq13(T,
                     np.float_([-1.29399287e+2,
                                 4.00431027e-1,
                                 0            ,
                                 0            ,
                                 0            ,
                                 0            ,
                                 0            ,
                                 0            ]))
    
    C0    = np.zeros_like(T)
    
    C1    = np.zeros_like(T)
    
    alph1 = np.float_(1.4)
    alph2 = np.float_(12)
    omega = -9
    
    valid = np.logical_and(T >= 298.15, T <= 523.15)
    
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: sodium chloride -----------------------------------------------------

def NaCl_M88(T):
    
    b0    = M88_eq13(T,
                     np.float_([ 1.43783204e+1,
                                 5.60767406e-3,
                                -4.22185236e+2,
                                -2.51226677e00,
                                 0            ,
                                -2.61718135e-6,
                                 4.43854508e00,
                                -1.70502337e00]))
    
    b1    = M88_eq13(T,
                     np.float_([-4.83060685e-1,
                                 1.40677479e-3,
                                 1.19311989e+2,
                                 0            ,
                                 0            ,
                                 0            ,
                                 0            ,
                                -4.23433299e00]))
    
    b2    = np.zeros_like(T)
    
    Cphi  = M88_eq13(T,
                     np.float_([-1.00588714e-1,
                                -1.80529413e-5,
                                 8.61185543e00,
                                 1.24880954e-2,
                                 0            ,
                                 3.41172108e-8,
                                 6.83040995e-2,
                                 2.93922611e-1]))
    
    zNa   = np.float_(+1)
    zCl   = np.float_(-1)
    C0    = Cphi / (2 * np.sqrt(np.abs(zNa*zCl)))
    
    C1    = np.zeros_like(T)
    
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    
    valid = np.logical_and(T >= 273.15, T <= 573.15)
    
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- bC: sodium sulfate ------------------------------------------------------

def NaSO4_M88(T):
    
    b0    = M88_eq13(T,
                     np.float_([ 8.16920027e+1,
                                 3.01104957e-2,
                                -2.32193726e+3,
                                -1.43780207e+1,
                                -6.66496111e-1,
                                -1.03923656e-5,
                                 0            ,
                                 0            ]))
    
    b1    = M88_eq13(T,
                     np.float_([ 1.00463018e+3,
                                 5.77453682e-1,
                                -2.18434467e+4,
                                -1.89110656e+2,
                                -2.03550548e-1,
                                -3.23949532e-4,
                                 1.46772243e+3,
                                 0            ]))
    
    b2    = np.zeros_like(T)
    
    Cphi  = M88_eq13(T,
                     np.float_([-8.07816886e+1,
                                -3.54521126e-2,
                                 2.02438830e+3,
                                 1.46197730e+1,
                                -9.16974740e-2,
                                 1.43946005e-5,
                                -2.42272049e00,
                                 0            ]))
    
    zNa   = np.float_(+1)
    zSO4  = np.float_(-2)
    C0    = Cphi / (2 * np.sqrt(np.abs(zNa*zSO4)))
    
    C1    = np.zeros_like(T)
    
    alph1 = np.float_(2)
    alph2 = -9
    omega = -9
    
    valid = np.logical_and(T >= 273.15, T <= 573.15)
    
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid

# --- theta: calcium sodium ---------------------------------------------------
    
def CaNa_M88(T):
    
    theta = np.full_like(T,0.05, dtype='float64')
    
    valid = np.logical_and(T >= 298.15, T <= 523.15)
    
    return theta, valid

# --- theta: chloride sulfate -------------------------------------------------
    
def ClSO4_M88(T):
    
    theta = np.full_like(T,0.07, dtype='float64')
    
    valid = np.logical_and(T >= 298.15, T <= 423.15)
    
    return theta, valid

# --- psi: calcium sodium chloride --------------------------------------------
    
def CaNaCl_M88(T):
    
    psi = np.full_like(T,-0.003, dtype='float64')
    
    valid = np.logical_and(T >= 298.15, T <= 523.15)
    
    return psi, valid

# --- psi: calcium sodium sulfate ---------------------------------------------
    
def CaNaSO4_M88(T):
    
    psi = np.full_like(T,-0.012, dtype='float64')
    
    valid = np.logical_and(T >= 298.15, T <= 523.15)
    
    return psi, valid

# --- psi: calcium chloride sulfate -------------------------------------------
    
def CaClSO4_M88(T):
    
    psi = np.full_like(T,-0.018, dtype='float64')
    
    valid = np.logical_and(T >= 298.15, T <= 523.15)
    
    return psi, valid

# --- psi: sodium chloride sulfate --------------------------------------------
    
def NaClSO4_M88(T):
    
    psi = np.full_like(T,-0.009, dtype='float64')
    
    valid = np.logical_and(T >= 298.15, T <= 423.15)
    
    return psi, valid

# --- dissociation: water -----------------------------------------------------
    
def Kw_M88(T):
    
    lnKw  = M88_eq13(T,
                     np.float_([ 1.04031130e+3,
                                 4.86092851e-1,
                                -3.26224352e+4,
                                -1.90877133e+2,
                                -5.35204850e-1,
                                -2.32009393e-4,
                                 5.20549183e+1,
                                 0            ]))
    
    valid = np.logical_and(T >= 298.15, T <= 523.15)
    
    return np.exp(lnKw), valid

# === MOLLER 1988 =============================================================
###############################################################################
