def theta_Cl_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,0.02, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_Na_Cl_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0014, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_Cl_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_Cl_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.018, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_Cl_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.004, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_MgOH_Cl_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_Cl_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_Cl_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,-0.006, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_Na_Cl_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.006, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_Cl_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_Cl_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_Cl_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_MgOH_Cl_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_Cl_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.013, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_Cl_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,-0.05, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_Na_Cl_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.006, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_Cl_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.006, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_Cl_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.025, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_Cl_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_MgOH_Cl_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_Cl_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_Cl_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,0.03, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_Na_Cl_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.15, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_Cl_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_Cl_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_Cl_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.096, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_MgOH_Cl_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_Cl_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_CO3_Cl_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,-0.02, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_Na_CO3_Cl_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0085, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_CO3_Cl_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.004, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_CO3_Cl_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_CO3_Cl_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_MgOH_CO3_Cl_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_CO3_Cl_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_HSO4_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_Na_HSO4_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.0094, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_HSO4_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.0677, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_HSO4_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_HSO4_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.0425, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_MgOH_HSO4_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_HSO4_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_OH_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,-0.013, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_Na_OH_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.009, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_OH_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.05, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_OH_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_OH_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_MgOH_OH_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_OH_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_HCO3_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,0.01, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_Na_HCO3_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.005, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_HCO3_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_HCO3_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_HCO3_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.161, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_MgOH_HCO3_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_HCO3_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_CO3_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,0.02, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_Na_CO3_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.005, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_CO3_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.009, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_CO3_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_CO3_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_MgOH_CO3_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_CO3_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_HSO4_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_Na_HSO4_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_HSO4_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_HSO4_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_HSO4_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_MgOH_HSO4_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_HSO4_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_HCO3_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_Na_HCO3_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_HCO3_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_HCO3_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_HCO3_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_MgOH_HCO3_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_HCO3_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_CO3_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_Na_CO3_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_CO3_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,nan, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_CO3_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_CO3_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_MgOH_CO3_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_CO3_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_HCO3_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_Na_HCO3_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_HCO3_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_HCO3_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_HCO3_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_MgOH_HCO3_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_HCO3_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_CO3_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,0.1, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_Na_CO3_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.017, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_CO3_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.01, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_CO3_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_CO3_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_MgOH_CO3_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_CO3_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_CO3_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,-0.04, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_Na_CO3_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.002, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_CO3_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.012, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_CO3_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_CO3_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_MgOH_CO3_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_CO3_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

