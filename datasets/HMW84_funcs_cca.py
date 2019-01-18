def theta_K_Na_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,-0.012, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_K_Na_Cl_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.0018, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_Na_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.01, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_Na_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_Na_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_Na_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.003, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_Na_CO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.003, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_Ca_Na_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,0.07, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_Ca_Na_Cl_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.007, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_Na_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.055, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_Na_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_Na_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_Na_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_Na_CO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_Mg_Na_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,0.07, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_Mg_Na_Cl_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.012, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_Na_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.015, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_Na_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_Na_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_Na_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_Na_CO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_MgOH_Na_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_MgOH_Na_Cl_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_MgOH_Na_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_MgOH_Na_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_MgOH_Na_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_MgOH_Na_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_MgOH_Na_CO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_H_Na_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,0.036, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_H_Na_Cl_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.004, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_Na_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_Na_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.0129, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_Na_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_Na_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_Na_CO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_Ca_K_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,0.032, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_Ca_K_Cl_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.025, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_K_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_K_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_K_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_K_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_K_CO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_K_Mg_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_K_Mg_Cl_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.022, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_Mg_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.048, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_Mg_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_Mg_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_Mg_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_Mg_CO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_K_MgOH_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_K_MgOH_Cl_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_MgOH_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_MgOH_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_MgOH_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_MgOH_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_K_MgOH_CO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_H_K_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,0.005, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_H_K_Cl_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.011, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_K_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.197, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_K_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.0265, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_K_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_K_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_K_CO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_Ca_Mg_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,0.007, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_Ca_Mg_Cl_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.012, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_Mg_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.024, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_Mg_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_Mg_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_Mg_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_Mg_CO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_Ca_MgOH_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_Ca_MgOH_Cl_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_MgOH_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_MgOH_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_MgOH_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_MgOH_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_MgOH_CO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_Ca_H_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,0.092, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_Ca_H_Cl_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.015, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_H_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_H_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_H_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_H_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Ca_H_CO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_Mg_MgOH_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_Mg_MgOH_Cl_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.028, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_MgOH_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_MgOH_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_MgOH_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_MgOH_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_Mg_MgOH_CO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_H_Mg_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,0.1, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_H_Mg_Cl_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.011, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_Mg_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_Mg_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,-0.0178, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_Mg_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_Mg_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_Mg_CO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def theta_H_MgOH_HMW84(T):
# Coefficients from HMW84 Table 2
    theta = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return theta, valid

def psi_H_MgOH_Cl_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_MgOH_SO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_MgOH_HSO4_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_MgOH_OH_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_MgOH_HCO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

def psi_H_MgOH_CO3_HMW84(T):
# Coefficients from HMW84 Table 2
    psi = np.full_like(T,0.0, dtype='float64')
    valid = T == 298.15
    return psi, valid

