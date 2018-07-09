from autograd import numpy as np
import pytzer as pz

# Import coefficients
q = np.loadtxt('datasets/allmt_coeffs.res', skiprows=9)
q = q[:27,1]

# Import their covariance matrix
qmx = np.loadtxt('datasets/allmt_stats.res', skiprows=17)
qmx = qmx[:27,:27]

# Define test conditions
tot = np.vstack([1.5])
T   = np.vstack([308.15])

# Evaluate coefficients using new fit (ie. imported coefficients)
b0_H_HSO4 = pz.coeffs.CRP94_eq24(T,q[:4])
b1_H_HSO4 = pz.coeffs.CRP94_eq24(T,np.array([q[4], 1.5, q[5], q[6]]))
C0_H_HSO4 = pz.coeffs.CRP94_eq24(T,q[ 7:11]) / 2
b0_H_SO4  = pz.coeffs.CRP94_eq24(T,q[11:15])
b1_H_SO4  = pz.coeffs.CRP94_eq24(T,q[15:19])
C0_H_SO4  = pz.coeffs.CRP94_eq24(T,q[19:23]) / (2 * np.sqrt(2))
C1_H_HSO4 = pz.coeffs.CRP94_eq24(T,np.array([-0.025, q[23], q[24], 0.0025])) \
    / 2
C1_H_SO4  = pz.coeffs.CRP94_eq24(T,np.array([0, -0.176776695, q[25], 0])) \
    / (2 * np.sqrt(2))
alph1_H_SO4 = 2 + 100 * q[26] * (1/T - 1/298.15)
alph1_H_HSO4 = np.float_(2)
omega_H_SO4  = np.float_(2.5)
omega_H_HSO4 = np.float_(2.5)

# _x = target values following CRP94, as a sanity check
b0_H_HSO4_x,b1_H_HSO4_x,_,C0_H_HSO4_x,C1_H_HSO4_x,alph1_H_HSO4_x,_, \
    omega_H_HSO4_x,_ = pz.coeffs.H_HSO4_CRP94(T)
b0_H_SO4_x,b1_H_SO4_x,_,C0_H_SO4_x,C1_H_SO4_x,alph1_H_SO4_x,_, \
    omega_H_SO4_x,_ = pz.coeffs.H_SO4_CRP94(T)
    
#
    