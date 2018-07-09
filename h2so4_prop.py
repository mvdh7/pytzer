from autograd import numpy as np
from autograd import jacobian as jac
import pytzer as pz

# Import coefficients
q = np.loadtxt('datasets/allmt_coeffs.res', skiprows=9)
q = q[:27,1]

# Import their covariance matrix
qmx = np.loadtxt('datasets/allmt_stats.res', skiprows=17)
qmx = qmx[:27,:27]

# Define test conditions
tot = np.vstack([1.6])
T   = np.vstack([323.15])
dissoc = np.float_(0.19322) # from CRP94 Table 10

# Evaluate species distribution
mSO4  = tot * dissoc
mHSO4 = tot - mSO4
mH    = tot + mSO4
mols = np.concatenate((mH,mHSO4,mSO4), axis=1)
#ions = np.array(['H','HSO4','SO4'])
#cf = pz.cdicts.CRP94

zH    = np.float_(+1)
zHSO4 = np.float_(-1)
zSO4  = np.float_(-2)

# Calculate ionic strength etc.
I = (mH*zH**2 + mHSO4*zHSO4**2 + mSO4*zSO4**2) / 2
Z = mH*np.abs(zH) + mHSO4*np.abs(zHSO4) + mSO4*np.abs(zHSO4)

# Set up function to evaluate new fit (ie. imported coefficients)
def CRP94new(T,q):
    
    # H-HSO4
    b0_H_HSO4 = pz.coeffs.CRP94_eq24(T,q[:4])
    b1_H_HSO4 = pz.coeffs.CRP94_eq24(T,np.array([q[4], 1.5, q[5], q[6]]))
    C0_H_HSO4 = pz.coeffs.CRP94_eq24(T,q[ 7:11]) / 2
    C1_H_HSO4 = pz.coeffs.CRP94_eq24(T,np.array([-0.025,
                                                 q[23],
                                                 q[24],
                                                 0.0025])) / 2
    alph1_H_HSO4 = np.float_(2)
    omega_H_HSO4 = np.float_(2.5)
    
    # H-SO4
    b0_H_SO4  = pz.coeffs.CRP94_eq24(T,q[11:15])
    b1_H_SO4  = pz.coeffs.CRP94_eq24(T,q[15:19])
    C0_H_SO4  = pz.coeffs.CRP94_eq24(T,q[19:23]) / (2 * np.sqrt(2))
    C1_H_SO4  = pz.coeffs.CRP94_eq24(T,np.array([0,
                                                 -0.176776695,
                                                 q[25],
                                                 0])) / (2 * np.sqrt(2))
    alph1_H_SO4 = 2 + 100 * q[26] * (1/T - 1/298.15)
    omega_H_SO4  = np.float_(2.5)
    
    return b0_H_HSO4,b1_H_HSO4,C0_H_HSO4,C1_H_HSO4,alph1_H_HSO4,omega_H_HSO4, \
           b0_H_SO4 ,b1_H_SO4 ,C0_H_SO4 ,C1_H_SO4 ,alph1_H_SO4 ,omega_H_SO4

# Evaluate coefficients with new fit
b0_H_HSO4,b1_H_HSO4,C0_H_HSO4,C1_H_HSO4,alph1_H_HSO4,omega_H_HSO4, \
    b0_H_SO4 ,b1_H_SO4 ,C0_H_SO4 ,C1_H_SO4 ,alph1_H_SO4 ,omega_H_SO4 \
    = CRP94new(T,q)

# _x = target values following CRP94, as a sanity check
b0_H_HSO4_x,b1_H_HSO4_x,_,C0_H_HSO4_x,C1_H_HSO4_x,alph1_H_HSO4_x,_, \
    omega_H_HSO4_x,_ = pz.coeffs.H_HSO4_CRP94(T)
b0_H_SO4_x,b1_H_SO4_x,_,C0_H_SO4_x,C1_H_SO4_x,alph1_H_SO4_x,_, \
    omega_H_SO4_x,_ = pz.coeffs.H_SO4_CRP94(T)
    
# Set up BC function for new fit
def fx_BC_H_HSO4(T,I,Z,q):
    
    b0_H_HSO4,b1_H_HSO4,C0_H_HSO4,C1_H_HSO4,alph1_H_HSO4,omega_H_HSO4, \
        _,_,_,_,_,_ = CRP94new(T,q)
        
    B = b0_H_HSO4 + b1_H_HSO4 * pz.model.g(alph1_H_HSO4 * np.sqrt(I))
    
    CT = C0_H_HSO4 + 4*C1_H_HSO4 * pz.model.h(omega_H_HSO4 * np.sqrt(I))
    
    return 2*B + Z*CT

fx_JBC_H_HSO4 = jac(fx_BC_H_HSO4, argnum=3)

# Calculate BC
BC_H_HSO4  = fx_BC_H_HSO4(T,I,Z,q)
JBC_H_HSO4 = fx_JBC_H_HSO4(T,I,Z,q).ravel()
    
# Directly estimate its uncertainty
UBC_H_HSO4_dir = JBC_H_HSO4 @ qmx @ JBC_H_HSO4.transpose()

# Monte-Carlo for BC uncertainty - CORRECT!
Ureps = int(1e2)
BC_H_HSO4_mc = np.full(Ureps,np.nan)

for i in range(Ureps):
    
    iq = np.random.multivariate_normal(q,qmx)
    BC_H_HSO4_mc[i] = fx_BC_H_HSO4(T,I,Z,iq)
    
UBC_H_HSO4_mc = np.var(BC_H_HSO4_mc)
    
# Covariance between b0 and b1 (for example)
Ureps = int(1e3)
b0 = np.full(Ureps,np.nan)
b1 = np.full(Ureps,np.nan)

for i in range(Ureps):
    iq = np.random.multivariate_normal(q,qmx)
    b0[i],b1[i],_,_,_,_,_ ,_ ,_ ,_ ,_ ,_ = CRP94new(T,iq)

CV_b0_b1_mc = np.cov(b0,b1)

fx_Jb0 = jac(lambda q: CRP94new(T,q)[0])
Jb0 = fx_Jb0(q).ravel()

fx_Jb1 = jac(lambda q: CRP94new(T,q)[1])
Jb1 = fx_Jb1(q).ravel()

#CV_b0_b1_dir = Jb0 @ qmx @ Jb1.transpose()
