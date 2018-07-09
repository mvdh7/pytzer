from autograd import numpy as np
from autograd import jacobian as jac
from autograd import hessian as hess
from autograd import elementwise_grad as egrad
import pandas as pd
import pytzer as pz
#from matplotlib import pyplot as plt
import pickle

# Import coefficients
q = np.loadtxt('datasets/allmt_coeffs.res', skiprows=9)
q = q[:27,1]

# Import their covariance matrix
qmx = np.loadtxt('datasets/allmt_stats.res', skiprows=17)
qmx = qmx[:27,:27]

## Define test conditions
#tot = np.vstack([1.6])
#T   = np.vstack([323.15])
#dissoc = np.float_(0.19322) # from CRP94 Table 10
#
## Evaluate species distribution
#mSO4  = tot * dissoc
#mHSO4 = tot - mSO4
#mH    = tot + mSO4
#mols = np.concatenate((mH,mHSO4,mSO4), axis=1)

zH    = np.float_(+1)
zHSO4 = np.float_(-1)
zSO4  = np.float_(-2)

## Calculate ionic strength etc.
#I = (mH*zH**2 + mHSO4*zHSO4**2 + mSO4*zSO4**2) / 2
#Z = mH*np.abs(zH) + mHSO4*np.abs(zHSO4) + mSO4*np.abs(zHSO4)

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

## Evaluate coefficients with new fit
#b0_H_HSO4,b1_H_HSO4,C0_H_HSO4,C1_H_HSO4,alph1_H_HSO4,omega_H_HSO4, \
#    b0_H_SO4 ,b1_H_SO4 ,C0_H_SO4 ,C1_H_SO4 ,alph1_H_SO4 ,omega_H_SO4 \
#    = CRP94new(T,q)
#
## _x = target values following CRP94, as a sanity check
#b0_H_HSO4_x,b1_H_HSO4_x,_,C0_H_HSO4_x,C1_H_HSO4_x,alph1_H_HSO4_x,_, \
#    omega_H_HSO4_x,_ = pz.coeffs.H_HSO4_CRP94(T)
#b0_H_SO4_x,b1_H_SO4_x,_,C0_H_SO4_x,C1_H_SO4_x,alph1_H_SO4_x,_, \
#    omega_H_SO4_x,_ = pz.coeffs.H_SO4_CRP94(T)
#    
## Set up BC function for new fit
#def fx_BC_H_HSO4(T,I,Z,q):
#    
#    b0_H_HSO4,b1_H_HSO4,C0_H_HSO4,C1_H_HSO4,alph1_H_HSO4,omega_H_HSO4, \
#        _,_,_,_,_,_ = CRP94new(T,q)
#        
#    B = b0_H_HSO4 + b1_H_HSO4 * pz.model.g(alph1_H_HSO4 * np.sqrt(I))
#    
#    CT = C0_H_HSO4 + 4*C1_H_HSO4 * pz.model.h(omega_H_HSO4 * np.sqrt(I))
#    
#    return 2*B + Z*CT
#
#fx_JBC_H_HSO4 = jac(fx_BC_H_HSO4, argnum=3)
#
#def fx_BC_H_SO4(T,I,Z,q):
#    
#    _,_,_,_,_,_,b0_H_SO4,b1_H_SO4,C0_H_SO4,C1_H_SO4,alph1_H_SO4,omega_H_SO4 \
#        = CRP94new(T,q)
#        
#    B = b0_H_SO4 + b1_H_SO4 * pz.model.g(alph1_H_SO4 * np.sqrt(I))
#    
#    CT = C0_H_SO4 + 4*C1_H_SO4 * pz.model.h(omega_H_SO4 * np.sqrt(I))
#    
#    return 2*B + Z*CT
#
#def fx_BC_both(T,I,Z,q,mH,mHSO4,mSO4):
#    
#    return mH*mHSO4 * fx_BC_H_HSO4(T,I,Z,q) \
#         + mH*mSO4  * fx_BC_H_SO4 (T,I,Z,q) \
#
#fx_JBC_both = jac(fx_BC_both, argnum=3)
#
## Calculate BC
#T = np.vstack([298.15,323.15])
#I = np.vstack([I,I])
#Z = np.vstack([Z,Z])
#mH    = np.vstack([mH,mH])
#mHSO4 = np.vstack([mHSO4,mHSO4])
#mSO4  = np.vstack([mSO4,mSO4])
#
#BC_H_HSO4  = fx_BC_H_HSO4 (T,I,Z,q)
#JBC_H_HSO4 = np.squeeze(fx_JBC_H_HSO4(T,I,Z,q))
#    
## Directly estimate its uncertainty
#UBC_H_HSO4_dir = JBC_H_HSO4 @ qmx @ JBC_H_HSO4.transpose()
#
#BC_both = fx_BC_both(T,I,Z,q,mH,mHSO4,mSO4)
#JBC_both = np.squeeze(fx_JBC_both(T,I,Z,q,mH,mHSO4,mSO4))
#
#UBC_both_dir = JBC_both @ qmx @ JBC_both.transpose()

##### Plot results using dissocs from CRP94 tables 8-10 #######################

# Import CRP94 tables
crp94 = pd.read_excel('datasets/CRP94 Tables 8-10.xlsx')

# Evaluate speciation
crp94['mSO4' ] = crp94.tot * crp94.dissoc
crp94['mHSO4'] = crp94.tot - crp94.mSO4
crp94['mH'   ] = crp94.tot + crp94.mSO4

mols = np.vstack((crp94.mH,crp94.mHSO4,crp94.mSO4)).transpose()
T = np.vstack(crp94.temp.values)
ions = np.array(['H','HSO4','SO4'])
cf = pz.cdicts.CRP94

Gex = pz.model.Gex_nRT(mols,ions,T,cf)
acfs = pz.model.acfs(mols,ions,T,cf)

crp94['acfPM_pz'] = np.cbrt((acfs[:,0]*mols[:,0])**2 * acfs[:,2]*mols[:,2] \
    / (4 * crp94.tot.values**3))

# _x = target values following CRP94, as a sanity check
b0_H_HSO4_x,b1_H_HSO4_x,_,C0_H_HSO4_x,C1_H_HSO4_x,alph1_H_HSO4_x,_, \
    omega_H_HSO4_x,_ = pz.coeffs.H_HSO4_CRP94(T)
b0_H_SO4_x,b1_H_SO4_x,_,C0_H_SO4_x,C1_H_SO4_x,alph1_H_SO4_x,_, \
    omega_H_SO4_x,_ = pz.coeffs.H_SO4_CRP94(T)

# Re-evaluate using pz.fitting functions
Gex2 = pz.fitting.Gex_MXY(mols,zH,zHSO4,zSO4,T,
    b0_H_HSO4_x,b1_H_HSO4_x,0,C0_H_HSO4_x,C1_H_HSO4_x,
    alph1_H_HSO4_x,-9,omega_H_HSO4_x,
    b0_H_SO4_x,b1_H_SO4_x,0,C0_H_SO4_x,C1_H_SO4_x,
    alph1_H_SO4_x,-9,omega_H_SO4_x)

acfs2 = np.exp(pz.fitting.ln_acfs_MXY(mols,zH,zHSO4,zSO4,T,
    b0_H_HSO4_x,b1_H_HSO4_x,0,C0_H_HSO4_x,C1_H_HSO4_x,
    alph1_H_HSO4_x,-9,omega_H_HSO4_x,
    b0_H_SO4_x,b1_H_SO4_x,0,C0_H_SO4_x,C1_H_SO4_x,
    alph1_H_SO4_x,-9,omega_H_SO4_x))

# Build acfs function with q input
def qacfPM(T,q,tot,mols):
    
    b0_H_HSO4,b1_H_HSO4,C0_H_HSO4,C1_H_HSO4,alph1_H_HSO4,omega_H_HSO4, \
        b0_H_SO4 ,b1_H_SO4 ,C0_H_SO4 ,C1_H_SO4 ,alph1_H_SO4 ,omega_H_SO4 \
        = CRP94new(T,q)
        
    acfs = np.exp(pz.fitting.ln_acfs_MXY(mols,zH,zHSO4,zSO4,T,
        b0_H_HSO4,b1_H_HSO4,0,C0_H_HSO4,C1_H_HSO4,
        alph1_H_HSO4,-9,omega_H_HSO4,
        b0_H_SO4,b1_H_SO4,0,C0_H_SO4,C1_H_SO4,
        alph1_H_SO4,-9,omega_H_SO4))
    
    acfPM = ((acfs[:,0]*mols[:,0])**2 * acfs[:,2]*mols[:,2] \
        / (4 * tot**3))**(1/3)
    
    return acfPM
   
crp94['acfPM_new'] = qacfPM(T,q,crp94.tot.values,mols)
crp94['lnacfPM_new'] = np.log(crp94.acfPM_new)

# Get Jacobian & propagate uncertainties from qmx into acfPM
tot = crp94.tot.values
fx_JqacfPM = jac(qacfPM, argnum=1)
fx_JlnqacfPM = jac(lambda T,q,tot,mols:np.log(qacfPM(T,q,tot,mols)), argnum=1)
qtest = egrad(qacfPM, argnum=1)
JqacfPM = fx_JqacfPM(T,q,tot,mols)
JlnqacfPM = fx_JlnqacfPM(T,q,tot,mols)

# estimate Hessian... but don't know what to do with it after! See:
# https://en.wikipedia.org/wiki/
#     Taylor_expansions_for_the_moments_of_functions_of_random_variables
HlnqacfPM = JlnqacfPM.transpose() @ JlnqacfPM

crp94['acfPM_unc'] = np.diagonal(JqacfPM @ qmx @ JqacfPM.transpose())
crp94['lnacfPM_unc'] = np.diagonal(JlnqacfPM @ qmx @ JlnqacfPM.transpose())

# Monte-Carlo propagation
Ureps = int(1e2)
UacfPM   = np.full((np.size(T),Ureps),np.nan)
UlnacfPM = np.full((np.size(T),Ureps),np.nan)

for i in range(Ureps):
    
    iq = np.random.multivariate_normal(q,qmx)
    UacfPM  [:,i] = qacfPM(T,iq,tot,mols)
    UlnacfPM[:,i] = np.log(UacfPM[:,i])

UacfPM_var   = np.var(UacfPM  , axis=1)
UlnacfPM_var = np.var(UlnacfPM, axis=1)

## Pickle results for plotting
#with open('pickles/h2so4_prop.pkl','wb') as f:
#    pickle.dump((crp94,UlnacfPM,UlnacfPM_var,UacfPM,UacfPM_var),f)

## Visualise results
#fig,ax = plt.subplots(1,1)
#
#crp94[crp94.temp == 273.15].plot('tot','lnacfPM_unc', ax=ax, c='b',
#     label='273.15 K')
#crp94[crp94.temp == 298.15].plot('tot','lnacfPM_unc', ax=ax, c='g',
#     label='298.15 K')
#crp94[crp94.temp == 323.15].plot('tot','lnacfPM_unc', ax=ax, c='orange',
#     label='323.15 K')
#
#ax.scatter(crp94.tot,UlnacfPM_var, c='k', alpha=0.5)
#
#ax.grid(alpha=0.5)

## Monte-Carlo for BC uncertainty - CORRECT!
#Ureps = int(1e3)
#BC_H_HSO4_mc = np.full(Ureps,np.nan)
#
#for i in range(Ureps):
#    
#    iq = np.random.multivariate_normal(q,qmx)
#    BC_H_HSO4_mc[i] = fx_BC_H_HSO4(T,I,Z,iq)
#    
#UBC_H_HSO4_mc = np.var(BC_H_HSO4_mc)
    
## Covariance between b0 and b1 (for example) - CORRECT!
#Ureps = int(1e3)
#b0 = np.full(Ureps,np.nan)
#b1 = np.full(Ureps,np.nan)
#
#for i in range(Ureps):
#    iq = np.random.multivariate_normal(q,qmx)
#    b0[i],b1[i],_,_,_,_,_ ,_ ,_ ,_ ,_ ,_ = CRP94new(T,iq)
#
#CV_b0_b1_mc = np.cov(b0,b1)
#
#fx_Jb0 = jac(lambda q: CRP94new(T,q)[0])
#Jb0 = fx_Jb0(q).ravel()
#
#fx_Jb1 = jac(lambda q: CRP94new(T,q)[1])
#Jb1 = fx_Jb1(q).ravel()
#
#JJ = np.vstack((Jb0,Jb1))
#
#CV_b0_b1_dir = JJ @ qmx @ JJ.transpose()
