import autograd.numpy as np
from . import coeffs, jfuncs

class cdict:
    def __init__(self):
        self.dh    = {}
        self.bC    = {'H-OH': coeffs.zero_bC}
        self.theta = {}
        self.jfunc = []
        self.psi   = {}
        self.K     = {}
        
    def getKeq(self,T, mH=None, gH=None, mOH=None, gOH=None, 
               mHSO4=None, gHSO4=None, mSO4=None, gSO4=None):
        
        # Find which equilibria are stored in dict
        Klist = self.K.keys()
        
        # Initialise Keq equation
        Keq = np.zeros_like(T, dtype='float64')
        
        if 'H2O' in Klist:
            Keq = Keq + np.log(gH*mH.ravel() * gOH*mOH.ravel()) \
                - np.log(self.K['H2O'](T)[0])
        
        if 'HSO4' in Klist:
            Keq = Keq + np.log(gH*mH.ravel() * gSO4*mSO4.ravel() \
                / (gHSO4*mHSO4.ravel())) \
                - np.log(self.K['HSO4'](T)[0])
            
        return Keq
        

# === MOLLER 1988 =============================================================
M88 = cdict()

# Debye-Hueckel slope as cf.dh['Aosm']
M88.dh['Aosm'] = coeffs.Aosm_M88

# betas and Cs as cf.bC['cation-anion']
M88.bC['Ca-Cl' ] = coeffs.Ca_Cl_M88
M88.bC['Ca-SO4'] = coeffs.Ca_SO4_M88
M88.bC['Na-Cl' ] = coeffs.Na_Cl_M88
M88.bC['Na-SO4'] = coeffs.Na_SO4_M88

# thetas as cf.theta['cation1-cation2'] with cations in alph. order
# c-c'
M88.theta['Ca-Na' ] = coeffs.Ca_Na_M88
# a-a'
M88.theta['Cl-SO4'] = coeffs.Cl_SO4_M88

# Unsymmetrical mixing terms
M88.jfunc = jfuncs.P75_eq47

# psis as cf.psi['cation1-cation2-anion'] with cations in alph. order
#   or as cf.psi['cation-anion1-anion2']  with anions  in alph. order
# c-c'-a
M88.psi['Ca-Na-Cl' ] = coeffs.Ca_Na_Cl_M88
M88.psi['Ca-Na-SO4'] = coeffs.Ca_Na_SO4_M88
# c-a-a'
M88.psi['Ca-Cl-SO4'] = coeffs.Ca_Cl_SO4_M88
M88.psi['Na-Cl-SO4'] = coeffs.Na_Cl_SO4_M88

# Dissociation constants as cf.K['acid']
M88.K['H2O'] = coeffs.Kw_M88

# === GREENBERG & MOLLER 1989 =================================================
GM89 = cdict()

# Debye-Hueckel slope
GM89.dh['Aosm'] = coeffs.Aosm_M88

# betas and Cs as cf.bC['cation-anion']
GM89.bC['Ca-Cl' ] = coeffs.Ca_Cl_GM89
GM89.bC['Ca-SO4'] = coeffs.Ca_SO4_M88
GM89.bC['K-Cl'  ] = coeffs.K_Cl_GM89
GM89.bC['K-SO4' ] = coeffs.K_SO4_GM89
GM89.bC['Na-Cl' ] = coeffs.Na_Cl_M88
GM89.bC['Na-SO4'] = coeffs.Na_SO4_M88

# thetas as cf.theta['cation1-cation2'] with cations in alph. order
# c-c'
GM89.theta['Ca-K'  ] = coeffs.Ca_K_GM89
GM89.theta['Ca-Na' ] = coeffs.Ca_Na_M88
GM89.theta['K-Na'  ] = coeffs.K_Na_GM89
# a-a'
GM89.theta['Cl-SO4'] = coeffs.Cl_SO4_M88

# Unsymmetrical mixing terms
GM89.jfunc = jfuncs.P75_eq47

# psis as cf.psi['cation1-cation2-anion'] with cations in alph. order
#   or as cf.psi['cation-anion1-anion2']  with anions  in alph. order
# c-c'-a
GM89.psi['Ca-K-Cl'  ] = coeffs.Ca_K_Cl_GM89
GM89.psi['Ca-K-SO4' ] = coeffs.Ca_K_SO4_GM89
GM89.psi['Ca-Na-Cl' ] = coeffs.Ca_Na_Cl_M88
GM89.psi['Ca-Na-SO4'] = coeffs.Ca_Na_SO4_M88
GM89.psi['K-Na-Cl'  ] = coeffs.K_Na_Cl_GM89
GM89.psi['K-Na-SO4' ] = coeffs.K_Na_SO4_GM89
# c-a-a'
GM89.psi['Ca-Cl-SO4'] = coeffs.Ca_Cl_SO4_M88
GM89.psi['K-Cl-SO4' ] = coeffs.K_Cl_SO4_GM89
GM89.psi['Na-Cl-SO4'] = coeffs.Na_Cl_SO4_M88

# Dissociation constants as cf.K['acid']
GM89.K['H2O'] = coeffs.Kw_M88

# === CLEGG ET AL 1994======= =================================================
CRP94 = cdict()

CRP94.dh['Aosm'] = coeffs.Aosm_CRP94

CRP94.bC['H-HSO4'] = coeffs.H_HSO4_CRP94
CRP94.bC['H-SO4' ] = coeffs.H_SO4_CRP94

CRP94.theta['HSO4-SO4'] = coeffs.HSO4_SO4_CRP94

CRP94.jfunc = jfuncs.P75_eq47

CRP94.psi['H-HSO4-SO4'] = coeffs.H_HSO4_SO4_CRP94

CRP94.K['HSO4'] = coeffs.KHSO4_CRP94
