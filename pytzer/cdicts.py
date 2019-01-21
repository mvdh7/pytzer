from . import coeffs, jfuncs, props

# Define cdict class
class cdict:
    def __init__(self):
        self.dh    = {}
        self.bC    = {}
        self.theta = {}
        self.jfunc = []
        self.psi   = {}
        
    
    # Populate with zero-functions
    def add_zeros(self,ions):
        
        _,cats,anis = props.charges(ions)

        # Populate cdict with zero functions
        for cat in cats:
            for ani in anis:
                
                istr = cat + '-' + ani
                if istr not in self.bC.keys():
                    self.bC[istr] = coeffs.bC_zero
                
        for C0 in range(len(cats)):
            for C1 in range(C0+1,len(cats)):
                
                istr = cats[C0] + '-' + cats[C1]
                if istr not in self.theta.keys():
                    self.theta[istr] = coeffs.theta_zero
                
                for ani in anis:
                    
                    istr = cats[C0] + '-' + cats[C1] + '-' + ani
                    if istr not in self.psi.keys():
                        self.psi[istr] = coeffs.psi_zero
        
        for A0 in range(len(anis)):
            for A1 in range(A0+1,len(anis)):
                
                istr = anis[A0] + '-' + anis[A1]
                if istr not in self.theta.keys():
                    self.theta[istr] = coeffs.theta_zero
                
                for cat in cats:
                    
                    istr = cat + '-' + anis[A0] + '-' + anis[A1]
                    if istr not in self.psi.keys():
                        self.psi[istr] = coeffs.psi_zero
        
        
# === MOLLER 1988 =============================================================

M88 = cdict()

# Debye-Hueckel slope as cf.dh['Aosm']
M88.dh['Aosm'] = coeffs.Aosm_M88

# betas and Cs as cf.bC['cation-anion']
M88.bC['Ca-Cl' ] = coeffs.bC_Ca_Cl_M88
M88.bC['Ca-SO4'] = coeffs.bC_Ca_SO4_M88
M88.bC['Na-Cl' ] = coeffs.bC_Na_Cl_M88
M88.bC['Na-SO4'] = coeffs.bC_Na_SO4_M88

# thetas as cf.theta['cation1-cation2'] with cations in alph. order
# c-c'
M88.theta['Ca-Na' ] = coeffs.theta_Ca_Na_M88
# a-a'
M88.theta['Cl-SO4'] = coeffs.theta_Cl_SO4_M88

# Unsymmetrical mixing terms
M88.jfunc = jfuncs.P75_eq47

# psis as cf.psi['cation1-cation2-anion'] with cations in alph. order
#   or as cf.psi['cation-anion1-anion2']  with anions  in alph. order
# c-c'-a
M88.psi['Ca-Na-Cl' ] = coeffs.psi_Ca_Na_Cl_M88
M88.psi['Ca-Na-SO4'] = coeffs.psi_Ca_Na_SO4_M88
# c-a-a'
M88.psi['Ca-Cl-SO4'] = coeffs.psi_Ca_Cl_SO4_M88
M88.psi['Na-Cl-SO4'] = coeffs.psi_Na_Cl_SO4_M88


# === GREENBERG & MOLLER 1989 =================================================

GM89 = cdict()

# Debye-Hueckel slope
GM89.dh['Aosm'] = coeffs.Aosm_M88

# betas and Cs as cf.bC['cation-anion']
GM89.bC['Ca-Cl' ] = coeffs.bC_Ca_Cl_GM89
GM89.bC['Ca-SO4'] = coeffs.bC_Ca_SO4_M88
GM89.bC['K-Cl'  ] = coeffs.bC_K_Cl_GM89
GM89.bC['K-SO4' ] = coeffs.bC_K_SO4_GM89
GM89.bC['Na-Cl' ] = coeffs.bC_Na_Cl_M88
GM89.bC['Na-SO4'] = coeffs.bC_Na_SO4_M88

# thetas as cf.theta['cation1-cation2'] with cations in alph. order
# c-c'
GM89.theta['Ca-K'  ] = coeffs.theta_Ca_K_GM89
GM89.theta['Ca-Na' ] = coeffs.theta_Ca_Na_M88
GM89.theta['K-Na'  ] = coeffs.theta_K_Na_GM89
# a-a'
GM89.theta['Cl-SO4'] = coeffs.theta_Cl_SO4_M88

# Unsymmetrical mixing terms
GM89.jfunc = jfuncs.P75_eq47

# psis as cf.psi['cation1-cation2-anion'] with cations in alph. order
#   or as cf.psi['cation-anion1-anion2']  with anions  in alph. order
# c-c'-a
GM89.psi['Ca-K-Cl'  ] = coeffs.psi_Ca_K_Cl_GM89
GM89.psi['Ca-K-SO4' ] = coeffs.psi_Ca_K_SO4_GM89
GM89.psi['Ca-Na-Cl' ] = coeffs.psi_Ca_Na_Cl_M88
GM89.psi['Ca-Na-SO4'] = coeffs.psi_Ca_Na_SO4_M88
GM89.psi['K-Na-Cl'  ] = coeffs.psi_K_Na_Cl_GM89
GM89.psi['K-Na-SO4' ] = coeffs.psi_K_Na_SO4_GM89
# c-a-a'
GM89.psi['Ca-Cl-SO4'] = coeffs.psi_Ca_Cl_SO4_M88
GM89.psi['K-Cl-SO4' ] = coeffs.psi_K_Cl_SO4_GM89
GM89.psi['Na-Cl-SO4'] = coeffs.psi_Na_Cl_SO4_M88


# === CLEGG ET AL 1994=========================================================

CRP94 = cdict()

CRP94.dh['Aosm'] = coeffs.Aosm_CRP94

CRP94.bC['H-HSO4'] = coeffs.bC_H_HSO4_CRP94
CRP94.bC['H-SO4' ] = coeffs.bC_H_SO4_CRP94

CRP94.theta['HSO4-SO4'] = coeffs.theta_HSO4_SO4_CRP94

CRP94.jfunc = jfuncs.P75_eq47

CRP94.psi['H-HSO4-SO4'] = coeffs.psi_H_HSO4_SO4_CRP94

