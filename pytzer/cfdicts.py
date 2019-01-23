# pytzer: the Pitzer model for chemical speciation
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)

from . import coeffs, jfuncs, props

#==============================================================================
#===================================== Define CoefficientDictionary class =====

class CoefficientDictionary:

    # Initialise
    def __init__(self):
        self.dh    = {}
        self.bC    = {}
        self.theta = {}
        self.jfunc = []
        self.psi   = {}

    # Populate with zero-functions
    def add_zeros(self,ions):

        # Get lists of cations and anions
        _,cats,anis = props.charges(ions)

        # Sort lists into alphabetical order
        cats.sort()
        anis.sort()

        # Populate cfdict with zero functions where no function exists
        for cat in cats:
            for ani in anis:

                istr = '-'.join((cat,ani))
                if istr not in self.bC.keys():
                    self.bC[istr] = coeffs.bC_zero

        for C0, cat0 in enumerate(cats):
            for cat1 in cats[C0+1:]:

                istr = '-'.join((cat0,cat1))
                if istr not in self.theta.keys():
                    self.theta[istr] = coeffs.theta_zero

                for ani in anis:

                    istr = '-'.join((cat0,cat1,ani))
                    if istr not in self.psi.keys():
                        self.psi[istr] = coeffs.psi_zero

        for A0, ani0 in enumerate(anis):
            for ani1 in anis[A0+1:]:

                istr = '-'.join((ani0,ani1))
                if istr not in self.theta.keys():
                    self.theta[istr] = coeffs.theta_zero

                for cat in cats:

                    istr = '-'.join((cat,ani0,ani1))
                    if istr not in self.psi.keys():
                        self.psi[istr] = coeffs.psi_zero


#==============================================================================
#=============================== Define specific coefficient dictionaries =====

#------------------------------------------------------------ Møller 1988 -----

# Møller (1988). Geochim. Cosmochim. Acta 52, 821-837,
#  doi:10.1016/0016-7037(88)90354-7
#
# System: Na-Ca-Cl-SO4

M88 = CoefficientDictionary()

# Debye-Hueckel limiting slope
M88.dh['Aosm'] = coeffs.Aosm_M88

# Cation-anion interactions (betas and Cs)
M88.bC['Ca-Cl' ] = coeffs.bC_Ca_Cl_M88
M88.bC['Ca-SO4'] = coeffs.bC_Ca_SO4_M88
M88.bC['Na-Cl' ] = coeffs.bC_Na_Cl_M88
M88.bC['Na-SO4'] = coeffs.bC_Na_SO4_M88

# Cation-cation and anion-anion interactions (theta)
# c-c'
M88.theta['Ca-Na' ] = coeffs.theta_Ca_Na_M88
# a-a'
M88.theta['Cl-SO4'] = coeffs.theta_Cl_SO4_M88

# Unsymmetrical mixing functions
M88.jfunc = jfuncs.Harvie

# Triplet interactions (psi)
# c-c'-a
M88.psi['Ca-Na-Cl' ] = coeffs.psi_Ca_Na_Cl_M88
M88.psi['Ca-Na-SO4'] = coeffs.psi_Ca_Na_SO4_M88
# c-a-a'
M88.psi['Ca-Cl-SO4'] = coeffs.psi_Ca_Cl_SO4_M88
M88.psi['Na-Cl-SO4'] = coeffs.psi_Na_Cl_SO4_M88


#------------------------------------------------ Greenberg & Møller 1989 -----

# Greenberg & Møller (1988). Geochim. Cosmochim. Acta 53, 2503-2518,
#  doi:10.1016/0016-7037(89)90124-5
#
# System: Na-K-Ca-Cl-SO4

GM89 = CoefficientDictionary()

# Debye-Hueckel limiting slope
GM89.dh['Aosm'] = coeffs.Aosm_M88

# Cation-anion interactions (betas and Cs)
GM89.bC['Ca-Cl' ] = coeffs.bC_Ca_Cl_GM89
GM89.bC['Ca-SO4'] = coeffs.bC_Ca_SO4_M88
GM89.bC['K-Cl'  ] = coeffs.bC_K_Cl_GM89
GM89.bC['K-SO4' ] = coeffs.bC_K_SO4_GM89
GM89.bC['Na-Cl' ] = coeffs.bC_Na_Cl_M88
GM89.bC['Na-SO4'] = coeffs.bC_Na_SO4_M88

# Cation-cation and anion-anion interactions (theta)
# c-c'
GM89.theta['Ca-K'  ] = coeffs.theta_Ca_K_GM89
GM89.theta['Ca-Na' ] = coeffs.theta_Ca_Na_M88
GM89.theta['K-Na'  ] = coeffs.theta_K_Na_GM89
# a-a'
GM89.theta['Cl-SO4'] = coeffs.theta_Cl_SO4_M88

# Unsymmetrical mixing terms
GM89.jfunc = jfuncs.Harvie

# Triplet interactions (psi)
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


#------------------------------------------------------ Clegg et al. 1994 -----

# Clegg et al. (1994). J. Chem. Soc., Faraday Trans. 90, 1875-1894,
#  doi:10.1039/FT9949001875
#
# System: H-HSO4-SO4

CRP94 = CoefficientDictionary()

# Debye-Hueckel limiting slope
CRP94.dh['Aosm'] = coeffs.Aosm_CRP94

# Cation-anion interactions (betas and Cs)
CRP94.bC['H-HSO4'] = coeffs.bC_H_HSO4_CRP94
CRP94.bC['H-SO4' ] = coeffs.bC_H_SO4_CRP94

# Cation-cation and anion-anion interactions (theta)
# a-a'
CRP94.theta['HSO4-SO4'] = coeffs.theta_HSO4_SO4_CRP94

# Unsymmetrical mixing terms
CRP94.jfunc = jfuncs.P75_eq47

# Triplet interactions (psi)
# c-a-a'
CRP94.psi['H-HSO4-SO4'] = coeffs.psi_H_HSO4_SO4_CRP94


#==============================================================================
