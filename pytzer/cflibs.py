# pytzer: Pitzer model for chemical activities in aqueous solutions
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)

from . import coeffs, jfuncs, props
from .meta import version
from autograd.numpy import array, concatenate, unique
from copy import deepcopy

#==============================================================================
#================================================== Define CoeffLib class =====

class CoeffLib:


# ------------------------------------------------------------ Initialise -----

    def __init__(self):
        self.name  = ''

        self.dh    = {} # Aosm
        self.bC    = {} # c-a
        self.theta = {} # c-c' and a-a'
        self.jfunc = [] # unsymmetrical mixing
        self.psi   = {} # c-c'-a and c-a-a'
        self.lambd = {} # n-c and n-a
        self.zeta  = {} # n-c-a
        self.mu    = {} # n-n-n

        self.ions  = array([])
        self.srcs  = array([])

# ------------------------------------------ Populate with zero-functions -----

    def add_zeros(self,ions):

        # Get lists of cations and anions
        _,cations,anions,neutrals = props.charges(ions)

        # Sort lists into alphabetical order
        cations.sort()
        anions.sort()
        neutrals.sort()

        # Populate CoeffLib with zero functions where no function exists

        # betas and Cs
        for cation in cations:
            for anion in anions:

                istr = '-'.join((cation,anion))
                if istr not in self.bC.keys():
                    self.bC[istr] = coeffs.bC_none

        # c-c'-a thetas and psis
        for C0, cation0 in enumerate(cations):
            for cation1 in cations[C0+1:]:

                istr = '-'.join((cation0,cation1))
                if istr not in self.theta.keys():
                    self.theta[istr] = coeffs.theta_none

                for anion in anions:

                    istr = '-'.join((cation0,cation1,anion))
                    if istr not in self.psi.keys():
                        self.psi[istr] = coeffs.psi_none

        # c-a-a' thetas and psis
        for A0, anion0 in enumerate(anions):
            for anion1 in anions[A0+1:]:

                istr = '-'.join((anion0,anion1))
                if istr not in self.theta.keys():
                    self.theta[istr] = coeffs.theta_none

                for cation in cations:

                    istr = '-'.join((cation,anion0,anion1))
                    if istr not in self.psi.keys():
                        self.psi[istr] = coeffs.psi_none

        # Neutral interactions
        for N0, neutral0 in enumerate(neutrals):

            # n-c lambdas
            for cation in cations:
                inc = '-'.join((neutral0,cation))
                if inc not in self.lambd.keys():
                    self.lambd[inc] = coeffs.lambd_none

                # n-c-a zetas
                for anion in anions:
                    inca = '-'.join((neutral0,cation,anion))
                    if inca not in self.zeta.keys():
                        self.zeta[inca] = coeffs.zeta_none

            # n-a lambdas
            for anion in anions:
                ina = '-'.join((neutral0,anion))
                if ina not in self.lambd.keys():
                    self.lambd[ina] = coeffs.lambd_none

            # n-n' lambdas including n-n
            for neutral1 in neutrals[N0:]:
                inn = '-'.join((neutral0,neutral1))
                if inn not in self.lambd.keys():
                    self.lambd[inn] = coeffs.lambd_none

            # n-n-n mus
            innn = '-'.join((neutral0,neutral0,neutral0))
            if innn not in self.mu.keys():
                self.mu[innn] = coeffs.mu_none


# ------------------- Print all coefficient values at a given temperature -----

    def print_coeffs(self,T,filename):

        f = open(filename,'w')

        f.write('Coefficient library: {} [pytzer-v{}]\n\n'.format( \
                self.name,version))

        ionslist = 'Ions: ' + (len(self.ions)-1)*'{}, ' + '{}\n\n'
        f.write(ionslist.format(*self.ions))

#        srcslist = 'Sources: ' + (len(self.srcs)-1)*'{}, ' + '{}\n\n'
#        f.write(srcslist.format(*self.srcs))

        f.write('Temperature: {} K\n\n'.format(T))

        # Debye-Hueckel slope
        f.write('Debye-Hueckel limiting slope\n')
        f.write('============================\n')

        eval_Aosm = self.dh['Aosm'](array([T]))[0][0]

        src = self.dh['Aosm'].__name__.split('_')[-1]

        f.write('{:^12}  {:15}\n'.format('Aosm','source'))
        f.write('{:>12.9f}  {:15}\n'.format(eval_Aosm,src))

        # Write cation-anion coefficients (betas and Cs)
        f.write('\n')
        f.write('c-a pairs (betas and Cs)\n')
        f.write('========================\n')

        bChead = 2*'{:7}' + 5*'{:^13}'    + 3*'{:>6}'    + '  {:15}\n'
        bCvals = 2*'{:7}' + 5*'{:>13.5e}' + 3*'{:>6.1f}' + '  {:15}\n'
        f.write(bChead.format('cat','ani', 'b0','b1','b2','C0','C1',
                              'al1','al2','omg', 'source'))

        for bC in self.bC.keys():

            cation,anion = bC.split('-')
            b0,b1,b2,C0,C1, alph1,alph2,omega, _ = self.bC[bC](T)

            src = self.bC[bC].__name__.split('_')[-1]
            
            f.write(bCvals.format(cation,anion, b0,b1,b2,C0,C1,
                                  alph1,alph2,omega, src))

        # Write same charge ion-ion coefficients (thetas)
        f.write('\n')
        f.write('c-c\' and a-a\' pairs (thetas)\n')
        f.write('============================\n')

        thetaHead = 2*'{:7}' + '{:^13}'    + '  {:15}\n'
        thetaVals = 2*'{:7}' + '{:>13.5e}' + '  {:15}\n'

        f.write(thetaHead.format('ion1','ion2','theta','source'))

        for theta in self.theta.keys():

            ion0,ion1 = theta.split('-')
            eval_theta = self.theta[theta](T)[0]

            src = self.theta[theta].__name__.split('_')[-1]

            f.write(thetaVals.format(ion0,ion1, eval_theta, src))

        # Write ion triplet coefficients (psis)
        f.write('\n')
        f.write('c-c\'-a and c-a-a\' triplets (psis)\n')
        f.write('=================================\n')

        psiHead = 3*'{:7}' + '{:^12}'    + '  {:15}\n'
        psiVals = 3*'{:7}' + '{:>12.5e}' + '  {:15}\n'

        f.write(psiHead.format('ion1','ion2','ion3','psi','source'))

        for psi in self.psi.keys():

            ion0,ion1,ion2 = psi.split('-')
            eval_psi = self.psi[psi](T)[0]

            src = self.psi[psi].__name__.split('_')[-1]

            f.write(psiVals.format(ion0,ion1,ion2,eval_psi,src))

        # Write neutral-ion coefficients (lambdas)
        f.write('\n')
        f.write('n-c, n-a and n-n\' pairs (lambdas)\n')
        f.write('=================================\n')

        lambdHead = 2*'{:7}' + '{:^13}'    + '  {:15}\n'
        lambdVals = 2*'{:7}' + '{:>13.5e}' + '  {:15}\n'

        f.write(lambdHead.format('neut','ion','lambda','source'))

        for lambd in self.lambd.keys():

            neut,ion = lambd.split('-')
            eval_lambd = self.lambd[lambd](T)[0]

            src = self.lambd[lambd].__name__.split('_')[-1]

            f.write(lambdVals.format(neut,ion, eval_lambd, src))

        # Write neutral-cation-anion triplet coefficients (zetas)
        f.write('\n')
        f.write('n-c-a triplets (zetas)\n')
        f.write('======================\n')

        zetaHead = 3*'{:7}' + '{:^12}'    + '  {:15}\n'
        zetaVals = 3*'{:7}' + '{:>12.5e}' + '  {:15}\n'

        f.write(zetaHead.format('neut','cat','ani','zeta','source'))

        for zeta in self.zeta.keys():

            neut,cat,ani = zeta.split('-')
            eval_zeta = self.zeta[zeta](T)[0]

            src = self.zeta[zeta].__name__.split('_')[-1]

            f.write(zetaVals.format(neut,cat,ani,eval_zeta,src))

        # Write neutral-neutral-neutral triplet coefficients (mus)
        f.write('\n')
        f.write('n-n-n triplets (mus)\n')
        f.write('====================\n')

        muHead = 3*'{:7}' + '{:^12}'    + '  {:15}\n'
        muVals = 3*'{:7}' + '{:>12.5e}' + '  {:15}\n'

        f.write(muHead.format('neut1','neut2','neut3','mu','source'))

        for mu in self.mu.keys():

            neut1,neut2,neut3 = mu.split('-')
            eval_mu = self.mu[mu](T)[0]

            src = self.mu[mu].__name__.split('_')[-1]

            f.write(muVals.format(neut1,neut2,neut3,eval_mu,src))


# ------------------------------ Get all ions and sources in the CoeffLib -----

    def get_contents(self):

        # Get list of non-empty function dicts
        ctypes = [self.bC, self.theta, self.psi, self.lambd, 
                  self.zeta, self.mu]
        ctypes = [ctype for ctype in ctypes if any(ctype)]

        # Get unique list of "ions" (includes neutrals)
        self.ions = unique(concatenate([
            key.split('-') for ctype in ctypes for key in ctype.keys()]))

        # Get unique list of literature sources
        self.srcs = unique(concatenate([[ctype[key].__name__.split('_')[-1] \
             for key in ctype.keys()] for ctype in ctypes]))

        # Sort lists alphabetically
        self.ions.sort()
        self.srcs.sort()


#==============================================================================
#================================== Define specific coefficient libraries =====

#------------------------------------------------------------ Møller 1988 -----
#
# Møller (1988). Geochim. Cosmochim. Acta 52, 821-837,
#  doi:10.1016/0016-7037(88)90354-7
#
# System: Na-Ca-Cl-SO4

M88 = CoeffLib()
M88.name = 'M88'

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

M88.get_contents()

#------------------------------------------------ Greenberg & Møller 1989 -----
#
# Greenberg & Møller (1988). Geochim. Cosmochim. Acta 53, 2503-2518,
#  doi:10.1016/0016-7037(89)90124-5
#
# System: Na-K-Ca-Cl-SO4

GM89 = CoeffLib()
GM89.name = 'GM89'

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

GM89.get_contents()

#------------------------------------------------------ Clegg et al. 1994 -----
#
# Clegg et al. (1994). J. Chem. Soc., Faraday Trans. 90, 1875-1894,
#  doi:10.1039/FT9949001875
#
# System: H-HSO4-SO4

CRP94 = CoeffLib()
CRP94.name = 'CRP94'

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

CRP94.get_contents()

#-------------------------------------------------- Waters & Millero 2013 -----
#
# Waters and Millero (2013). Mar. Chem. 149, 8-22,
#  doi:10.1016/j.marchem.2012.11.003

WM13 = CoeffLib()
WM13.name = 'WM13'

# Debye-Hueckel limiting slope and unsymmetrical mixing
WM13.dh['Aosm'] = coeffs.Aosm_M88
WM13.jfunc = jfuncs.Harvie

# Table A1: Na salts
WM13.bC['Na-Cl'  ] = coeffs.bC_Na_Cl_M88
WM13.bC['Na-SO4' ] = coeffs.bC_Na_SO4_HM86
WM13.bC['Na-HSO4'] = coeffs.bC_Na_HSO4_HPR93
WM13.bC['Na-OH'  ] = coeffs.bC_Na_OH_PP87i

# Table A2: Mg salts
WM13.bC['Mg-Cl'  ] = coeffs.bC_Mg_Cl_dLP83
WM13.bC['Mg-SO4' ] = coeffs.bC_Mg_SO4_PP86ii
WM13.bC['Mg-HSO4'] = coeffs.bC_Mg_HSO4_RC99

# Table A3: Ca salts
WM13.bC['Ca-Cl'  ] = coeffs.bC_Ca_Cl_GM89
WM13.bC['Ca-SO4' ] = coeffs.bC_Ca_SO4_WM13
WM13.bC['Ca-HSO4'] = coeffs.bC_Ca_HSO4_WM13
WM13.bC['Ca-OH'  ] = coeffs.bC_Ca_OH_HMW84

# Table A4: K salts
WM13.bC['K-Cl'  ] = coeffs.bC_K_Cl_GM89
WM13.bC['K-SO4' ] = coeffs.bC_K_SO4_HM86
WM13.bC['K-HSO4'] = coeffs.bC_K_HSO4_WM13
WM13.bC['K-OH'  ] = coeffs.bC_K_OH_HMW84

# Table A5: H+ interactions
WM13.bC['H-Cl'  ] = coeffs.bC_H_Cl_CMR93
WM13.bC['H-SO4' ] = coeffs.bC_H_SO4_CRP94
WM13.bC['H-HSO4'] = coeffs.bC_H_HSO4_CRP94

# Table A6: MgOH+ interactions
WM13.bC['MgOH-Cl'] = coeffs.bC_MgOH_Cl_HMW84

# Table A7: cation-cation interactions
WM13.theta['H-Na' ] = coeffs.theta_H_Na_CMR93
WM13.theta['H-Mg' ] = coeffs.theta_H_Mg_RGB80
WM13.theta['Ca-H' ] = coeffs.theta_Ca_H_RGO82 # WM13 citation error
WM13.theta['H-K'  ] = coeffs.theta_H_K_CMR93
WM13.theta['Mg-Na'] = coeffs.theta_Mg_Na_HMW84
WM13.theta['Ca-Na'] = coeffs.theta_Ca_Na_HMW84
WM13.theta['K-Na' ] = coeffs.theta_K_Na_HMW84
WM13.theta['Ca-Mg'] = coeffs.theta_Ca_Mg_HMW84
WM13.theta['K-Mg' ] = coeffs.theta_K_Mg_HMW84
WM13.theta['Ca-K' ] = coeffs.theta_Ca_K_HMW84

# Table A7: anion-anion interactions
WM13.theta['Cl-SO4'  ] = coeffs.theta_Cl_SO4_HMW84
WM13.theta['Cl-HSO4' ] = coeffs.theta_Cl_HSO4_HMW84
WM13.theta['Cl-OH'   ] = coeffs.theta_Cl_OH_HMW84
WM13.theta['HSO4-SO4'] = coeffs.theta_HSO4_SO4_WM13
WM13.theta['OH-SO4'  ] = coeffs.theta_OH_SO4_HMW84

# Table A8: c-a-a' triplets
WM13.psi['H-Cl-SO4' ] = coeffs.psi_H_Cl_SO4_WM13 # agrees with HMW84
WM13.psi['Na-Cl-SO4'] = coeffs.psi_Na_Cl_SO4_HMW84
WM13.psi['Mg-Cl-SO4'] = coeffs.psi_Mg_Cl_SO4_HMW84
WM13.psi['Ca-Cl-SO4'] = coeffs.psi_Ca_Cl_SO4_HMW84
WM13.psi['K-Cl-SO4' ] = coeffs.psi_K_Cl_SO4_HMW84

WM13.psi['H-Cl-HSO4' ] = coeffs.psi_H_Cl_HSO4_HMW84
WM13.psi['Na-Cl-HSO4'] = coeffs.psi_Na_Cl_HSO4_HMW84
WM13.psi['Mg-Cl-HSO4'] = coeffs.psi_Mg_Cl_HSO4_HMW84
WM13.psi['Ca-Cl-HSO4'] = coeffs.psi_Ca_Cl_HSO4_HMW84
WM13.psi['K-Cl-HSO4' ] = coeffs.psi_K_Cl_HSO4_HMW84

WM13.psi['H-Cl-OH' ] = coeffs.psi_H_Cl_OH_WM13 # agrees with HMW84
WM13.psi['Na-Cl-OH'] = coeffs.psi_Na_Cl_OH_HMW84
WM13.psi['Mg-Cl-OH'] = coeffs.psi_Mg_Cl_OH_WM13 # agrees with HMW84
WM13.psi['Ca-Cl-OH'] = coeffs.psi_Ca_Cl_OH_HMW84
WM13.psi['K-Cl-OH' ] = coeffs.psi_K_Cl_OH_HMW84

WM13.psi['H-HSO4-SO4' ] = coeffs.psi_H_HSO4_SO4_HMW84
WM13.psi['Na-HSO4-SO4'] = coeffs.psi_Na_HSO4_SO4_HMW84
WM13.psi['Mg-HSO4-SO4'] = coeffs.psi_Mg_HSO4_SO4_RC99
WM13.psi['Ca-HSO4-SO4'] = coeffs.psi_Ca_HSO4_SO4_WM13 # agrees with HMW84
WM13.psi['K-HSO4-SO4' ] = coeffs.psi_K_HSO4_SO4_HMW84

WM13.psi['H-OH-SO4' ] = coeffs.psi_H_OH_SO4_WM13 # agrees with HMW84
WM13.psi['Na-OH-SO4'] = coeffs.psi_Na_OH_SO4_HMW84
WM13.psi['Mg-OH-SO4'] = coeffs.psi_Mg_OH_SO4_WM13 # agrees with HMW84
WM13.psi['Ca-OH-SO4'] = coeffs.psi_Ca_OH_SO4_WM13 # agrees with HMW84
WM13.psi['K-OH-SO4' ] = coeffs.psi_K_OH_SO4_HMW84

# Table A9: c-c'-a triplets
WM13.psi['H-Na-Cl'  ] = coeffs.psi_H_Na_Cl_HMW84
WM13.psi['H-Na-SO4' ] = coeffs.psi_H_Na_SO4_WM13 # agrees with HMW84
WM13.psi['H-Na-HSO4'] = coeffs.psi_H_Na_HSO4_HMW84

WM13.psi['H-Mg-Cl'] = coeffs.psi_H_Mg_Cl_HMW84
WM13.psi['H-Mg-SO4'] = coeffs.psi_H_Mg_SO4_RC99
WM13.psi['H-Mg-HSO4'] = coeffs.psi_H_Mg_HSO4_RC99

WM13.psi['Ca-H-Cl'  ] = coeffs.psi_Ca_H_Cl_HMW84
WM13.psi['Ca-H-SO4' ] = coeffs.psi_Ca_H_SO4_WM13 # agrees with HMW84
WM13.psi['Ca-H-HSO4'] = coeffs.psi_Ca_H_HSO4_WM13 # agrees with HMW84

WM13.psi['H-K-Cl'  ] = coeffs.psi_H_K_Cl_HMW84
WM13.psi['H-K-SO4' ] = coeffs.psi_H_K_SO4_HMW84
WM13.psi['H-K-HSO4'] = coeffs.psi_H_K_HSO4_HMW84

WM13.psi['Mg-Na-Cl'  ] = coeffs.psi_Mg_Na_Cl_HMW84
WM13.psi['Mg-Na-SO4' ] = coeffs.psi_Mg_Na_SO4_HMW84
WM13.psi['Mg-Na-HSO4'] = coeffs.psi_Mg_Na_HSO4_WM13 # agrees with HMW84

WM13.psi['Ca-Na-Cl'  ] = coeffs.psi_Ca_Na_Cl_HMW84
WM13.psi['Ca-Na-SO4' ] = coeffs.psi_Ca_Na_SO4_HMW84
WM13.psi['Ca-Na-HSO4'] = coeffs.psi_Ca_Na_HSO4_WM13 # agrees with HMW84

WM13.psi['K-Na-Cl'  ] = coeffs.psi_K_Na_Cl_HMW84
WM13.psi['K-Na-SO4' ] = coeffs.psi_K_Na_SO4_HMW84
WM13.psi['K-Na-HSO4'] = coeffs.psi_K_Na_HSO4_WM13 # agrees with HMW84

WM13.psi['Ca-Mg-Cl'  ] = coeffs.psi_Ca_Mg_Cl_HMW84
WM13.psi['Ca-Mg-SO4' ] = coeffs.psi_Ca_Mg_SO4_HMW84
WM13.psi['Ca-Mg-HSO4'] = coeffs.psi_Ca_Mg_HSO4_WM13 # agrees with HMW84

WM13.psi['K-Mg-Cl'  ] = coeffs.psi_K_Mg_Cl_HMW84
WM13.psi['K-Mg-SO4' ] = coeffs.psi_K_Mg_SO4_HMW84
WM13.psi['K-Mg-HSO4'] = coeffs.psi_K_Mg_HSO4_WM13 # agrees with HMW84

WM13.psi['Ca-K-Cl'  ] = coeffs.psi_Ca_K_Cl_HMW84
WM13.psi['Ca-K-SO4' ] = coeffs.psi_Ca_K_SO4_WM13 # agrees with HMW84
WM13.psi['Ca-K-HSO4'] = coeffs.psi_Ca_K_HSO4_WM13 # agrees with HMW84

WM13.get_contents()


#----------------------------------------------------- WM13_MarChemSpec25 -----

WM13_MarChemSpec25 = deepcopy(WM13)
WM13_MarChemSpec25.name = 'WM13_MarChemSpec25'

WM13_MarChemSpec25.dh['Aosm'] = coeffs.Aosm_MarChemSpec25
WM13_MarChemSpec25.jfunc = jfuncs.P75_eq47

WM13_MarChemSpec25.theta['H-Na'] = coeffs.theta_H_Na_MarChemSpec25
WM13_MarChemSpec25.theta['H-K' ] = coeffs.theta_H_K_MarChemSpec25
WM13_MarChemSpec25.theta['Ca-H'] = coeffs.theta_Ca_H_MarChemSpec

WM13_MarChemSpec25.psi['Mg-MgOH-Cl'] = coeffs.psi_Mg_MgOH_Cl_HMW84

WM13_MarChemSpec25.get_contents()


#------------------------------------------------------------ MarChemSpec -----

# Begin with WM13_MarChemSpec25
MarChemSpec25 = deepcopy(WM13_MarChemSpec25)
MarChemSpec25.name = 'MarChemSpec25'

# Add coefficients from GT17 Supp. Info. Table S6 (simultaneous optimisation)
#MarChemSpec25.bC['Na-Cl'    ] = coeffs.bC_Na_Cl_GT17simopt
MarChemSpec25.bC['trisH-SO4'] = coeffs.bC_trisH_SO4_GT17simopt
MarChemSpec25.bC['trisH-Cl' ] = coeffs.bC_trisH_Cl_GT17simopt

MarChemSpec25.theta['H-trisH'] = coeffs.theta_H_trisH_GT17simopt

MarChemSpec25.psi['H-trisH-Cl'] = coeffs.psi_H_trisH_Cl_GT17simopt

MarChemSpec25.lambd['tris-trisH'] = coeffs.lambd_tris_trisH_GT17simopt
MarChemSpec25.lambd['tris-Na'   ] = coeffs.lambd_tris_Na_GT17simopt
MarChemSpec25.lambd['tris-K'    ] = coeffs.lambd_tris_K_GT17simopt
MarChemSpec25.lambd['tris-Mg'   ] = coeffs.lambd_tris_Mg_GT17simopt
MarChemSpec25.lambd['tris-Ca'   ] = coeffs.lambd_tris_Ca_GT17simopt
MarChemSpec25.lambd['tris-tris' ] = coeffs.lambd_tris_tris_MarChemSpec25

MarChemSpec25.zeta['tris-Na-Cl'] = coeffs.zeta_tris_Na_Cl_MarChemSpec25

MarChemSpec25.mu['tris-tris-tris'] = coeffs.mu_tris_tris_tris_MarChemSpec25

MarChemSpec25.add_zeros(array(['H','Na','Mg','Ca','K','MgOH','trisH','Cl',
                               'SO4','HSO4','OH','tris']))
MarChemSpec25.get_contents()


# Begin with WM13_MarChemSpec25, switch to constant 5 degC Aosm
MarChemSpec05 = deepcopy(MarChemSpec25)
MarChemSpec05.name = 'MarChemSpec05'

MarChemSpec05.dh['Aosm'] = coeffs.Aosm_MarChemSpec05

# Begin with WM13_MarChemSpec25, switch to CRP94 corrected Aosm
MarChemSpec = deepcopy(MarChemSpec25)
MarChemSpec.name = 'MarChemSpec'

MarChemSpec.dh['Aosm'] = coeffs.Aosm_MarChemSpec

#--------------------------------------- Millero & Pierrot 1998 aka MIAMI -----
#
#~~~~~~~~~~~~~~~~~~~~~~~ WORK IN PROGRESS !!!!! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MIAMI = CoeffLib()
MIAMI.name = 'MIAMI'

MIAMI.dh['Aosm'] = coeffs.Aosm_M88
MIAMI.jfunc = jfuncs.Harvie

# Table A1
MIAMI.bC['Na-Cl' ] = coeffs.bC_Na_Cl_M88
MIAMI.bC['K-Cl'  ] = coeffs.bC_K_Cl_GM89
MIAMI.bC['K-SO4' ] = coeffs.bC_K_SO4_GM89
MIAMI.bC['Ca-Cl' ] = coeffs.bC_Ca_Cl_GM89
MIAMI.bC['Ca-SO4'] = coeffs.bC_Ca_SO4_M88
MIAMI.bC['Ca-SO3'] = coeffs.bC_Ca_SO3_MP98
MIAMI.bC['Sr-SO4'] = coeffs.bC_Sr_SO4_MP98

# Table A2
MIAMI.bC['Mg-Cl' ] = coeffs.bC_Mg_Cl_PP87i
MIAMI.bC['Mg-SO4'] = coeffs.bC_Mg_SO4_PP86ii

# Table A3
MIAMI.bC['Na-HSO4'] = coeffs.bC_Na_HSO4_MP98
MIAMI.bC['Na-HCO3'] = coeffs.bC_Na_HCO3_PP82
MIAMI.bC['Na-SO4' ] = coeffs.bC_Na_SO4_HPR93
MIAMI.bC['Na-CO3' ] = coeffs.bC_Na_CO3_PP82
MIAMI.bC['Na-BOH4'] = coeffs.bC_Na_BOH4_SRRJ87
MIAMI.bC['Na-HS'  ] = coeffs.bC_Na_HS_HPM88
#MIAMI.bC['Na-SCN' ] = coeffs.bC_Na_SCN_SP78
#MIAMI.bC['Na-SO3' ] = coeffs.bC_Na_SO3_M89
#MIAMI.bC['Na-HSO3'] = coeffs.bC_Na_HSO3_M89

# Table A4
#MIAMI.bC['K-HCO3 '] = coeffs.bC_K_HCO3_R83
#MIAMI.bC['K-CO3'  ] = coeffs.bC_K_CO3_S87a
MIAMI.bC['K-BOH4' ] = coeffs.bC_K_BOH4_SRRJ87
MIAMI.bC['K-HS'   ] = coeffs.bC_K_HS_HPM88
MIAMI.bC['K-H2PO4'] = coeffs.bC_K_H2PO4_SP78
MIAMI.bC['K-SCN'  ] = coeffs.bC_K_SCN_SP78

# Table A5
#MIAMI.bC['Mg-Br'  ] = coeffs.bC_Mg_Br_SP78
MIAMI.bC['Mg-BOH4'] = coeffs.bC_Mg_BOH4_SRM87
#MIAMI.bC['Mg-ClO4'] = coeffs.bC_Mg_ClO4_SP78
#MIAMI.bC['Ca-Br'  ] = coeffs.bC_Ca_Br_SP78
MIAMI.bC['Ca-BOH4'] = coeffs.bC_Ca_BOH4_SRM87
#MIAMI.bC['Ca-ClO4'] = coeffs.bC_Ca_ClO4_SP78

# Table A6
MIAMI.bC['Sr-Br'  ] = coeffs.bC_Sr_Br_SP78
MIAMI.bC['Sr-Cl'  ] = coeffs.bC_Sr_Cl_SP78 # not in table but in text §4.6
#MIAMI.bC['Sr-NO3' ] = coeffs.bC_Sr_NO3_SP78
#MIAMI.bC['Sr-ClO4'] = coeffs.bC_Sr_ClO4_SP78
#MIAMI.bC['Sr-HSO3'] = coeffs.bC_Sr_HSO3_SP78
MIAMI.bC['Sr-BOH4'] = coeffs.bC_Sr_BOH4_MP98

# Table A7
MIAMI.bC['Na-I'   ] = coeffs.bC_Na_I_MP98
MIAMI.bC['Na-Br'  ] = coeffs.bC_Na_Br_MP98
MIAMI.bC['Na-F'   ] = coeffs.bC_Na_F_MP98
MIAMI.bC['K-Br'   ] = coeffs.bC_K_Br_MP98
MIAMI.bC['K-F'    ] = coeffs.bC_K_F_MP98
MIAMI.bC['K-OH'   ] = coeffs.bC_K_OH_MP98
MIAMI.bC['K-I'    ] = coeffs.bC_K_I_MP98
#MIAMI.bC['Na-ClO3'] = coeffs.bC_Na_ClO3_MP98
#MIAMI.bC['K-ClO3' ] = coeffs.bC_K_ClO3_MP98
#MIAMI.bC['Na-ClO4'] = coeffs.bC_Na_ClO4_MP98
#MIAMI.bC['Na-BrO3'] = coeffs.bC_Na_BrO3_MP98
#MIAMI.bC['K-BrO3' ] = coeffs.bC_K_BrO3_MP98
#MIAMI.bC['Na-NO3' ] = coeffs.bC_Na_NO3_MP98
#MIAMI.bC['K-NO3'  ] = coeffs.bC_K_NO3_MP98
#MIAMI.bC['Mg-NO3' ] = coeffs.bC_Mg_NO3_MP98
#MIAMI.bC['Ca-NO3' ] = coeffs.bC_Ca_NO3_MP98
#MIAMI.bC['H-Br'   ] = coeffs.bC_H_Br_MP98
#MIAMI.bC['Sr-Cl'  ] = coeffs.bC_Sr_Cl_MP98
#MIAMI.bC['NH4-Cl' ] = coeffs.bC_NH4_Cl_MP98
#MIAMI.bC['NH4-Br' ] = coeffs.bC_NH4_Br_MP98
#MIAMI.bC['NH4-F'  ] = coeffs.bC_NH4_F_MP98

# Table A8
MIAMI.bC.update({iset: lambda T: coeffs.bC_PM73(T,iset) \
    for iset in ['Sr-I', 'Na-NO2', 'Na-H2PO4', 'Na-HPO4', 'Na-PO4',
                 'K-NO2', 'K-HPO4', 'K-PO4', 'Mg-I', 'Ca-I', 'SO4-NH4']})
#MIAMI.bC['Na-H2AsO4' ] = coeffs.bC_Na_H2AsO4_PM73
#MIAMI.bC['K-HAsO4'   ] = coeffs.bC_K_HAsO4_PM73
#MIAMI.bC['Na-HAsO4'  ] = coeffs.bC_Na_HAsO4_PM73
#MIAMI.bC['Na-AsO4'   ] = coeffs.bC_Na_AsO4_PM73
#MIAMI.bC['Na-acetate'] = coeffs.bC_Na_acetate_PM73
MIAMI.bC['K-HSO4' ] = coeffs.bC_K_HSO4_HMW84
#MIAMI.bC['K-AsO4'    ] = coeffs.bC_K_AsO4_PM73
#MIAMI.bC['K-acetate' ] = coeffs.bC_K_acetate_PM73
MIAMI.bC['Mg-HS'  ] = coeffs.bC_Mg_HS_HPM88
MIAMI.bC['Ca-HSO4'] = coeffs.bC_Ca_HSO4_HMW84
MIAMI.bC['Ca-HCO3'] = coeffs.bC_Ca_HCO3_HMW84
MIAMI.bC['Ca-HS'  ] = coeffs.bC_Ca_HS_HPM88
MIAMI.bC['Ca-OH'  ] = coeffs.bC_Ca_OH_HMW84
MIAMI.bC['MgOH-Cl'] = coeffs.bC_MgOH_Cl_HMW84

# Table A9
MIAMI.bC['H-Cl' ] = coeffs.bC_H_Cl_CMR93
#MIAMI.bC['H-SO4'] = coeffs.bC_H_SO4_Pierrot

# Table A10
MIAMI.theta['Cl-CO3' ] = coeffs.theta_Cl_CO3_PP82
MIAMI.theta['Cl-HCO3'] = coeffs.theta_Cl_HCO3_PP82

# Get contents
MIAMI.get_contents()

#==============================================================================