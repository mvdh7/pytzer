# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
from autograd.numpy import array, concatenate, unique
from .. import parameters as prm
from .. import properties
from ..meta import version

class ParameterLibrary:
    def __init__(self, module=None, name='', dh={}, bC={}, theta={}, jfunc=[],
            psi={}, lambd={}, zeta={}, mu={}, lnk={}, ions=array([]),
            srcs=array([])):
        self.name = name
        self.dh = dh # Aosm
        self.bC = bC # c-a
        self.theta = theta # c-c' and a-a'
        self.jfunc = jfunc # unsymmetrical mixing
        self.psi = psi # c-c'-a and c-a-a'
        self.lambd = lambd # # n-c, n-a, n-n and n-n'
        self.zeta = zeta # n-c-a
        self.mu = mu # n-n-n
        self.lnk = lnk # thermodynamic equilibrium constants
        self.ions = ions
        self.srcs = srcs
        if module is not None:
            if 'name' in dir(module):
                self.name = module.name
            if 'dh' in dir(module):
                self.dh = module.dh
            if 'bC' in dir(module):
                self.bC = module.bC
            if 'theta' in dir(module):
                self.theta = module.theta
            if 'jfunc' in dir(module):
                self.jfunc = module.jfunc
            if 'psi' in dir(module):
                self.psi = module.psi
            if 'lambd' in dir(module):
                self.lambd = module.lambd
            if 'zeta' in dir(module):
                self.zeta = module.zeta
            if 'mu' in dir(module):
                self.mu = module.mu
            if 'lnk' in dir(module):
                self.lnk = module.lnk
            if 'ions' in dir(module):
                self.ions = module.ions
            if 'srcs' in dir(module):
                self.srcs = module.srcs

    def add_zeros(self, ions):
        """Add zero-functions for missing combinations of solutes."""
        _, cations, anions, neutrals = properties.charges(ions)
        cations.sort()
        anions.sort()
        neutrals.sort()
        # betas and Cs
        for cation in cations:
            for anion in anions:
                istr = '-'.join((cation, anion))
                if istr not in self.bC.keys():
                    self.bC[istr] = prm.bC_none
        # c-c'-a thetas and psis
        for C0, cation0 in enumerate(cations):
            for cation1 in cations[C0+1:]:
                istr = '-'.join((cation0, cation1))
                if istr not in self.theta.keys():
                    self.theta[istr] = prm.theta_none
                for anion in anions:
                    istr = '-'.join((cation0, cation1, anion))
                    if istr not in self.psi.keys():
                        self.psi[istr] = prm.psi_none
        # c-a-a' thetas and psis
        for A0, anion0 in enumerate(anions):
            for anion1 in anions[A0+1:]:
                istr = '-'.join((anion0, anion1))
                if istr not in self.theta.keys():
                    self.theta[istr] = prm.theta_none
                for cation in cations:
                    istr = '-'.join((cation, anion0, anion1))
                    if istr not in self.psi.keys():
                        self.psi[istr] = prm.psi_none
        # Neutral interactions
        for N0, neutral0 in enumerate(neutrals):
            # n-c lambdas
            for cation in cations:
                inc = '-'.join((neutral0, cation))
                if inc not in self.lambd.keys():
                    self.lambd[inc] = prm.lambd_none
                # n-c-a zetas
                for anion in anions:
                    inca = '-'.join((neutral0, cation, anion))
                    if inca not in self.zeta.keys():
                        self.zeta[inca] = prm.zeta_none
            # n-a lambdas
            for anion in anions:
                ina = '-'.join((neutral0, anion))
                if ina not in self.lambd.keys():
                    self.lambd[ina] = prm.lambd_none
            # n-n' lambdas including n-n
            for neutral1 in neutrals[N0:]:
                inn = '-'.join((neutral0, neutral1))
                if inn not in self.lambd.keys():
                    self.lambd[inn] = prm.lambd_none
            # n-n-n mus
            innn = '-'.join((neutral0, neutral0, neutral0))
            if innn not in self.mu.keys():
                self.mu[innn] = prm.mu_none

    def print_parameters(self, T, P, filename):
        """Print all parameter values at a given temperature and pressure
        to a text file.
        """
        f = open(filename, 'w')
        f.write('Parameter library: {} [pytzer-v{}]\n\n'.format(
            self.name, version))
        ionslist = 'Ions: ' + (len(self.ions)-1)*'{}, ' + '{}\n\n'
        f.write(ionslist.format(*self.ions))
#        srcslist = 'Sources: ' + (len(self.srcs)-1)*'{}, ' + '{}\n\n'
#        f.write(srcslist.format(*self.srcs))
        f.write('Temperature: {} K\n'.format(T))
        f.write('   Pressure: {} dbar\n\n'.format(P))
        # Debye-Hueckel slope
        f.write('Debye-Hueckel limiting slope\n')
        f.write('============================\n')
        eval_Aosm = self.dh['Aosm'](T, P)[0]
        src = self.dh['Aosm'].__name__.split('_')[-1]
        f.write('{:^12}  {:15}\n'.format('Aosm','source'))
        f.write('{:>12.9f}  {:15}\n'.format(eval_Aosm, src))
        # Write cation-anion parameters (betas and Cs)
        f.write('\n')
        f.write('c-a pairs (betas and Cs)\n')
        f.write('========================\n')
        bChead = 2*'{:7}' + 5*'{:^13}'    + 3*'{:>6}'    + '  {:15}\n'
        bCvals = 2*'{:7}' + 5*'{:>13.5e}' + 3*'{:>6.1f}' + '  {:15}\n'
        f.write(bChead.format('cat', 'ani', 'b0', 'b1', 'b2', 'C0', 'C1',
            'al1', 'al2', 'omg', 'source'))
        for bC in self.bC.keys():
            cation,anion = bC.split('-')
            b0, b1, b2, C0, C1, alph1, alph2, omega, _ = self.bC[bC](T, P)
            src = self.bC[bC].__name__.split('_')[-1]
            f.write(bCvals.format(cation, anion, b0, b1, b2, C0, C1,
                alph1, alph2, omega, src))
        # Write same charge ion-ion parameters (thetas)
        f.write('\n')
        f.write('c-c\' and a-a\' pairs (thetas)\n')
        f.write('============================\n')
        thetaHead = 2*'{:7}' + '{:^13}'    + '  {:15}\n'
        thetaVals = 2*'{:7}' + '{:>13.5e}' + '  {:15}\n'
        f.write(thetaHead.format('ion1','ion2','theta','source'))
        for theta in self.theta.keys():
            ion0, ion1 = theta.split('-')
            eval_theta = self.theta[theta](T, P)[0]
            src = self.theta[theta].__name__.split('_')[-1]
            f.write(thetaVals.format(ion0, ion1, eval_theta, src))
        # Write ion triplet parameters (psis)
        f.write('\n')
        f.write('c-c\'-a and c-a-a\' triplets (psis)\n')
        f.write('=================================\n')
        psiHead = 3*'{:7}' + '{:^12}'    + '  {:15}\n'
        psiVals = 3*'{:7}' + '{:>12.5e}' + '  {:15}\n'
        f.write(psiHead.format('ion1', 'ion2', 'ion3', 'psi', 'source'))
        for psi in self.psi.keys():
            ion0, ion1, ion2 = psi.split('-')
            eval_psi = self.psi[psi](T, P)[0]
            src = self.psi[psi].__name__.split('_')[-1]
            f.write(psiVals.format(ion0, ion1, ion2, eval_psi, src))
        # Write neutral-ion parameters (lambdas)
        f.write('\n')
        f.write('n-c, n-a and n-n\' pairs (lambdas)\n')
        f.write('=================================\n')
        lambdHead = 2*'{:7}' + '{:^13}'    + '  {:15}\n'
        lambdVals = 2*'{:7}' + '{:>13.5e}' + '  {:15}\n'
        f.write(lambdHead.format('neut', 'ion', 'lambda', 'source'))
        for lambd in self.lambd.keys():
            neut, ion = lambd.split('-')
            eval_lambd = self.lambd[lambd](T, P)[0]
            src = self.lambd[lambd].__name__.split('_')[-1]
            f.write(lambdVals.format(neut, ion, eval_lambd, src))
        # Write neutral-cation-anion triplet parameters (zetas)
        f.write('\n')
        f.write('n-c-a triplets (zetas)\n')
        f.write('======================\n')
        zetaHead = 3*'{:7}' + '{:^12}'    + '  {:15}\n'
        zetaVals = 3*'{:7}' + '{:>12.5e}' + '  {:15}\n'
        f.write(zetaHead.format('neut', 'cat', 'ani', 'zeta', 'source'))
        for zeta in self.zeta.keys():
            neut,cat,ani = zeta.split('-')
            eval_zeta = self.zeta[zeta](T, P)[0]
            src = self.zeta[zeta].__name__.split('_')[-1]
            f.write(zetaVals.format(neut, cat, ani, eval_zeta, src))
        # Write neutral-neutral-neutral triplet parameters (mus)
        f.write('\n')
        f.write('n-n-n triplets (mus)\n')
        f.write('====================\n')
        muHead = 3*'{:7}' + '{:^12}'    + '  {:15}\n'
        muVals = 3*'{:7}' + '{:>12.5e}' + '  {:15}\n'
        f.write(muHead.format('neut1','neut2','neut3','mu','source'))
        for mu in self.mu.keys():
            neut1, neut2, neut3 = mu.split('-')
            eval_mu = self.mu[mu](T, P)[0]
            src = self.mu[mu].__name__.split('_')[-1]
            f.write(muVals.format(neut1, neut2, neut3, eval_mu, src))

    def get_contents(self):
        """Get all ions and sources in the parameter library."""
        ctypes = [self.bC, self.theta, self.psi, self.lambd,
            self.zeta, self.mu]
        ctypes = [ctype for ctype in ctypes if any(ctype)]
        self.ions = unique(concatenate([
            key.split('-') for ctype in ctypes for key in ctype.keys()]))
        self.srcs = unique(concatenate([[ctype[key].__name__.split('_')[-1] \
             for key in ctype.keys()] for ctype in ctypes]))
        self.ions.sort()
        self.srcs.sort()
