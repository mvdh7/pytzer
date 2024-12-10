# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2024  M.P. Humphreys  (GNU GPLv3)
from jax import numpy as np
from .. import parameters
from ..convert import solute_to_charge

# Equations for reference
# =======================
#
# Buffer (Dickson) alkalinity:
# ----------------------------
#     return (
#         add_if_in("OH")
#         - add_if_in("H")
#         + add_if_in("MgOH")
#         - add_if_in("HF")
#         + add_if_in("HCO3")
#         + add_if_in("CO3") * 2
#         + add_if_in("HPO4")
#         + add_if_in("PO4") * 2
#         - add_if_in("H3PO4")
#         + add_if_in("MgCO3") * 2
#         + add_if_in("CaCO3") * 2
#         + add_if_in("SrCO3") * 2
#         + add_if_in("MgHPO4")
#         + add_if_in("MgPO4") * 2
#         + add_if_in("CaHPO4")
#         + add_if_in("CaPO4") * 2
#         - add_if_in("HSO4")
#         + add_if_in("HS")
#         + add_if_in("BOH4")
#         + add_if_in("NH3")
#         + add_if_in("H3SiO4")
#         - add_if_in("HNO2")
#         + add_if_in("tris")
#     )
#
# Explicit alkalinity:
# --------------------
#     return (
#         add_if_in("Na")
#         + add_if_in("K")
#         - add_if_in("Cl")
#         - add_if_in("Br")
#         + add_if_in("Mg") * 2
#         + add_if_in("Ca") * 2
#         + add_if_in("Sr") * 2
#         - add_if_in("F")
#         - add_if_in("PO4")
#         - add_if_in("SO4") * 2
#         + add_if_in("NH3")
#         - add_if_in("NO2")
#         + add_if_in("tris")
#     )


class Library:
    def __init__(self, name=""):
        self.name = name
        self.cations = tuple()
        self.anions = tuple()
        self.neutrals = tuple()
        self.charges = {}
        self.charges_cat = np.array([])
        self.charges_ani = np.array([])
        self.Aphi = None
        self.func_J = None
        self.ca = {}
        self.ca_combos = np.array([])
        self.get_ca_values = None
        self.cc = {}
        self.cc_combos = np.array([])
        self.get_cc_values = None
        self.cca = {}
        self.cca_combos = np.array([])
        self.get_cca_values = None
        self.aa = {}
        self.aa_combos = np.array([])
        self.get_aa_values = None
        self.caa = {}
        self.caa_combos = np.array([])
        self.get_caa_values = None
        self.nnn = {}
        self.nnn_combos = np.array([])
        self.get_nnn_values = None
        self.nn = {}
        self.nn_combos = np.array([])
        self.get_nn_values = None
        self.nc = {}
        self.nc_combos = np.array([])
        self.get_nc_values = None
        self.na = {}
        self.na_combos = np.array([])
        self.get_na_values = None
        self.nca = {}
        self.nca_combos = np.array([])
        self.get_nca_values = None
        self.solver_targets = tuple()
        self.totals_all = set()
        self.equilibria = {}
        self.equilibria_all = tuple()
        self.get_ks_constants = None
        self.totals_to_solutes = None
        self.get_alkalinity_explicit = None
        self.get_stoich_error = None
        self.get_stoich_targets = None
        self.get_stoich_error_jac = None
        self.stoich_init = np.array([])

    def get_solutes(self, sanity_check=True, **solutes):
        if sanity_check:
            for k, v in solutes.items():
                assert (
                    k in self.cations or k in self.anions or k in self.neutrals
                ), "Solute {} is not part of this Library!".format(k)
                assert v >= 0, "All solute molalities must be >= 0."
        self._expand_solutes(solutes)
        solutes = {k: float(v) for k, v in solutes.items()}
        return solutes

    def _expand_solutes(self, solutes, inplace=True):
        if not inplace:
            solutes = solutes.copy()
        for cation in self.cations:
            if cation not in solutes:
                solutes[cation] = 0.0
        for anion in self.anions:
            if anion not in solutes:
                solutes[anion] = 0.0
        for neutral in self.neutrals:
            if neutral not in solutes:
                solutes[neutral] = 0.0
        return solutes

    def get_totals(self, sanity_check=True, **totals):
        if sanity_check:
            for k, v in totals.items():
                assert (
                    k in self.totals_all
                ), "Total {} is not part of this Library!".format(k)
                assert v >= 0, "All total molalities must be >= 0."
        self._expand_totals(totals)
        totals = {k: float(v) for k, v in totals.items()}
        return totals

    def _expand_totals(self, totals, inplace=True):
        if not inplace:
            totals = totals.copy()
        for total in self.totals_all:
            if total not in totals:
                totals[total] = 0.0
        return totals

    def _get_charges(self):
        self.charges = {}
        self.charges.update({c: solute_to_charge[c] for c in self.cations})
        self.charges.update({a: solute_to_charge[a] for a in self.anions})
        self.charges_cat = np.array([self.charges[c] for c in self.cations])
        self.charges_ani = np.array([self.charges[a] for a in self.anions])

    def update_Aphi(self, func):
        self.Aphi = func

    def update_func_J(self, func):
        self.func_J = func

    def update_ca(self, cation, anion, func=parameters.bC_none):
        # Add ions to tuples if needed
        if cation not in self.cations:
            self.cations = (*self.cations, cation)
        if anion not in self.anions:
            self.anions = (*self.anions, anion)
        # Add bC function to ca dict
        if cation not in self.ca:
            self.ca[cation] = {}
        self.ca[cation][anion] = func
        # Add bC function to ca_combos array and get_ca_values function
        self._update_all()

    def update_cc(self, cation0, cation1, func=parameters.theta_none):
        # Add ions to tuples if needed
        if cation0 not in self.cations:
            self.cations = (*self.cations, cation0)
        if cation1 not in self.cations:
            self.cations = (*self.cations, cation1)
        # Add theta function to cc dict
        cations = [cation0, cation1]
        cations.sort()
        if cations[0] not in self.cc:
            self.cc[cations[0]] = {}
        self.cc[cations[0]][cations[1]] = func
        # Update all combos arrays and get_xxx_values functions
        self._update_all()

    def update_cca(self, cation0, cation1, anion, func=parameters.psi_none):
        # Add ions to tuples if needed
        if cation0 not in self.cations:
            self.cations = (*self.cations, cation0)
        if cation1 not in self.cations:
            self.cations = (*self.cations, cation1)
        if anion not in self.anions:
            self.anions = (*self.anions, anion)
        # Add psi function to cca dict
        cations = [cation0, cation1]
        cations.sort()
        if cations[0] not in self.cca:
            self.cca[cations[0]] = {}
        if cations[1] not in self.cca[cations[0]]:
            self.cca[cations[0]][cations[1]] = {}
        self.cca[cations[0]][cations[1]][anion] = func
        # Update all combos arrays and get_xxx_values functions
        self._update_all()

    def update_aa(self, anion0, anion1, func=parameters.theta_none):
        # Add ions to tuples if needed
        if anion0 not in self.anions:
            self.anions = (*self.anions, anion0)
        if anion1 not in self.anions:
            self.anions = (*self.anions, anion1)
        # Add theta function to aa dict
        anions = [anion0, anion1]
        anions.sort()
        if anions[0] not in self.aa:
            self.aa[anions[0]] = {}
        self.aa[anions[0]][anions[1]] = func
        # Update all combos arrays and get_xxx_values functions
        self._update_all()

    def update_caa(self, cation, anion0, anion1, func=parameters.psi_none):
        # Add ions to tuples if needed
        if cation not in self.cations:
            self.cations = (*self.cations, cation)
        if anion0 not in self.anions:
            self.anions = (*self.anions, anion0)
        if anion1 not in self.anions:
            self.anions = (*self.anions, anion1)
        # Add psi function to caa dict
        anions = [anion0, anion1]
        anions.sort()
        if cation not in self.caa:
            self.caa[cation] = {}
        if anions[0] not in self.caa[cation]:
            self.caa[cation][anions[0]] = {}
        self.caa[cation][anions[0]][anions[1]] = func
        # Update all combos arrays and get_xxx_values functions
        self._update_all()

    def update_nnn(self, neutral, func=parameters.mu_none):
        # Add solute to tuple if needed
        if neutral not in self.neutrals:
            self.neutrals = (*self.neutrals, neutral)
        # Add mu function to nnn dict
        self.nnn[neutral] = func
        # Update all combos arrays and get_xxx_values functions
        self._update_all()

    def update_nn(self, neutral0, neutral1, func=parameters.lambd_none):
        # Add solutes to tuples if needed
        if neutral0 not in self.neutrals:
            self.neutrals = (*self.neutrals, neutral0)
        if neutral1 not in self.neutrals:
            self.neutrals = (*self.neutrals, neutral1)
        # Add lambda function to nn dict
        neutrals = [neutral0, neutral1]
        neutrals.sort()
        if neutrals[0] not in self.nn:
            self.nn[neutrals[0]] = {}
        self.nn[neutrals[0]][neutrals[1]] = func
        # Update all combos arrays and get_xxx_values functions
        self._update_all()

    def update_nc(self, neutral, cation, func=parameters.lambd_none):
        # Add solutes to tuples if needed
        if neutral not in self.neutrals:
            self.neutrals = (*self.neutrals, neutral)
        if cation not in self.cations:
            self.cations = (*self.cations, cation)
        # Add lambda function to nc dict
        if neutral not in self.nc:
            self.nc[neutral] = {}
        self.nc[neutral][cation] = func
        # Update all combos arrays and get_xxx_values functions
        self._update_all()

    def update_na(self, neutral, anion, func=parameters.lambd_none):
        # Add solutes to tuples if needed
        if neutral not in self.neutrals:
            self.neutrals = (*self.neutrals, neutral)
        if anion not in self.anions:
            self.anions = (*self.anions, anion)
        # Add lambda function to na dict
        if neutral not in self.na:
            self.na[neutral] = {}
        self.na[neutral][anion] = func
        # Update all combos arrays and get_xxx_values functions
        self._update_all()

    def update_nca(self, neutral, cation, anion, func=parameters.zeta_none):
        # Add solutes to tuples if needed
        if neutral not in self.neutrals:
            self.neutrals = (*self.neutrals, neutral)
        if cation not in self.cations:
            self.cations = (*self.cations, cation)
        if anion not in self.anions:
            self.anions = (*self.anions, anion)
        # Add zeta function to nca dict
        if neutral not in self.nca:
            self.nca[neutral] = {}
        if cation not in self.nca[neutral]:
            self.nca[neutral][cation] = {}
        self.nca[neutral][cation][anion] = func
        # Update all combos arrays and get_xxx_values functions
        self._update_all()

    def _update_all(self):
        self._get_charges()
        self._get_ca_combos()
        self._get_ca_values_func()
        self._get_cc_combos()
        self._get_cc_values_func()
        self._get_cca_combos()
        self._get_cca_values_func()
        self._get_aa_combos()
        self._get_aa_values_func()
        self._get_caa_combos()
        self._get_caa_values_func()
        self._get_nnn_combos()
        self._get_nnn_values_func()
        self._get_nn_combos()
        self._get_nn_values_func()
        self._get_nc_combos()
        self._get_nc_values_func()
        self._get_na_combos()
        self._get_na_values_func()
        self._get_nca_combos()
        self._get_nca_values_func()

    def _get_ca_combos(self):
        self.ca_combos = []
        for cation in self.ca:
            c = self.cations.index(cation)
            for anion in self.ca[cation]:
                a = self.anions.index(anion)
                self.ca_combos.append([c, a])
        self.ca_combos = np.array(self.ca_combos)

    def _get_cc_combos(self):
        # This works differently because we still need to include the combo even if
        # there is no theta term in the library, because etheta still gets added
        self.cc_combos = []
        for cation0 in self.cations:
            for cation1 in self.cations:
                if cation0 != cation1:
                    cations = [cation0, cation1]
                    cations.sort()
                    c0 = self.cations.index(cations[0])
                    c1 = self.cations.index(cations[1])
                    if [c0, c1] not in self.cc_combos:
                        self.cc_combos.append([c0, c1])
        self.cc_combos = np.array(self.cc_combos)

    def _get_cca_combos(self):
        self.cca_combos = []
        for cation0 in self.cca:
            c0 = self.cations.index(cation0)
            for cation1 in self.cca[cation0]:
                c1 = self.cations.index(cation1)
                for anion in self.cca[cation0][cation1]:
                    a = self.anions.index(anion)
                    self.cca_combos.append([c0, c1, a])
        self.cca_combos = np.array(self.cca_combos)

    def _get_aa_combos(self):
        # This works differently because we still need to include the combo even if
        # there is no theta term in the library, because the etheta term still gets
        # added if charges are different
        self.aa_combos = []
        for anion0 in self.anions:
            for anion1 in self.anions:
                if anion0 != anion1:
                    anions = [anion0, anion1]
                    anions.sort()
                    a0 = self.anions.index(anions[0])
                    a1 = self.anions.index(anions[1])
                    if [a0, a1] not in self.aa_combos:
                        self.aa_combos.append([a0, a1])
        self.aa_combos = np.array(self.aa_combos)

    def _get_caa_combos(self):
        self.caa_combos = []
        for cation in self.caa:
            c = self.cations.index(cation)
            for anion0 in self.caa[cation]:
                a0 = self.anions.index(anion0)
                for anion1 in self.caa[cation][anion0]:
                    a1 = self.anions.index(anion1)
                    self.caa_combos.append([c, a0, a1])
        self.caa_combos = np.array(self.caa_combos)

    def _get_nnn_combos(self):
        self.nnn_combos = []
        for neutral in self.nnn:
            n = self.neutrals.index(neutral)
            self.nnn_combos.append(n)
        self.nnn_combos = np.array(self.nnn_combos)

    def _get_nn_combos(self):
        self.nn_combos = []
        for neutral0 in self.nn:
            n0 = self.neutrals.index(neutral0)
            for neutral1 in self.nn[neutral0]:
                n1 = self.neutrals.index(neutral1)
                self.nn_combos.append([n0, n1])
        self.nn_combos = np.array(self.nn_combos)

    def _get_nc_combos(self):
        self.nc_combos = []
        for neutral in self.nc:
            n = self.neutrals.index(neutral)
            for cation in self.nc[neutral]:
                c = self.cations.index(cation)
                self.nc_combos.append([n, c])
        self.nc_combos = np.array(self.nc_combos)

    def _get_na_combos(self):
        self.na_combos = []
        for neutral in self.na:
            n = self.neutrals.index(neutral)
            for anion in self.na[neutral]:
                a = self.anions.index(anion)
                self.na_combos.append([n, a])
        self.na_combos = np.array(self.na_combos)

    def _get_nca_combos(self):
        self.nca_combos = []
        for neutral in self.nca:
            n = self.neutrals.index(neutral)
            for cation in self.nca[neutral]:
                c = self.cations.index(cation)
                for anion in self.nca[neutral][cation]:
                    a = self.anions.index(anion)
                    self.nca_combos.append([n, c, a])
        self.nca_combos = np.array(self.nca_combos)

    def _get_ca_values_func(self):
        self.get_ca_values = lambda temperature, pressure: np.array(
            [
                [
                    (
                        self.ca[cation][anion](temperature, pressure)[:-1]
                        if cation in self.ca and anion in self.ca[cation]
                        else parameters.bC_none(temperature, pressure)[:-1]
                    )
                    for anion in self.anions
                ]
                for cation in self.cations
            ]
        )

    def _get_cc_values_func(self):
        self.get_cc_values = lambda temperature, pressure: np.array(
            [
                [
                    (
                        self.cc[cation0][cation1](temperature, pressure)[0]
                        if cation0 in self.cc and cation1 in self.cc[cation0]
                        else 0.0
                    )
                    for cation1 in self.cations
                ]
                for cation0 in self.cations
            ]
        )

    def _get_cca_values_func(self):
        self.get_cca_values = lambda temperature, pressure: np.array(
            [
                [
                    [
                        (
                            self.cca[cation0][cation1][anion](temperature, pressure)[0]
                            if cation0 in self.cca
                            and cation1 in self.cca[cation0]
                            and anion in self.cca[cation0][cation1]
                            else 0.0
                        )
                        for anion in self.anions
                    ]
                    for cation1 in self.cations
                ]
                for cation0 in self.cations
            ]
        )

    def _get_aa_values_func(self):
        self.get_aa_values = lambda temperature, pressure: np.array(
            [
                [
                    (
                        self.aa[anion0][anion1](temperature, pressure)[0]
                        if anion0 in self.aa and anion1 in self.aa[anion0]
                        else 0.0
                    )
                    for anion1 in self.anions
                ]
                for anion0 in self.anions
            ]
        )

    def _get_caa_values_func(self):
        self.get_caa_values = lambda temperature, pressure: np.array(
            [
                [
                    [
                        (
                            self.caa[cation][anion0][anion1](temperature, pressure)[0]
                            if cation in self.caa
                            and anion0 in self.caa[cation]
                            and anion1 in self.caa[cation][anion0]
                            else 0.0
                        )
                        for anion1 in self.anions
                    ]
                    for anion0 in self.anions
                ]
                for cation in self.cations
            ]
        )

    def _get_nnn_values_func(self):
        self.get_nnn_values = lambda temperature, pressure: np.array(
            [
                (
                    self.nnn[neutral](temperature, pressure)[0]
                    if neutral in self.nnn
                    else 0.0
                )
                for neutral in self.neutrals
            ]
        )

    def _get_nn_values_func(self):
        self.get_nn_values = lambda temperature, pressure: np.array(
            [
                [
                    (
                        self.nn[neutral0][neutral1](temperature, pressure)[0]
                        if neutral0 in self.nn and neutral1 in self.nn[neutral0]
                        else 0.0
                    )
                    for neutral1 in self.neutrals
                ]
                for neutral0 in self.neutrals
            ]
        )

    def _get_nc_values_func(self):
        self.get_nc_values = lambda temperature, pressure: np.array(
            [
                [
                    (
                        self.nc[neutral][cation](temperature, pressure)[0]
                        if neutral in self.nc and cation in self.nc[neutral]
                        else 0.0
                    )
                    for cation in self.cations
                ]
                for neutral in self.neutrals
            ]
        )

    def _get_na_values_func(self):
        self.get_na_values = lambda temperature, pressure: np.array(
            [
                [
                    (
                        self.na[neutral][anion](temperature, pressure)[0]
                        if neutral in self.na and anion in self.na[neutral]
                        else 0.0
                    )
                    for anion in self.anions
                ]
                for neutral in self.neutrals
            ]
        )

    def _get_nca_values_func(self):
        self.get_nca_values = lambda temperature, pressure: np.array(
            [
                [
                    [
                        (
                            self.nca[neutral][cation][anion](temperature, pressure)[0]
                            if neutral in self.nca
                            and cation in self.nca[neutral]
                            and anion in self.nca[neutral][cation]
                            else 0.0
                        )
                        for anion in self.anions
                    ]
                    for cation in self.cations
                ]
                for neutral in self.neutrals
            ]
        )

    def update_equilibrium(self, equilibrium, func):
        """Add or update the function for a thermodynamic equilibrium constant."""
        self.equilibria[equilibrium] = func
        self._get_equilibria_all()

    def _get_equilibria_all(self):
        self.equilibria_all = tuple(self.equilibria.keys())
