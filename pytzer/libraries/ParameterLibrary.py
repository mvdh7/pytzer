# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2023  Matthew P. Humphreys  (GNU GPLv3)
from copy import deepcopy
from collections import OrderedDict
import numpy as np
from scipy.special import comb
from .. import debyehueckel as dh, parameters as prm, convert
from ..equilibrate import thermodynamic
from ..meta import update_func_J


class ParameterLibrary(dict):
    def __init__(self, name="ParameterLibrary"):
        self["name"] = name

    def update_Aphi(self, func=dh.Aosm_M88):
        """Add or update a Debye-Hueckel limiting slope function."""
        self.update({"Aphi": func})

    def update_ca(self, cation, anion, func=prm.bC_none):
        """Add or update a cation-anion interaction parameters function."""
        if "ca" not in self:
            self["ca"] = {}
        if cation not in self["ca"]:
            self["ca"][cation] = {}
        self["ca"][cation].update({anion: func})

    def update_cc(self, cation1, cation2, func=prm.theta_none):
        """Add or update a cation-cation interaction parameter function."""
        if "cc" not in self:
            self["cc"] = {}
        if cation1 not in self["cc"]:
            self["cc"][cation1] = {}
        if cation2 not in self["cc"]:
            self["cc"][cation2] = {}
        self["cc"][cation1].update({cation2: func})
        self["cc"][cation2].update({cation1: func})

    def update_aa(self, anion1, anion2, func=prm.theta_none):
        """Add or update an anion-anion interaction parameter function."""
        if "aa" not in self:
            self["aa"] = {}
        if anion1 not in self["aa"]:
            self["aa"][anion1] = {}
        if anion2 not in self["aa"]:
            self["aa"][anion2] = {}
        self["aa"][anion1].update({anion2: func})
        self["aa"][anion2].update({anion1: func})

    def update_cca(self, cation1, cation2, anion, func=prm.psi_none):
        """Add or update a cation-cation-anion interaction parameter function."""
        if "cca" not in self:
            self["cca"] = {}
        if cation1 not in self["cca"]:
            self["cca"][cation1] = {}
        if cation2 not in self["cca"][cation1]:
            self["cca"][cation1][cation2] = {}
        if cation2 not in self["cca"]:
            self["cca"][cation2] = {}
        if cation1 not in self["cca"][cation2]:
            self["cca"][cation2][cation1] = {}
        self["cca"][cation1][cation2].update({anion: func})
        self["cca"][cation2][cation1].update({anion: func})

    def update_caa(self, cation, anion1, anion2, func=prm.psi_none):
        """Add or update a cation-anion-anion interaction parameter function."""
        if "caa" not in self:
            self["caa"] = {}
        if cation not in self["caa"]:
            self["caa"][cation] = {}
        if anion1 not in self["caa"][cation]:
            self["caa"][cation][anion1] = {}
        if anion2 not in self["caa"][cation]:
            self["caa"][cation][anion2] = {}
        self["caa"][cation][anion1].update({anion2: func})
        self["caa"][cation][anion2].update({anion1: func})

    def update_nc(self, neutral, cation, func=prm.lambd_none):
        """Add or update a neutral-cation interaction parameter function."""
        if "nc" not in self:
            self["nc"] = {}
        if neutral not in self["nc"]:
            self["nc"][neutral] = {}
        self["nc"][neutral].update({cation: func})

    def update_na(self, neutral, anion, func=prm.lambd_none):
        """Add or update a cation-anion interaction parameter function."""
        if "na" not in self:
            self["na"] = {}
        if neutral not in self["na"]:
            self["na"][neutral] = {}
        self["na"][neutral].update({anion: func})

    def update_nca(self, neutral, cation, anion, func=prm.zeta_none):
        """Add or update a neutral-cation-anion interaction parameter function."""
        if "nca" not in self:
            self["nca"] = {}
        if neutral not in self["nca"]:
            self["nca"][neutral] = {}
        if cation not in self["nca"][neutral]:
            self["nca"][neutral][cation] = {}
        self["nca"][neutral][cation].update({anion: func})

    def update_nn(self, neutral1, neutral2, func=prm.lambd_none):
        """Add or update a neutral-neutral interaction parameter function."""
        if "nn" not in self:
            self["nn"] = {}
        if neutral1 not in self["nn"]:
            self["nn"][neutral1] = {}
        if neutral2 not in self["nn"]:
            self["nn"][neutral2] = {}
        self["nn"][neutral1].update({neutral2: func})
        self["nn"][neutral2].update({neutral1: func})

    def update_nnn(self, neutral, func=prm.mu_none):
        """Add or update a neutral-neutral-neutral interaction parameter function."""
        if "nnn" not in self:
            self["nnn"] = {}
        self["nnn"].update({neutral: func})

    def assign_func_J(self, func_J):
        """Assign which J function should be used for unsymmetrical mixing terms."""
        self["func_J"] = func_J

    def set_func_J(self, pytzer, func_J=None):
        """Implement the assigned J function throughout pytzer."""
        if func_J is None:
            assert "func_J" in self
        else:
            self.assign_func_J(func_J)
        pytzer = update_func_J(pytzer, self["func_J"])
        return pytzer

    def update_equilibrium(self, equilibrium, func):
        """Add or update the function for a thermodynamic equilibrium constant."""
        if "equilibria" not in self:
            self["equilibria"] = OrderedDict()
        self["equilibria"][equilibrium] = func

    def get_parameters(
        self,
        solutes=None,
        temperature=298.15,
        pressure=10.1023,
        verbose=True,
    ):
        """Evaluate all interaction parameters under specific conditions for
        non-equilibrating calculations.
        """
        if verbose:
            missing_coeffs = []

        def report_missing_coeffs(*solutes):
            if verbose:
                solutes_list = list(solutes)
                solutes_list.sort()
                if solutes_list not in missing_coeffs:
                    print(
                        (
                            "{} has no interaction coefficients for "
                            + "-".join(["{}"] * len(solutes))
                            + "; using zero."
                        ).format(self["name"], *solutes)
                    )
                    missing_coeffs.append(solutes_list)

        if solutes is None:
            cations = anions = neutrals = None
        else:
            cations = [s for s in solutes if s in convert.all_cations]
            anions = [s for s in solutes if s in convert.all_anions]
            neutrals = [s for s in solutes if s in convert.all_neutrals]
        if len(cations) == 0:
            cations = None
        if len(anions) == 0:
            anions = None
        if len(neutrals) == 0:
            neutrals = None

        parameters = {"temperature": temperature, "pressure": pressure}
        TP = (temperature, pressure)
        if "Aphi" in self:
            parameters.update({"Aphi": self["Aphi"](*TP)[0]})
        else:
            if verbose:
                print(
                    "{} has no Aphi function; no value returned.".format(self["name"])
                )
        if cations is not None and anions is not None:
            parameters["ca"] = np.zeros((len(cations), len(anions), 8))
            for c, cation in enumerate(cations):
                for a, anion in enumerate(anions):
                    try:
                        ca = self["ca"][cation][anion](*TP)[:-1]
                    except KeyError:
                        ca = [0, 0, 0, 0, 0, -9, -9, -9]
                        report_missing_coeffs(cation, anion)
                    parameters["ca"][c][a] = ca
            parameters["cc"] = np.zeros((len(cations), len(cations)))
            for c1, cation1 in enumerate(cations):
                for c2, cation2 in enumerate(cations):
                    try:
                        cc = self["cc"][cation1][cation2](*TP)[0]
                    except KeyError:
                        cc = 0.0
                        if cation1 != cation2:
                            report_missing_coeffs(cation1, cation2)
                    parameters["cc"][c1][c2] = cc
            parameters["aa"] = np.zeros((len(anions), len(anions)))
            for a1, anion1 in enumerate(anions):
                for a2, anion2 in enumerate(anions):
                    try:
                        aa = self["aa"][anion1][anion2](*TP)[0]
                    except KeyError:
                        aa = 0.0
                        if anion1 != anion2:
                            report_missing_coeffs(anion1, anion2)
                    parameters["aa"][a1][a2] = aa
            parameters["cca"] = np.zeros((len(cations), len(cations), len(anions)))
            for c1, cation1 in enumerate(cations):
                for c2, cation2 in enumerate(cations):
                    for a, anion in enumerate(anions):
                        try:
                            cca = self["cca"][cation1][cation2][anion](*TP)[0]
                        except KeyError:
                            cca = 0.0
                            if cation1 != cation2:
                                report_missing_coeffs(cation1, cation2, anion)
                        parameters["cca"][c1][c2][a] = cca
            parameters["caa"] = np.zeros((len(cations), len(anions), len(anions)))
            for c, cation in enumerate(cations):
                for a1, anion1 in enumerate(anions):
                    for a2, anion2 in enumerate(anions):
                        try:
                            caa = self["caa"][cation][anion1][anion2](*TP)[0]
                        except KeyError:
                            caa = 0.0
                            if anion1 != anion2:
                                report_missing_coeffs(cation, anion1, anion2)
                        parameters["caa"][c][a1][a2] = caa
        if neutrals is not None and cations is not None:
            parameters["nc"] = np.zeros((len(neutrals), len(cations)))
            for n, neutral in enumerate(neutrals):
                for c, cation in enumerate(cations):
                    try:
                        nc = self["nc"][neutral][cation](*TP)[0]
                    except KeyError:
                        nc = 0.0
                        report_missing_coeffs(neutral, cation)
                    parameters["nc"][n][c] = nc
        if neutrals is not None and anions is not None:
            parameters["na"] = np.zeros((len(neutrals), len(anions)))
            for n, neutral in enumerate(neutrals):
                for a, anion in enumerate(anions):
                    try:
                        na = self["na"][neutral][anion](*TP)[0]
                    except KeyError:
                        na = 0.0
                        report_missing_coeffs(neutral, anion)
                    parameters["na"][n][a] = na
            if cations is not None:
                parameters["nca"] = np.zeros((len(neutrals), len(cations), len(anions)))
                for n, neutral in enumerate(neutrals):
                    for c, cation in enumerate(cations):
                        for a, anion in enumerate(anions):
                            try:
                                nca = self["nca"][neutral][cation][anion](*TP)[0]
                            except KeyError:
                                nca = 0.0
                                report_missing_coeffs(neutral, cation, anion)
                            parameters["nca"][n][c][a] = nca
        if neutrals is not None:
            parameters["nn"] = np.zeros((len(neutrals), len(neutrals)))
            parameters["nnn"] = np.zeros(len(neutrals))
            for n1, neutral1 in enumerate(neutrals):
                for n2, neutral2 in enumerate(neutrals):
                    try:
                        nn = self["nn"][neutral1][neutral2](*TP)[0]
                    except KeyError:
                        nn = 0.0
                        report_missing_coeffs(neutral1, neutral2)
                    parameters["nn"][n1][n2] = nn
                try:
                    nnn = self["nnn"][neutral1](*TP)[0]
                except KeyError:
                    nnn = 0.0
                    report_missing_coeffs(neutral1, neutral1, neutral1)
                parameters["nnn"][n1] = nnn
        return parameters

    def get_equilibria(self, solutes=None, temperature=298.15):
        log_kt_constants = OrderedDict()
        if "equilibria" in self:
            for eq, eqf in self["equilibria"].items():
                include_here = True
                if solutes is not None:
                    for s in thermodynamic.solutes_required[eq]:
                        include_here = include_here and s in solutes
                if include_here:
                    log_kt_constants[eq] = eqf(T=temperature)
        return log_kt_constants

    def get_parameters_equilibria(
        self,
        solutes=None,
        temperature=298.15,
        pressure=10.1023,
        verbose=True,
    ):
        """Calculate Pitzer model parameters and thermodynamic equilibrium constants
        as log_kt_constants for equilibrating calculations.
        """
        parameters = self.get_parameters(
            solutes=solutes,
            temperature=temperature,
            pressure=pressure,
            verbose=verbose,
        )
        log_kt_constants = self.get_equilibria(solutes=solutes, temperature=temperature)
        return parameters, log_kt_constants

    def get_matrices(
        self,
        solutes,
        temperature=298.15,
        pressure=10.1023,
        verbose=True,
    ):
        """Assemble parameter matrices for the matrix model."""
        if verbose:
            missing_coeffs = []

        def report_missing_coeffs(*solutes):
            if verbose:
                solutes_list = list(solutes)
                solutes_list.sort()
                if solutes_list not in missing_coeffs:
                    print(
                        (
                            "{} has no interaction coefficients for "
                            + "-".join(["{}"] * len(solutes))
                            + "; using zero."
                        ).format(self["name"], *solutes)
                    )
                    missing_coeffs.append(solutes_list)

        cations = [s for s in solutes if s in convert.all_cations]
        anions = [s for s in solutes if s in convert.all_anions]
        neutrals = [s for s in solutes if s in convert.all_neutrals]
        # Set up parameters dict
        parameters = {"temperature": temperature, "pressure": pressure}
        TP = (temperature, pressure)
        if "Aphi" in self:
            parameters.update({"Aphi": self["Aphi"](*TP)[0]})
        else:
            if verbose:
                print(
                    "{} has no Aphi function; no value returned.".format(self["name"])
                )
        # Preallocate empty matrices
        for p in [
            "beta0_ca",
            "beta1_ca",
            "beta2_ca",
            "c0_ca",
            "c1_ca",
            "theta_xx",
            "lambda_nx",
        ]:
            parameters[p] = np.zeros((len(solutes), len(solutes)))
        for p in ["alpha1_ca", "alpha2_ca", "omega_ca"]:
            parameters[p] = np.full((len(solutes), len(solutes)), -9.0)
        for p in ["psi_cca", "psi_caa"]:
            parameters[p] = np.zeros((int(comb(len(cations), 2)), len(anions)))
        parameters["zeta_nca"] = np.zeros((len(neutrals), len(cations) * len(anions)))
        parameters["mu_nnn"] = np.zeros((len(neutrals), 1))
        # Fill the matrices with coefficients: ca, ii and nx interactions
        for x, solute_x in enumerate(solutes):
            for y, solute_y in enumerate(solutes):
                try:
                    params_ca = self["ca"][solute_x][solute_y](*TP)[:-1]
                    (
                        parameters["beta0_ca"][x, y],
                        parameters["beta1_ca"][x, y],
                        parameters["beta2_ca"][x, y],
                        parameters["c0_ca"][x, y],
                        parameters["c1_ca"][x, y],
                        parameters["alpha1_ca"][x, y],
                        parameters["alpha2_ca"][x, y],
                        parameters["omega_ca"][x, y],
                    ) = (
                        parameters["beta0_ca"][y, x],
                        parameters["beta1_ca"][y, x],
                        parameters["beta2_ca"][y, x],
                        parameters["c0_ca"][y, x],
                        parameters["c1_ca"][y, x],
                        parameters["alpha1_ca"][y, x],
                        parameters["alpha2_ca"][y, x],
                        parameters["omega_ca"][y, x],
                    ) = params_ca
                except KeyError:
                    if (
                        solute_x in convert.all_cations
                        and solute_y in convert.all_anions
                    ):
                        report_missing_coeffs(solute_x, solute_y)
                try:
                    param_theta = self["cc"][solute_x][solute_y](*TP)[0]
                    parameters["theta_xx"][x, y] = parameters["theta_xx"][
                        y, x
                    ] = param_theta
                except KeyError:
                    if solute_x != solute_y:
                        if (
                            solute_x in convert.all_cations
                            and solute_y in convert.all_cations
                        ):
                            report_missing_coeffs(solute_x, solute_y)
                try:
                    param_theta = self["aa"][solute_x][solute_y](*TP)[0]
                    parameters["theta_xx"][x, y] = parameters["theta_xx"][
                        y, x
                    ] = param_theta
                except KeyError:
                    if solute_x != solute_y:
                        if (
                            solute_x in convert.all_anions
                            and solute_y in convert.all_anions
                        ):
                            report_missing_coeffs(solute_x, solute_y)
                try:
                    param_lambda = self["nc"][solute_x][solute_y](*TP)[0]
                    parameters["lambda_nx"][x, y] = parameters["lambda_nx"][
                        x, y
                    ] = param_lambda
                except KeyError:
                    if solute_x in convert.all_neutrals:
                        report_missing_coeffs(solute_x, solute_y)
                try:
                    param_lambda = self["na"][solute_x][solute_y](*TP)[0]
                    parameters["lambda_nx"][x, y] = parameters["lambda_nx"][
                        x, y
                    ] = param_lambda
                except KeyError:
                    if solute_x in convert.all_neutrals:
                        report_missing_coeffs(solute_x, solute_y)
                try:
                    param_lambda = self["nn"][solute_x][solute_y](*TP)[0]
                    parameters["lambda_nx"][x, y] = parameters["lambda_nx"][
                        x, y
                    ] = param_lambda
                except KeyError:
                    if solute_x in convert.all_neutrals:
                        report_missing_coeffs(solute_x, solute_y)
        # Fill the matrices with coefficients: cca and caa interactions
        CC = 0
        for CX, cationx in enumerate(cations):
            for xCY, cationy in enumerate(cations[CX + 1 :]):
                for A, anion in enumerate(anions):
                    parameters["psi_cca"][CC, A] = self["cca"][cationx][cationy][anion](
                        *TP
                    )[0]
                CC = CC + 1
        AA = 0
        for AX, anionx in enumerate(anions):
            for xAY, aniony in enumerate(anions[AX + 1 :]):
                for C, cation in enumerate(cations):
                    parameters["psi_caa"][AA, C] = self["caa"][cation][anionx][aniony](
                        *TP
                    )[0]
                AA = AA + 1
        # Fill the matrices with coefficients: nca and nnn interactions
        for N, neutral in enumerate(neutrals):
            parameters["mu_nnn"][N, 0] = self["nnn"][neutral](*TP)[0]
            for C, cation in enumerate(cations):
                for A, anion in enumerate(anions):
                    parameters["zeta_nca"][N, C * len(anions) + A] = self["nca"][
                        neutral
                    ][cation][anion](*TP)[0]
        return parameters
