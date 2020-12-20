from pytzer import parameters as prm, debyehueckel as dh, unsymmetrical as unsym
import numpy as np, pytzer as pz


class ParameterLibrary(dict):
    def update_Aphi(self, func=dh.Aosm_M88):
        self.update({"Aphi": func})

    def update_ca(self, cation, anion, func=prm.bC_none):
        if "ca" not in self:
            self["ca"] = {}
        if cation not in self["ca"]:
            self["ca"][cation] = {}
        self["ca"][cation].update({anion: func})

    def update_cc(self, cation1, cation2, func=prm.theta_none):
        if "cc" not in self:
            self["cc"] = {}
        if cation1 not in self["cc"]:
            self["cc"][cation1] = {}
        if cation2 not in self["cc"]:
            self["cc"][cation2] = {}
        self["cc"][cation1].update({cation2: func})
        self["cc"][cation2].update({cation1: func})

    def update_aa(self, anion1, anion2, func=prm.theta_none):
        if "aa" not in self:
            self["aa"] = {}
        if anion1 not in self["aa"]:
            self["aa"][anion1] = {}
        if anion2 not in self["aa"]:
            self["aa"][anion2] = {}
        self["aa"][anion1].update({anion2: func})
        self["aa"][anion2].update({anion1: func})

    def update_cca(self, cation1, cation2, anion, func=prm.psi_none):
        if "cca" not in self:
            self["cca"] = {}
        if cation1 not in self["cca"]:
            self["cca"][cation1] = {cation2: {anion: func}}
        if cation2 not in self["cca"][cation1]:
            self["cca"][cation1].update({cation2: {anion: func}})
        if anion not in self["cca"][cation1][cation2]:
            self["cca"][cation1][cation2].update({anion: func})
        if cation2 not in self["cca"]:
            self["cca"][cation2] = {cation1: {anion: func}}
        if cation1 not in self["cca"][cation2]:
            self["cca"][cation2].update({cation1: {anion: func}})
        if anion not in self["cca"][cation2][cation1]:
            self["cca"][cation2][cation1].update({anion: func})

    def update_caa(self, cation, anion1, anion2, func=prm.psi_none):
        if "caa" not in self:
            self["caa"] = {}
        if cation not in self["caa"]:
            self["caa"][cation] = {anion1: {anion2: func}, anion2: {anion1: func}}
        if anion1 not in self["caa"][cation]:
            self["caa"][cation][anion1] = {anion2: func}
        if anion2 not in self["caa"][cation]:
            self["caa"][cation][anion2] = {anion1: func}

    def assign_func_J(self, func_J):
        self["func_J"] = func_J

    def set_func_J(self, pytzer, func_J=None):
        if func_J is None:
            assert "func_J" in self
        else:
            self.assign_func_J(func_J)
        pytzer = pz.update_func_J(pytzer, self["func_J"])
        return pytzer

    def get_parameters(
        self,
        cations=None,
        anions=None,
        neutrals=None,
        temperature=298.15,
        pressure=10.1023,
    ):
        parameters = {}
        TP = (temperature, pressure)
        if "Aphi" in self:
            parameters.update({"Aphi": self["Aphi"](*TP)[0]})
        if cations is not None and anions is not None:
            if "ca" in self:
                parameters["ca"] = np.zeros((len(cations), len(anions), 8))
                for c, cation in enumerate(cations):
                    for a, anion in enumerate(anions):
                        try:
                            ca = self["ca"][cation][anion](*TP)[:-1]
                        except KeyError:
                            ca = np.zeros(8)
                        parameters["ca"][c][a] = ca
            if "cc" in self:
                parameters["cc"] = np.zeros((len(cations), len(cations)))
                for c1, cation1 in enumerate(cations):
                    for c2, cation2 in enumerate(cations):
                        try:
                            cc = self["cc"][cation1][cation2](*TP)[0]
                        except KeyError:
                            cc = 0.0
                        parameters["cc"][c1][c2] = cc
            if "aa" in self:
                parameters["aa"] = np.zeros((len(anions), len(anions)))
                for a1, anion1 in enumerate(anions):
                    for a2, anion2 in enumerate(anions):
                        try:
                            aa = self["aa"][anion1][anion2](*TP)[0]
                        except KeyError:
                            aa = 0.0
                        parameters["aa"][a1][a2] = aa
            if "cca" in self:
                parameters["cca"] = np.zeros((len(cations), len(cations), len(anions)))
                for c1, cation1 in enumerate(cations):
                    for c2, cation2 in enumerate(cations):
                        for a, anion in enumerate(anions):
                            try:
                                cca = self["cca"][cation1][cation2][anion](*TP)[0]
                            except KeyError:
                                cca = 0.0
                            parameters["cca"][c1][c2][a] = cca
            if "caa" in self:
                parameters["caa"] = np.zeros((len(cations), len(anions), len(anions)))
                for c, cation in enumerate(cations):
                    for a1, anion1 in enumerate(anions):
                        for a2, anion2 in enumerate(anions):
                            try:
                                caa = self["caa"][cation][anion1][anion2](*TP)[0]
                            except KeyError:
                                caa = 0.0
                            parameters["caa"][c][a1][a2] = caa
        return parameters


Moller88 = ParameterLibrary()
Moller88.update_Aphi(dh.Aosm_M88)
Moller88.update_ca("Ca", "Cl", prm.bC_Ca_Cl_M88)
Moller88.update_ca("Ca", "SO4", prm.bC_Ca_SO4_M88)
Moller88.update_ca("Na", "Cl", prm.bC_Na_Cl_M88)
Moller88.update_ca("Na", "SO4", prm.bC_Na_SO4_M88)
Moller88.update_cc("Ca", "Na", prm.theta_Ca_Na_M88)
Moller88.update_aa("Cl", "SO4", prm.theta_Cl_SO4_M88)
Moller88.update_cca("Ca", "Na", "Cl", prm.psi_Ca_Na_Cl_M88)
Moller88.update_cca("Ca", "Na", "SO4", prm.psi_Ca_Na_SO4_M88)
Moller88.update_caa("Ca", "Cl", "SO4", prm.psi_Ca_Cl_SO4_M88)
Moller88.update_caa("Na", "Cl", "SO4", prm.psi_Na_Cl_SO4_M88)
Moller88.assign_func_J(unsym.Harvie)

params = Moller88.get_parameters(cations=["Na", "Ca"], anions=["Cl", "SO4"])
# print(params)

pz.model.func_J = unsym.none

molalities = np.array([1.0, 1.0, 1.0, 1.0])
charges = np.array([+1, -1, +2, -2])
args = pz.split_molalities_charges(molalities, charges)
gibbs = pz.Gibbs_nRT(*args, **params)
acf = pz.activity_coefficients(*args, **params)
print(acf)

# import importlib
# pz.model = importlib.reload(pz.model)
# pz = importlib.reload(pz)
# pz.model.func_J = unsym.Harvie

# pz.update_func_J(pz, unsym.Harvie)
Moller88.set_func_J(pz)
acf = pz.activity_coefficients(*args, **params)
print(acf)

Moller88.set_func_J(pz)
acf = pz.activity_coefficients(*args, **params)
print(acf)
