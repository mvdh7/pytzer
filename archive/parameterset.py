from jax import numpy as np


def ca_none(T, P):
    """cation:anion --- no interaction effect."""
    return {
        "beta0": 0,
        "beta1": 0,
        "alpha1": 1,
        "beta2": 0,
        "alpha2": 1,
        "C0": 0,
        "C1": 0,
        "omega": 1,
        "valid": (T > 0) & (P > 0),
    }


def g(x):
    """g function following CRP94 Eq. (AI13)."""
    return 2 * (1 - (1 + x) * np.exp(-x)) / x ** 2


def h(x):
    """h function following CRP94 Eq. (AI15)."""
    return (6 - (6 + x * (6 + 3 * x + x ** 2)) * np.exp(-x)) / x ** 4


def B(I, beta0, beta1, alpha1, beta2, alpha2):
    """B function following CRP94 Eq. (AI7)."""
    return beta0 + beta1 * g(alpha1 * np.sqrt(I)) + beta2 * g(alpha2 * np.sqrt(I))


def CT(I, C0, C1, omega):
    """CT function following CRP94 Eq. (AI10)."""
    return C0 + 4 * C1 * h(omega * np.sqrt(I))


class CationAnion:
    def __init__(self, cation, anion, func=ca_none):
        self.cation = cation
        self.anion = anion
        self.func = func

    def get_parameters(self, T=298.15, P=1.01325):
        return self.func(T, P)

    def get_BC(self, I):
        return


# class ParameterLibrary:
#     pass


class Interactions:
    def __init__(self, name=None):
        self.name = name
        self.interactions = {}

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

    def add_interaction(self, interaction):
        self.interactions[interaction.cation + ":" + interaction.anion] = interaction


ca = CationAnion("Na", "Cl")

test = Interactions()
test.add_interaction(ca)
