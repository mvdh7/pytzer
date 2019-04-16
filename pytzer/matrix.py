# Pytzer: Pitzer model for chemical activities in aqueous solutions
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)

from autograd.numpy import log, sqrt, transpose
from .constants import b
from .model import g

"""Matrix version of the Pitzer model."""

def Istr(mols, zs):
    """Calculate the ionic strength."""
    return mols @ transpose(zs**2) / 2

def fG(Aosm, I):
    """Calculate the Debye-Hueckel component of the excess Gibbs energy."""
    return -4*I*Aosm * log(1 + b*sqrt(I)) / b

def B