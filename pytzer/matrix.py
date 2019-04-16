# Pytzer: Pitzer model for chemical activities in aqueous solutions
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)

from autograd.numpy import transpose

"""Matrix version of the Pitzer model."""

def Istr(mols, zs):
    """Calculate ionic strength."""
    return mols @ transpose(zs**2) / 2
