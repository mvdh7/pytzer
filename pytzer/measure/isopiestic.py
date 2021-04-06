# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
"""Isopiestic equilibrium measurements."""
from .. import matrix, properties


def osmR2osmS(totS, totR, nS, nR, osmR):
    """Calculate osmotic coefficient of sample given a reference."""
    return totR * nR * osmR / (totS * nS)


def getosmS(totS, totR, eleS, eleR, tempK, pres, prmlib):
    """"""
    ions = properties.getallions([eleS, eleR], [])
    #    matrix.assemble(ions, tempK, pres, prmlib)
    return ions
