# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2020  Matthew Paul Humphreys  (GNU GPLv3)
"""Define package metadata."""
import importlib

# from . import parameters

version = "0.5.0"


def update_func_J(pytzer, func_J):
    """Update the unsymmetrical mixing function."""
    if pytzer.model.func_J is not func_J:
        pytzer.model = importlib.reload(pytzer.model)
        pytzer = importlib.reload(pytzer)
        pytzer.model.func_J = func_J
    return pytzer


# def getprmfuncs():
#     """Generate dict containing all interaction parameter functions."""
#     functypes = ("bC", "theta", "psi", "lambd", "zeta", "mu")
#     functypes_ = tuple(["{}_".format(functype) for functype in functypes])
#     allnames = [
#         name
#         for name in dir(parameters)
#         if name.startswith(functypes_) and not name.endswith("_none")
#     ]
#     prmfuncs = {
#         functype: {
#             name: getattr(parameters, name)
#             for name in allnames
#             if name.startswith("{}_".format(functype))
#         }
#         for functype in functypes
#     }
#     return prmfuncs


# def getifuncs(itype, ions):
#     """Extract all interaction functions for a particular interaction."""
#     prmfuncs = getprmfuncs()
#     ifuncs = {
#         name: prmfuncs[itype][name]
#         for name in prmfuncs[itype].keys()
#         if name.startswith(((1 + len(ions)) * "{}_").format(itype, *ions))
#     }
#     return ifuncs


# def evalifuncs(ifuncs, tempK, pres):
#     """Evaluate interaction parameters under given conditions."""
#     ivals = {name: ifuncs[name](tempK, pres) for name in ifuncs.keys()}
#     return ivals


# def getirefs(ifuncs):
#     """Generate a list of literature sources for the interaction functions."""
#     return [name.split("_")[-1] for name in ifuncs.keys()]
