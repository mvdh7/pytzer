# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
"""Assemble dicts of Pitzer model interaction parameter functions into
parameter libraries.
"""
from autograd.numpy import array
from .ParameterLibrary import ParameterLibrary
from . import (
    CRP94,
    GM89,
    HMW84,
    M88,
    MarChemSpec,
    MarChemSpec05,
    MarChemSpec25,
    MIAMI,
    Seawater,
    WM13,
    WM13_MarChemSpec25,
)

__all__ = [
    'CRP94',
    'GM89',
    'HMW84',
    'M88',
    'MarChemSpec',
    'MarChemSpec05',
    'MarChemSpec25',
    'MIAMI',
    'Seawater',
    'WM13',
    'WM13_MarChemSpec25',
]

MarChemSpecSolutes = array(['H', 'Na', 'Mg', 'Ca', 'K', 'MgOH', 'trisH',
    'Cl', 'SO4', 'HSO4', 'OH', 'tris'])

# Clegg et al. (1994)
CRP94 = ParameterLibrary(module=CRP94)
CRP94.get_contents()

# Greenberg & Møller (1989)
GM89 = ParameterLibrary(module=GM89)
GM89.get_contents()

# Harvie, Møller & Weare (1984)
HMW84 = ParameterLibrary(module=HMW84)
HMW84.get_contents()

# Møller (1988)
M88 = ParameterLibrary(module=M88)
M88.get_contents()

# Begin with WM13_MarChemSpec25, switch to CRP94 corrected Aosm, T-variable
MarChemSpec = ParameterLibrary(module=MarChemSpec)
MarChemSpec.add_zeros(MarChemSpecSolutes)
MarChemSpec.get_contents()

# MarChemSpec project (i.e. WM13 plus tris) with Aosm at 5 degC
MarChemSpec05 = ParameterLibrary(module=MarChemSpec05)
MarChemSpec05.add_zeros(MarChemSpecSolutes)
MarChemSpec05.get_contents()

# MarChemSpec project (i.e. WM13 plus tris) with Aosm at 25 degC
MarChemSpec25 = ParameterLibrary(module=MarChemSpec25)
MarChemSpec25.add_zeros(MarChemSpecSolutes)
MarChemSpec25.get_contents()

# Millero & Pierrot 1998 aka MIAMI - WORK IN PROGRESS!
MIAMI = ParameterLibrary(module=MIAMI)
MIAMI.get_contents()

# Seawater: MarChemSpec with pressure
Seawater = ParameterLibrary(module=Seawater)
Seawater.add_zeros(MarChemSpecSolutes)
Seawater.get_contents()

# Waters & Millero (2013)
WM13 = ParameterLibrary(module=WM13)
WM13.get_contents()

# Waters & Millero (2013) with Aosm at 25 degC
WM13_MarChemSpec25 = ParameterLibrary(module=WM13_MarChemSpec25)
WM13_MarChemSpec25.get_contents()
