# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2024  M.P. Humphreys  (GNU GPLv3)
"""Assemble parameter libraries."""

from .library_class import Library
from . import (
    lib_CHW22,
    lib_CRP94,
    lib_CWTD23,
    lib_HWT22,
    lib_M88,
)

# For convenience
libraries_all = {
    k.upper(): v.library
    for k, v in {
        "CHW22": lib_CHW22,
        "CRP94": lib_CRP94,
        "CWTD23": lib_CWTD23,
        "HWT22": lib_HWT22,
        "M88": lib_M88,
    }.items()
}
