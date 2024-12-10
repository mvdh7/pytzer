# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2024  M.P. Humphreys  (GNU GPLv3)
"""Package metadata and metatools."""

import importlib

from . import libraries

version = "0.6.0"
year = 2024
authors = ["Humphreys, M.P.", "Schiller, A.J."]
author = " and ".join(authors)


def set_library(pytzer, library):
    """Set the parameter library."""
    if isinstance(library, str):
        assert (
            library.upper() in libraries.libraries_all
        ), "Library must be in pz.libraries_all!"
        library = libraries.libraries_all[library.upper()]
    if pytzer.model.library is not library:
        pytzer.model = importlib.reload(pytzer.model)
        pytzer.model.library = library
        pytzer.equilibrate.solver = importlib.reload(pytzer.equilibrate.solver)
        pytzer = importlib.reload(pytzer)
    return pytzer


def hello():
    print(
        r"""
      ____          __                   
     / __ \ __  __ / /_ ____  ___   _____
    / /_/ // / / // __//_  / / _ \ / ___/
   / ____// /_/ // /_   / /_/  __// /    
  / /     \__, / \__/  /___/\___//_/     
  \/     /____/        
  
   M.P. Humphreys & A.J. Schiller ({year})
    v{version} | doi:{doi}
""".format(year=year, version=version, doi="10.5281/zenodo.2637914")
    )
