# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2024  Matthew P. Humphreys  (GNU GPLv3)
"""Package metadata and metatools."""
import importlib
from . import libraries

version = "0.6.0"
authors = ["Humphreys, M.P.", "Schiller, A.J."]
author = " and ".join(authors)


def update_func_J(pytzer, func_J):
    """Update the unsymmetrical mixing function."""
    if pytzer.model_old.func_J is not func_J:
        pytzer.model_old = importlib.reload(pytzer.model_old)
        pytzer = importlib.reload(pytzer)
        pytzer.model_old.func_J = func_J
    return pytzer


def update_library(pytzer, library):
    """Update the parameter library."""
    if isinstance(library, str):
        library = libraries.libraries[library.lower()]
    if pytzer.model.library is not library:
        pytzer.model = importlib.reload(pytzer.model)
        pytzer = importlib.reload(pytzer)
        pytzer.model.library = library
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
  
   M.P. Humphreys & A.J. Schiller (2023)
    v{version} | doi:{doi}
""".format(
            version=version, doi="10.5281/zenodo.2637914"
        )
    )
