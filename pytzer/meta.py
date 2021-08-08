# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
"""Define package metadata."""
import importlib

version = "0.5.0"
author = "Matthew P. Humphreys and Abigail J. Schiller"


def update_func_J(pytzer, func_J):
    """Update the unsymmetrical mixing function."""
    if pytzer.model.func_J is not func_J:
        pytzer.model = importlib.reload(pytzer.model)
        pytzer = importlib.reload(pytzer)
        pytzer.model.func_J = func_J
    return pytzer


def hello():
    print(
        """
      ____          __                   
     / __ \ __  __ / /_ ____  ___   _____
    / /_/ // / / // __//_  / / _ \ / ___/
   / ____// /_/ // /_   / /_/  __// /    
  / /     \__, / \__/  /___/\___//_/     
  \/     /____/        
  
   M.P. Humphreys & A.J. Schiller (2021)
    v{version} | doi:{doi}
""".format(
            version=version, doi="10.5281/zenodo.2637914"
        )
    )
