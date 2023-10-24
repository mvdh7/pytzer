# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2023  Matthew P. Humphreys  (GNU GPLv3)
"""Pytzer package metadata."""
import importlib

version = "0.5.3"
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
