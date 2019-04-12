---
title: 'Chemical activites in aqueous solutions using automatic differentiation of a Pitzer model (pytzer)'
tags:
  - Python
  - chemistry
  - activity
  - Pitzer model
authors:
  - name: Matthew P. Humphreys
    orcid: 0000-0002-9371-7128
    affiliation: 1
affiliations:
  - name: Centre for Ocean and Atmospheric Sciences, School of Environmental Sciences, University of East Anglia, Norwich, UK
    index: 1
date: 2019
bibliography: paper.bib
---


# Summary

Pytzer is a Python implementation of the Pitzer model for modelling chemical activities in aqueous solutions [@P91; @CRP94]. The Pitzer model is based on a 'master' equation for the excess Gibbs energy of a solution, which is differentiated with respect to different input variables to determine different properties of the solution. Previous implementations have used algebraic differentials of the excess Gibbs energy equation, but pytzer uses automatic differentiation instead, as implemented by the autograd package [@autograd]. This approach allows the code to be greatly shortened and simplified, and it substantially reduces the scope for typographical errors to arise.


# Future development

Many other properties of aqueous solutions can be determined from other differentials of the excess Gibbs energy equation, for example with respect to temperature (e.g. enthalpy, heat capacity) or pressure (e.g. molar volume). Implementing functions for these properties would be trivial using autograd, but for the results to be meaningful, all of the Pitzer model coefficients would need to be sensitive to the derivative variable, which is not yet the case.


# Availability

The latest stable release of pytzer can be installed via the Python package index ([https://pypi.org/project/pytzer/](https://pypi.org/project/pytzer/)). The code is available at [https://github.com/mvdh7/pytzer/](https://github.com/mvdh7/pytzer/), together with detailed documentation (which can also be found at [https://pytzer.readthedocs.io/](https://pytzer.readthedocs.io/)).


# Acknowledgements

We acknowledge funding from the Natural Environment Research Council (NERC, UK) through *NSFGEO-NERC: A Thermodynamic Chemical Speciation Model for the Oceans, Seas, and Estuaries* (NE/P012361/1).


# References
