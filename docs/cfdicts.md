title: Coefficient dictionaries

# Introduction

The `pytzer.cfdict` module provides specific combinations of coefficients that have been used in published Pitzer models.

To use a Pitzer model we need to define a set of coefficients that quantify the interactions between different combinations of ions. We do this by creating a `cfdict` (short for **coefficient dictionary**), which contains functions that evaluate the coefficients for every possible interaction. The functions themselves are defined in the separate module `pytzer.coeffs`.

# Pre-defined cfdicts

Several ready-to-use cfdicts are available in this module.

## GM89: Greenberg and Møller (1989)

Greenberg, J. P., and Møller, N. (1989). The prediction of mineral solubilities in natural waters: A chemical equilibrium model for the Na-K-Ca-Cl-SO<sub>4</sub>-H<sub>2</sub>O system to high concentration from 0 to 250°C. *Geochim. Cosmochim. Acta* 53, 2503–2518. <a href="https://doi.org/10.1016/0016-7037(89)90124-5">doi:10.1016/0016-7037(89)90124-5</a>.

### System

Ca - K - Na - Cl - SO<sub>4</sub> - H<sub>2</sub>O

### Validity range

  * **Temperature:** 0 °C to 250 °C


## M88: Møller (1988)

Møller, N. (1988). The prediction of mineral solubilities in natural waters: A chemical equilibrium model for the Na-Ca-Cl-SO<sub>4</sub>-H<sub>2</sub>O system, to high temperature and concentration. *Geochim. Cosmochim. Acta* 52, 821–837. <a href="https://doi.org/10.1016/0016-7037(88)90354-7">doi:10.1016/0016-7037(88)90354-7</a>.

### System

Ca - Na - Cl - SO<sub>4</sub> - H<sub>2</sub>O

### Validity range

  * **Temperature:** 25 °C to 250 °C

  * **Ionic strength:** 0 to ~18 mol·kg<sup>−1</sup>


# Modify an existing cfdict


# Create your own

You can also construct your own as follows. A `cfdict` is initialised using the `pytzer.cfdicts.CfDict` class. Functions from `pytzer.coeffs` are then added. For example, to generate a `cfdict` for the system Na-Ca-Cl using functions from Møller (1988), we would write:

```python
import pytzer as pz
import numpy as np

# Initialise
myCfdict = pz.cfdicts.CfDict()

# Debye-Hueckel limiting slope
myCfdict.dh['Aosm'] = coeffs.Aosm_M88

# Cation-anion interactions (betas and Cs)
myCfdict.bC['Ca-Cl' ] = coeffs.bC_Ca_Cl_M88
myCfdict.bC['Na-Cl' ] = coeffs.bC_Na_Cl_M88

# Cation-cation and anion-anion interactions (theta)
# c-c'
myCfdict.theta['Ca-Na' ] = coeffs.theta_Ca_Na_M88

# Unsymmetrical mixing functions
myCfdict.jfunc = jfuncs.Harvie

# Triplet interactions (psi)
# c-c'-a
myCfdict.psi['Ca-Na-Cl' ] = coeffs.psi_Ca_Na_Cl_M88

# Fill missing functions with zeros (none in this instance)
myCfdict.add_zeros(np.array(['Na','Ca','Cl']))
```

Any missing functions can be filled in with zeros using `CfDict.add_zeros()` at the end.

Two conventions must be followed for the strings that define which ions are involved in each interaction. In order of precedence they are:

  1. Cations before anions;

  1. Alphabetical order.
