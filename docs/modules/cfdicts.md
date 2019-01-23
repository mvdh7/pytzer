# Introduction

**pytzer.cfdicts** provides specific combinations of coefficients that have been used in published Pitzer models, to use with **pytzer**.

To use a Pitzer model we need to define a set of coefficients that quantify the interactions between different combinations of ions. We do this by creating a **cfdict** (short for **coefficient dictionary**), which contains functions that evaluate the coefficients for every possible interaction. The functions themselves are defined separately in **pytzer.coeffs**.

A number of [pre-defined cfdicts](#pre-defined-cfdicts) are included in **pytzer.cfdicts**. If you want to use of these, all you need to do is assign the variable `cfdict` appropriately:

```python
import pytzer as pz

# Use M88 coefficients
cfdict = pz.cfdicts.M88
```

This `cfdict` can then be passed into all of the **pytzer.model** functions.

<hr />

# Pre-defined cfdicts

Several ready-to-use **cfdicts** are available in this module.

## GM89: Greenberg and Møller (1989)

 **Source:** Greenberg, J. P., and Møller, N. (1989). The prediction of mineral solubilities in natural waters: A chemical equilibrium model for the Na-K-Ca-Cl-SO<sub>4</sub>-H<sub>2</sub>O system to high concentration from 0 to 250°C. *Geochim. Cosmochim. Acta* 53, 2503–2518. <a href="https://doi.org/10.1016/0016-7037(89)90124-5">doi:10.1016/0016-7037(89)90124-5</a>.

**System:** Ca - K - Na - Cl - SO<sub>4</sub>

**Validity:** *temperature* from 0 °C to 250 °C


## M88: Møller (1988)

**Source:** Møller, N. (1988). The prediction of mineral solubilities in natural waters: A chemical equilibrium model for the Na-Ca-Cl-SO<sub>4</sub>-H<sub>2</sub>O system, to high temperature and concentration. *Geochim. Cosmochim. Acta* 52, 821–837. <a href="https://doi.org/10.1016/0016-7037(88)90354-7">doi:10.1016/0016-7037(88)90354-7</a>.

**System:** Ca - Na - Cl - SO<sub>4</sub>

**Validity:** *temperature* from 25 °C to 250 °C; *ionic strength* from 0 to ~18 mol·kg<sup>−1</sup>

<hr />

# How cfdicts work

To modify an existing **cfdict**, or create a new one, it is first necessary to understand how they are used within **pytzer**, as follows. A basic understanding of the workings of the Pitzer model is assumed.

A **cfdict** is an object of the class `CoefficientDictionary` as defined within **pytzer.cfdicts**. From the initalisation function we can see that it contains the following fields:

```python
class CoefficientDictionary:

    # Initialise
    def __init__(self):
        self.name  = ''
        self.dh    = {} # Aosm
        self.bC    = {} # c-a
        self.theta = {} # c-c' and a-a'
        self.jfunc = [] # unsymmetrical mixing
        self.psi   = {} # c-c'-a and c-a-a'
        self.lambd = {} # n-c and n-a
        self.eta   = {} # n-c-a
        self.mu    = {} # n-n-n
```

Each field is then filled with functions from **pytzer.coeffs** that define the Pitzer model interaction coefficients, as follows. (Descriptions of the required contents of the functions themselves are in the separate <a href="../coeffs"><strong>pytzer.coeffs</strong> documentation</a>.)


### Debye-Hückel limiting slope

The function for the Debye-Hückel limiting slope (i.e. <i>A<sub>ϕ</sub></i>) is stored as `CoefficientDictionary.dh['Aosm']`.

### Cation-anion interactions

Functions to evaluate the *β* and *C* coefficients for interactions between cations and anions are contained within the `cfdict.bC` dict. The function for each specific interaction gets its own field within the dict. The fields are named as `<cation>-<anion>`, with the ionic names matching those described [for an input file](../io/#pytzeriogetmols) - see the page on [naming conventions](../../name-conventions) for a full description. Some examples:

```python
cfdict.bC['Na-Cl'] = <Na-Cl interaction coefficients function>
cfdict.bC['Mg-Cl'] = <Mg-Cl interaction coefficients function>
cfdict.bC['K-SO4'] = <K-SO4 interaction coefficients function>
```

### Cation-cation and anion-anion interactions

Functions that evaluate the *θ* coefficients for interactions between ion pairs with a common charge sign are contained within the `cfdict.theta` dict. The function for each specific interaction gets its own field within the dict. The fields are named as `<cation0>-<cation1>`, with the cations in alphabetical order, and with the ionic names matching those described [for an input file](../io/#pytzeriogetmols). Some examples:

```python
cfdict.theta['Ca-Mg']  = <Ca-Mg interaction coefficients function>
cfdict.theta['Mg-Na']  = <Mg-Na interaction coefficients function>
cfdict.theta['Cl-SO4'] = <Cl-SO4 interaction coefficients function>
```

### Triplet interactions

Functions that evaluate the *ψ* coefficients for interactions between ion pairs with a common charge sign and a third ion of opposite sign are contained within the `cfdict.psi` dict. The function for each specific triplet interaction gets its own field within the dict. The fields are named as `<ion0>-<ion1>-<ion2>`, with the order of the ions obeying the following rules, given here in order of precedence:

  1. Cations before anions;

  1. In alphabetical order.

The ionic names should match those described [for an input file](../io/#pytzeriogetmols).

Some examples:

```python
cfdict.psi['Ca-Mg-Cl']  = <Ca-Mg-Cl interaction coefficients function>
cfdict.psi['Mg-Na-SO4'] = <Mg-Na-SO4 interaction coefficients function>
cfdict.psi['Na-Cl-SO4'] = <Na-Cl-SO4 interaction coefficients function>
```


### Neutral interactions

Functions that evaluate the *λ*, *η* and *μ* coefficients for the interactions between a neutral solute and an ion (*λ*), the three-way between a neutral, cation and anion (*η*) and the three-way between three neutrals of the same kind (*μ*) are contained within `cfdict.lambd`, `cfdict.eta` and `cfdict.mu` respectively.

The field names obey the rules, in order of precedence:

  2. Neutrals first, then cations, then anions;

  2. In alphabetical order.

Assigning functions is exactly the same as described for the other interaction types.

### Unsymmetrical mixing terms

A function to evaluate the J and J' equations are contained in `cfdict.jfunc`. Unlike the other fields within the **cfdict**, only one function is provided, so this field directly contains the relevant function, rather than storing it in a dict.

<hr />

# Modify an existing cfdict

The functions within an existing cfdict can easily be switched by reassignment. For example, if you wanted to use the Møller (1988) model, but replace only the Na-Cl interaction equations with the model of Archer (1992), you could write:

```python
import pytzer as pz

# Get Møller (1988) cfdict
cfdict = pz.cfdicts.M88

# Update Na-Cl interaction function to Archer (1992)
cfdict.bC['Na-Cl'] = pz.coeffs.bC_Na_Cl_A92ii
```

Note that the statement to get the **cfdict** (`cfdict = pz.cfdicts.M88`) only references, not copies, from **pytzer.cfdicts**. To copy, and make changes without modifying the original, use:

```python
import pytzer as pz
from copy import deepcopy

# Get Møller (1988) cfdict
cfdict = deepcopy(pz.cfdicts.M88)
cfdict.name = 'M88-modified' # so we know it's been changed

# Update Na-Cl interaction function to Archer (1992)
cfdict.bC['Na-Cl'] = pz.coeffs.bC_Na_Cl_A92ii
```

<hr />

# Build your own

You can also construct your own **cfdict** from scratch. In the example below, we initialise a `cfdict` using the `pytzer.cfdicts.CoefficientDictionary` class. We add functions from `pytzer.coeffs` for the system Na-Ca-Cl using functions from Møller (1988). Finally, we use the method `add_zeros` to fill out any interactions that we have neglected to provide functions for with zeros.

```python
import pytzer as pz
import numpy as np

# Initialise
myCfdict = pz.cfdicts.CoefficientDictionary()

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

To explicitly assign zeros to any interaction (i.e. the interaction is ignored by the model), you can use the appropriate zero-functions from **pytzer.coeffs**:

```python
myCfdict.bC['Ba-SO4']   = coeffs.bC_zero    # ignore Ba-SO4 interactions
myCfdict.bC['H-Na']     = coeffs.theta_zero # ignore H-Na interactions
myCfdict.psi['H-Mg-OH'] = coeffs.psi_zero   # ignore H-Mg-OH interactions
```

## Print out coefficients

You can use the method **print_coeffs** on a **cfdict** to create a file containing every coefficient, evaluated at a single input temperature of your choice. For example:

```python
myCfdict.print_coeffs(298.15,'myCoeffs.txt')
```

would evaluate every coefficient at 298.15 K and print the results to the file **myCoeffs.txt**.
