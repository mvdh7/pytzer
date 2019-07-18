<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
MathJax.Ajax.config.path["mhchem"] =
  "https://cdnjs.cloudflare.com/ajax/libs/mathjax-mhchem/3.3.2";
MathJax.Hub.Config({TeX: {extensions: ["[mhchem]/mhchem.js"]}});
</script><script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML' async></script>

# Parameter libraries

*The casual user has no need to explicitly call this module.*

`.libraries` provides specific combinations of parameters that have been used in published Pitzer models, to use with Pytzer.

To use a Pitzer model we need to define a set of parameters that quantify the interactions between different combinations of ions. We do this by creating a **parameter library**, which contains functions that evaluate the parameters for every possible interaction. The functions themselves [are defined separately](../parameters).

A number of [pre-defined parameter libraries](#pre-defined-parameter-libraries) are included in **pytzer.libraries**. To use these, all you need to do is assign the variable `prmlib` appropriately:

```python
>>> import pytzer as pz
>>> prmlib = pz.libraries.M88 # Use M88 parameters
```

This `prmlib` can then be passed into all of the [Pitzer model functions](../model).

<hr />

## Pre-defined parameter libraries

Several ready-to-use parameter libraries are available in this module. To decode their sources, see the [literature references table](../../references).

<table><tr>

<td><strong>prmlib name</strong></td>
<td><strong>System</strong></td>
<td><strong>Source</strong></td>

</tr><tr>
<td>CRP94</td>
<td>$\ce{H^+}$, $\ce{HSO4^-}$, $\ce{SO4^2-}$</td>
<td><a href="../../references/#CRP94">CRP94</a></td>

</tr><tr>
<td>GM89</td>
<td>$\ce{Ca^2+}$, $\ce{Cl^-}$, $\ce{K^+}$, $\ce{Na^+}$, $\ce{SO4^2-}$</td>
<td><a href="../../references/#GM89">GM89</a></td>

</tr><tr>
<td>M88</td>
<td>$\ce{Ca^2+}$, $\ce{Cl^-}$, $\ce{Na^+}$, $\ce{SO4^2-}$</td>
<td><a href="../../references/#M88">M88</a></td>

</tr><tr>
<td>WM13</td>
<td>$\ce{Ca^2+}$, $\ce{Cl-}$, $\ce{H+}$, $\ce{HSO4-}$, $\ce{K+}$, $\ce{Mg^2+}$, $\ce{MgOH+}$, $\ce{Na+}$, $\ce{OH-}$, $\ce{SO4^2-}$</td>
<td><a href="../../references/#WM13">WM13</a></td>

</tr></table>

<hr />

## prmlib methods

A few handy methods are provided as part of the **CoefficientLibrary** class. Brief summaries are provided below, and here is a usage example of all of them together:

```python
import pytzer as pz
import numpy as np
from copy import deepcopy

# Copy a pre-defined prmlib
prmlib = deepcopy(pz.libraries.M88)

# Get ions within it
prmlib.get_contents()

# Add a new ion into the mix
prmlib.ions = np.append(prmlib.ions, 'K')

# Add zero-functions for all interactions with the new ion
prmlib.add_zeros(prmlib.ions)

# Update prmlib name to show we've changed it
prmlib.name = 'M88-modified'

# Print out the parameters evaluated at 298.15 K
prmlib.print_parameters(298.15, 'coeff_file.txt')
```

The methods are as follows:

### `.add_zeros` - add entries for missing interactions

`CoefficientLibrary.add_zeros(ions)` adds zero-functions for all missing interactions, given a list of ions.

### `.get_contents` - create lists of ions and sources

`CoefficientLibrary.get_contents()` scans through all functions within the **CoefficientLibrary**, and puts lists of all ions and of all sources in its **ions** and **srcs** fields.

The list of ions is determined from the dict keys, while sources are determined from the function names.

### `.print_parameters` - print out model parameters

`CoefficientLibrary.print_parameters(T,filename)` evaluates all parameters in a **prmlib** at a single input temperature and pressure, and prints the results to a text file (`filename`).

<hr />

## How a CoefficientLibrary works

To modify an existing **CoefficientLibrary**, or create a new one, it is first necessary to understand how they are used within Pytzer, as follows. A basic understanding of the workings of the Pitzer model is assumed.

A **CoefficientLibrary** or **prmlib** is an object of the class `CoefficientLibrary`. From the initalisation function we can see that it contains the following fields:

```python
class CoefficientLibrary:

    # Initialise
    def __init__(self):
        self.name  = ''
        self.dh    = {} # Aosm
        self.bC    = {} # c-a
        self.theta = {} # c-c' and a-a'
        self.jfunc = [] # unsymmetrical mixing
        self.psi   = {} # c-c'-a and c-a-a'
        self.lambd = {} # n-c and n-a
        self.zeta  = {} # n-c-a
        self.mu    = {} # n-n-n
        self.ions  = []
        self.srcs  = []
```

Each field is then filled with functions from modules **debyehueckel**, **parameters** or **unsymmetrical** that define the Pitzer model interaction parameters, as follows. (Descriptions of the required contents of the functions themselves are in the separate <a href="../parameters"><strong>parameters</strong> documentation</a>.)


### Debye-Hückel limiting slope

The function for the Debye-Hückel limiting slope (i.e. <i>A<sub>ϕ</sub></i>) is stored as `CoefficientLibrary.dh['Aosm']`.


### Cation-anion interactions

Functions to evaluate the $\beta$ and $C$ parameters for interactions between cations and anions are contained within the `prmlib.bC` dict. The function for each specific interaction gets its own field within the dict. The fields are named as `<cation>-<anion>`, with the ionic names matching those described [for an input file](../io/#pytzeriogetmols) - see the page on [naming conventions](../../name-conventions) for a full description. Some examples:

```python
prmlib.bC['Na-Cl'] = <Na-Cl interaction parameters function>
prmlib.bC['Mg-Cl'] = <Mg-Cl interaction parameters function>
prmlib.bC['K-SO4'] = <K-SO4 interaction parameters function>
```

### Cation-cation and anion-anion interactions

Functions that evaluate the $\theta$ parameters for interactions between ion pairs with a common charge sign are contained within the `prmlib.theta` dict. The function for each specific interaction gets its own field within the dict. The fields are named as `<cation0>-<cation1>`, with the cations in alphabetical order, and with the ionic names matching those described [for an input file](../io/#pytzeriogetmols). Some examples:

```python
prmlib.theta['Ca-Mg']  = <Ca-Mg interaction parameters function>
prmlib.theta['Mg-Na']  = <Mg-Na interaction parameters function>
prmlib.theta['Cl-SO4'] = <Cl-SO4 interaction parameters function>
```

### Triplet interactions

Functions that evaluate the $\psi$ parameters for interactions between ion pairs with a common charge sign and a third ion of opposite sign are contained within the `prmlib.psi` dict. The function for each specific triplet interaction gets its own field within the dict. The fields are named as `<ion0>-<ion1>-<ion2>`, with the order of the ions obeying the following rules, given here in order of precedence:

  1. Cations before anions;

  1. In alphabetical order.

The ionic names should match those described [for an input file](../io/#pytzeriogetmols).

Some examples:

```python
prmlib.psi['Ca-Mg-Cl']  = <Ca-Mg-Cl interaction parameters function>
prmlib.psi['Mg-Na-SO4'] = <Mg-Na-SO4 interaction parameters function>
prmlib.psi['Na-Cl-SO4'] = <Na-Cl-SO4 interaction parameters function>
```

### Neutral interactions

Functions that evaluate the $\lambda$, $\zeta$ and $\mu$ parameters for the interactions between a neutral solute and an ion or other neutral ($\lambda$), the three-way between a neutral, cation and anion ($\zeta$) and the three-way between three neutrals of the same kind ($\mu$) are contained within `prmlib.lambd`, `prmlib.eta` and `prmlib.mu` respectively.

The field names obey the rules, in order of precedence:

  2. Neutrals first, then cations, then anions;

  2. In alphabetical order.

Assigning functions is exactly the same as described for the other interaction types.


### Unsymmetrical mixing terms

A function to evaluate the J and J' equations are contained in `prmlib.jfunc`. Unlike the other fields within the **prmlib**, only one function is provided, so this field directly contains the relevant function, rather than storing it in a dict.

Different options for the functions needed here can be found in **pytzer.unsymmetrical**.


<hr />


## Modify an existing prmlib

The functions within an existing prmlib can easily be switched by reassignment. For example, if you wanted to use the Møller (1988) model, but replace only the Na-Cl interaction equations with the model of Archer (1992), you could write:

```python
import pytzer as pz

# Get Møller (1988) prmlib
prmlib = pz.libraries.M88

# Update Na-Cl interaction function to Archer (1992)
prmlib.bC['Na-Cl'] = pz.parameters.bC_Na_Cl_A92ii
```

Note that the statement to get the **prmlib** (`prmlib = pz.libraries.M88`) only references, not copies, from **pytzer.libraries**. To copy, and make changes without modifying the original, use:

```python
import pytzer as pz
from copy import deepcopy

# Get Møller (1988) prmlib
prmlib = deepcopy(pz.libraries.M88)
prmlib.name = 'M88-modified' # so we know it's been changed

# Update Na-Cl interaction function to Archer (1992)
prmlib.bC['Na-Cl'] = pz.parameters.bC_Na_Cl_A92ii
```

<hr />

## Build your own

You can also construct your own **prmlib** from scratch. In the example below, we initialise a `prmlib` using the `pytzer.libraries.CoefficientLibrary` class. We add functions from `pytzer.parameters` for the system Na-Ca-Cl using functions from Møller (1988). Finally, we use the method `add_zeros` to fill out any interactions that we have neglected to provide functions for with zeros.

```python
import pytzer as pz
import numpy as np

# Initialise
myprmlib = pz.libraries.CoefficientLibrary()

# Debye-Hueckel limiting slope
myprmlib.dh['Aosm'] = parameters.Aosm_M88

# Cation-anion interactions (betas and Cs)
myprmlib.bC['Ca-Cl' ] = parameters.bC_Ca_Cl_M88
myprmlib.bC['Na-Cl' ] = parameters.bC_Na_Cl_M88

# Cation-cation and anion-anion interactions (theta)
# c-c'
myprmlib.theta['Ca-Na' ] = parameters.theta_Ca_Na_M88

# Unsymmetrical mixing functions
myprmlib.jfunc = unsymmetrical.Harvie

# Triplet interactions (psi)
# c-c'-a
myprmlib.psi['Ca-Na-Cl' ] = parameters.psi_Ca_Na_Cl_M88

# Fill missing functions with zeros (none in this instance)
myprmlib.add_zeros(np.array(['Na','Ca','Cl']))
```

To explicitly assign zeros to any interaction (i.e. the interaction is ignored by the model), you can use the appropriate zero-functions from **pytzer.parameters**:

```python
myprmlib.bC['Ba-SO4']   = parameters.bC_zero    # ignore Ba-SO4 interactions
myprmlib.bC['H-Na']     = parameters.theta_zero # ignore H-Na interactions
myprmlib.psi['H-Mg-OH'] = parameters.psi_zero   # ignore H-Mg-OH interactions
```

## Print out parameters

You can use the function **CoefficientLibrary.print_parameters** to create a file containing every parameter, evaluated at a single input temperature and pressure of your choice. For example:

```python
myprmlib.print_parameters(298.15,'myparameters.txt')
```

would evaluate every parameter at 298.15 K and print the results to the file **myparameters.txt**.
