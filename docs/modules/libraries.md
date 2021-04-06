# Parameter libraries

*The casual user has no need to explicitly call this module.*

`.libraries` provides specific combinations of parameters that have been used in published Pitzer models, to use with Pytzer.

To use a Pitzer model we need to define a set of parameters that quantify the interactions between different combinations of ions. We do this by creating a **parameter library**, which contains functions that evaluate the parameters for every possible interaction. The functions themselves [are defined separately](../parameters).

A number of [pre-defined parameter libraries](#pre-defined-parameter-libraries) are included in this module. To use these, all you need to do is assign the variable `prmlib` appropriately:

```python
import pytzer as pz
prmlib = pz.libraries.M88 # Use M88 parameters
```

This `prmlib` can then be passed into all of the [Pitzer model functions](../model).

---

## Pre-defined parameter libraries

Several ready-to-use parameter libraries are available in this module. To decode their sources, see the [literature references table](../../refs).

<table><tr>

<td><strong>prmlib name</strong></td>
<td><strong>System</strong></td>
<td><strong>Source</strong></td>

</tr><tr>
<td>CRP94</td>
<td>$\ce{H^+}$, $\ce{HSO4^-}$, $\ce{SO4^2-}$</td>
<td><a href="../../refs/#CRP94">CRP94</a></td>

</tr><tr>
<td>GM89</td>
<td>$\ce{Ca^2+}$, $\ce{Cl^-}$, $\ce{K^+}$, $\ce{Na^+}$, $\ce{SO4^2-}$</td>
<td><a href="../../refs/#GM89">GM89</a></td>

</tr><tr>
<td>M88</td>
<td>$\ce{Ca^2+}$, $\ce{Cl^-}$, $\ce{Na^+}$, $\ce{SO4^2-}$</td>
<td><a href="../../refs/#M88">M88</a></td>

</tr><tr>
<td>WM13</td>
<td>$\ce{Ca^2+}$, $\ce{Cl-}$, $\ce{H+}$, $\ce{HSO4-}$, $\ce{K+}$, $\ce{Mg^2+}$, $\ce{MgOH+}$, $\ce{Na+}$, $\ce{OH-}$, $\ce{SO4^2-}$</td>
<td><a href="../../refs/#WM13">WM13</a></td>

</tr></table>

---

## Parameter library methods

A few handy methods are provided as part of the `ParameterLibrary` class. Brief summaries are provided below, and here is a usage example of all of them together:

```python
import pytzer as pz
import numpy as np
from copy import deepcopy

prmlib = deepcopy(pz.libraries.M88) # copy a pre-defined prmlib
prmlib.get_contents() # get ions within it
prmlib.ions = np.append(prmlib.ions, 'K') # add a new ion into the mix
# Add zero-functions for interactions with the new ion:
prmlib.add_zeros(prmlib.ions)
prmlib.name = 'M88-modified' # update prmlib name to show we've changed it
# Print out the parameters evaluated at 298.15 K and 10.1325 dbar:
prmlib.print_parameters(298.15, 10.1325, 'coeff_file.txt')
```

The methods are as follows:

### `.add_zeros` - add entries for missing interactions

`ParameterLibrary.add_zeros(ions)` adds zero-functions for all missing interactions, given a list of ions.

### `.get_contents` - create lists of ions and sources

`ParameterLibrary.get_contents()` scans through all functions within the parameter library, and puts lists of all ions and of all sources in its `ions` and `srcs` fields.

The list of ions is determined from the dict keys, while sources are determined from the function names.

### `.print_parameters` - print out model parameters

`ParameterLibrary.print_parameters(tempK, pres, filename)` evaluates all parameters in a parameter library at a single input temperature and pressure, and prints the results to a text file called `filename`.

---

## How a parameter library works

To modify an existing parameter library, or create a new one, it is first necessary to understand how they are used within Pytzer, as follows. A basic understanding of the workings of the Pitzer model is assumed.

A parameter library, or `prmlib`, is an object of the class `ParameterLibrary`. From the initalisation function we can see that it contains the following fields:

```python
class ParameterLibrary:
    def __init__(self):
        self.name  = ''
        self.dh    = {} # Aosm
        self.bC    = {} # c-a
        self.theta = {} # c-c' and a-a'
        self.jfunc = [] # unsymmetrical mixing
        self.psi   = {} # c-c'-a and c-a-a'
        self.lambd = {} # n-c, n-a, n-n and n-n'
        self.zeta  = {} # n-c-a
        self.mu    = {} # n-n-n
        self.lnk   = {} # thermodynamic equilibrium constants
        self.ions  = []
        self.srcs  = []
```

Each field is then filled with functions from modules [debyehueckel](../debyehueckel), [parameters](../parameters) or [unsymmetrical](../unsymmetrical) that define the Pitzer model interaction parameters, and [dissociation](../dissociation) for thermodynamic equilibrium constants, as follows. *Descriptions of the required contents of the functions themselves are in the separate [parameters](../parameters) documentation.*


### Debye-Hückel limiting slope

The function for the Debye-Hückel limiting slope (i.e. <i>A<sub>ϕ</sub></i>) is stored as `ParameterLibrary.dh['Aosm']`. Several functions are available in the [debyehueckel](../debyehueckel) module.

### Cation-anion interactions

Functions to evaluate the $\beta$ and $C$ parameters for interactions between cations and anions are contained within the `prmlib.bC` dict. The function for each specific interaction gets its own field within the dict. The fields are named as `<cation>-<anion>`, with the ionic names matching those described [for an input file](../io/#pytzeriogetmols) - see the [naming conventions](../../name-conventions) for a full description. **Some examples:**

```python
prmlib.bC['Na-Cl'] = pz.parameters.bC_Na_Cl_A92ii
prmlib.bC['Mg-Cl'] = pz.parameters.bC_Mg_Cl_PP87i
prmlib.bC['K-SO4'] = pz.parameters.bC_K_SO4_HM86
```

### Cation-cation and anion-anion interactions

Functions that evaluate the $\theta$ parameters for interactions between ion pairs with a common charge sign are contained within the `prmlib.theta` dict. The function for each specific interaction gets its own field within the dict. The fields are named as `<cation0>-<cation1>`, with the cations in alphabetical order, and with the ionic names matching those described [for an input file](../io/#pytzeriogetmols). **Some examples:**

```python
prmlib.theta['Ca-Mg'] = pz.parameters.theta_Ca_Mg_HMW84
prmlib.theta['Mg-Na'] = pz.parameters.theta_Mg_Na_HMW84
prmlib.theta['Cl-SO4'] = pz.parameters.theta_Cl_SO4_M88
```

### Triplet interactions

Functions that evaluate the $\psi$ parameters for interactions between ion pairs with a common charge sign and a third ion of opposite sign are contained within the `prmlib.psi` dict. The function for each specific triplet interaction gets its own field within the dict. The fields are named as `<ion0>-<ion1>-<ion2>`, with the order of the ions obeying the following rules, given here in order of precedence:

  1. Cations before anions;

  2. In alphabetical order.

The ionic names should match those described [for an input file](../io/#pytzeriogetmols).

**Some examples:**

```python
prmlib.psi['Ca-Mg-Cl'] = pz.parameters.psi_Ca_Mg_Cl_HMW84
prmlib.psi['Mg-Na-SO4'] = pz.parameters.psi_Mg_Na_SO4_HMW84
prmlib.psi['Na-Cl-SO4'] = pz.parameters.psi_Na_Cl_SO4_M88
```

### Neutral interactions

Functions that evaluate the $\lambda$, $\zeta$ and $\mu$ parameters for the interactions between a neutral solute and an ion or other neutral ($\lambda$), the three-way between a neutral, cation and anion ($\zeta$) and the three-way between three neutrals of the same kind ($\mu$) are contained within `prmlib.lambd`, `prmlib.zeta` and `prmlib.mu` respectively.

The field names obey the rules, in order of precedence:

  1. Neutrals first, then cations, then anions;

  2. In alphabetical order.

Assigning functions is exactly the same as described for the other interaction types.

**Some examples:**

```python
prmlib.lambd['tris-trisH'] = prm.lambd_tris_trisH_GT17simopt
prmlib.lambd['tris-Na'] = prm.lambd_tris_Na_GT17simopt
prmlib.lambd['tris-tris'] = prm.lambd_tris_tris_MarChemSpec25
prmlib.zeta['tris-Na-Cl'] = prm.zeta_tris_Na_Cl_MarChemSpec25
prmlib.mu['tris-tris-tris'] = prm.mu_tris_tris_tris_MarChemSpec25
```

### Unsymmetrical mixing terms

A function to evaluate the J and J' equations are contained in `prmlib.jfunc`. Unlike the other fields within the parameter library, only one function is provided, so this field directly contains the relevant function, rather than storing it in a dict.

Different options for the functions needed here can be found in the [unsymmetrical](../unsymmetrical) module.

**Example:**

```python
prmlib.plname.assign_func_J(unsymmetrical.Harvie)
```

### Thermodynamic equilibrium constants

Functions for the thermodynamic equilibrium constants are contined in `prmlib.lnk`. The functions themselves are declared in the [dissociation](../dissociation) module. The field names are equivalent to the `EQ` component of the function names.

**Some examples:**

```python
prmlib.lnk['H2O'] = pz.dissociation.H2O_MF
prmlib.lnk['HSO4'] = pz.dissociation.HSO4_CRP94
prmlib.lnk['Mg'] = pz.dissociation.Mg_CW91
prmlib.lnk['trisH'] = pz.dissociation.trisH_BH64
```

---

## Modify an existing prmlib

The functions within an existing parameter library can easily be switched by reassignment. For example, if we wanted to use the Møller (1988) model, but replace only the Na-Cl interaction equations with the model of Archer (1992), we could write:

```python
import pytzer as pz
from copy import deepcopy

prmlib = deepcopy(pz.libraries.M88)
prmlib.name = 'M88-modified' # so we know it's been changed
prmlib.bC['Na-Cl'] = pz.parameters.bC_Na_Cl_A92ii
```

We use `deepcopy` else changes made to the parameter library will also apply to the version stored in the [libraries](../libraries) module.

---

## Build your own

You can also construct your own parameter library from scratch. In the example below, we initialise using the `ParameterLibrary` class defined in the [libraries](module). We add functions from the [parameters](../parameters) module for the system Na-Ca-Cl using functions from Møller (1988). Finally, we use the parameter library method `add_zeros` to fill out any interactions that we have neglected to provide functions for with zeros.

```python
import pytzer as pz
import numpy as np

myprmlib = pz.libraries.ParameterLibrary()
myprmlib.dh['Aosm'] = parameters.Aosm_M88
myprmlib.bC['Ca-Cl' ] = parameters.bC_Ca_Cl_M88
myprmlib.bC['Na-Cl' ] = parameters.bC_Na_Cl_M88
myprmlib.theta['Ca-Na' ] = parameters.theta_Ca_Na_M88
myprmlib.psi['Ca-Na-Cl' ] = parameters.psi_Ca_Na_Cl_M88
myprmlib.plname.assign_func_J(unsymmetrical.Harvie)
myprmlib.add_zeros(np.array(['Na','Ca','Cl']))
```

To explicitly assign zeros to any interaction (i.e. the interaction is ignored by the model), you can use the appropriate zero-functions from the [parameters](../parameters) module:

```python
myprmlib.bC['Ba-SO4'] = parameters.bC_zero # ignore Ba-SO4 interactions
myprmlib.bC['H-Na'] = parameters.theta_zero # ignore H-Na interactions
myprmlib.psi['H-Mg-OH'] = parameters.psi_zero # ignore H-Mg-OH interactions
```

## Print out parameters

You can use the parameter library method `print_parameters` to create a file containing every parameter, evaluated at a single input temperature and pressure of your choice. For example:

```python
myprmlib.print_parameters(298.15, 10.1325, 'myparameters.txt')
```

would evaluate every parameter at 298.15 K and 10.1325 dbar (i.e. 1 atm) and print the results to **myparameters.txt**.
