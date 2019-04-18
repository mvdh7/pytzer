<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
MathJax.Ajax.config.path["mhchem"] =
  "https://cdnjs.cloudflare.com/ajax/libs/mathjax-mhchem/3.3.2";
MathJax.Hub.Config({TeX: {extensions: ["[mhchem]/mhchem.js"]}});
</script><script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML' async></script>

# Coefficient libraries

*The casual user has no need to explicitly call this module.*

`.cflibs` provides specific combinations of coefficients that have been used in published Pitzer models, to use with Pytzer.

To use a Pitzer model we need to define a set of coefficients that quantify the interactions between different combinations of ions. We do this by creating a **coefficient library**, which contains functions that evaluate the coefficients for every possible interaction. The functions themselves [are defined separately](../coeffs).

A number of [pre-defined coefficient libraries](#pre-defined-coefficient-libraries) are included in **pytzer.cflibs**. To use these, all you need to do is assign the variable `cflib` appropriately:

```python
>>> import pytzer as pz
>>> cflib = pz.cflibs.M88 # Use M88 coefficients
```

This `cflib` can then be passed into all of the [Pitzer model functions](../model).

<hr />

## Pre-defined coefficient libraries

Several ready-to-use coefficient libraries are available in this module. To decode their sources, see the [literature references table](../../references).

<table><tr>

<td><strong>cflib name</strong></td>
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

## cflib methods

A few handy methods are provided as part of the **CoeffLib** class. Brief summaries are provided below, and here is a usage example of all of them together:

```python
import pytzer as pz
import numpy as np
from copy import deepcopy

# Copy a pre-defined cflib
cflib = deepcopy(pz.cflibs.M88)

# Get ions within it
cflib.get_contents()

# Add a new ion into the mix
cflib.ions = np.append(cflib.ions, 'K')

# Add zero-functions for all interactions with the new ion
cflib.add_zeros(cflib.ions)

# Update cflib name to show we've changed it
cflib.name = 'M88-modified'

# Print out the coefficients evaluated at 298.15 K
cflib.print_coeffs(298.15, 'coeff_file.txt')
```

The methods are as follows:

### `.add_zeros` - add entries for missing interactions

`CoeffLib.add_zeros(ions)` adds zero-functions for all missing interactions, given a list of ions.

### `.get_contents` - create lists of ions and sources

`CoeffLib.get_contents()` scans through all functions within the **CoeffLib**, and puts lists of all ions and of all sources in its **ions** and **srcs** fields.

The list of ions is determined from the dict keys, while sources are determined from the function names.

### `.print_coeffs` - print out model coefficients

`CoeffLib.print_coeffs(T,filename)` evaluates all coefficients in a **cflib** at a single input temperature and pressure, and prints the results to a text file (`filename`).

<hr />

## How a CoeffLib works

To modify an existing **CoeffLib**, or create a new one, it is first necessary to understand how they are used within Pytzer, as follows. A basic understanding of the workings of the Pitzer model is assumed.

A **CoeffLib** or **cflib** is an object of the class `CoeffLib`. From the initalisation function we can see that it contains the following fields:

```python
class CoeffLib:

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

Each field is then filled with functions from modules **debyehueckel**, **coeffs** or **jfuncs** that define the Pitzer model interaction coefficients, as follows. (Descriptions of the required contents of the functions themselves are in the separate <a href="../coeffs"><strong>coeffs</strong> documentation</a>.)


### Debye-Hückel limiting slope

The function for the Debye-Hückel limiting slope (i.e. <i>A<sub>ϕ</sub></i>) is stored as `CoeffLib.dh['Aosm']`.


### Cation-anion interactions

Functions to evaluate the $\beta$ and $C$ coefficients for interactions between cations and anions are contained within the `cflib.bC` dict. The function for each specific interaction gets its own field within the dict. The fields are named as `<cation>-<anion>`, with the ionic names matching those described [for an input file](../io/#pytzeriogetmols) - see the page on [naming conventions](../../name-conventions) for a full description. Some examples:

```python
cflib.bC['Na-Cl'] = <Na-Cl interaction coefficients function>
cflib.bC['Mg-Cl'] = <Mg-Cl interaction coefficients function>
cflib.bC['K-SO4'] = <K-SO4 interaction coefficients function>
```

### Cation-cation and anion-anion interactions

Functions that evaluate the $\theta$ coefficients for interactions between ion pairs with a common charge sign are contained within the `cflib.theta` dict. The function for each specific interaction gets its own field within the dict. The fields are named as `<cation0>-<cation1>`, with the cations in alphabetical order, and with the ionic names matching those described [for an input file](../io/#pytzeriogetmols). Some examples:

```python
cflib.theta['Ca-Mg']  = <Ca-Mg interaction coefficients function>
cflib.theta['Mg-Na']  = <Mg-Na interaction coefficients function>
cflib.theta['Cl-SO4'] = <Cl-SO4 interaction coefficients function>
```

### Triplet interactions

Functions that evaluate the $\psi$ coefficients for interactions between ion pairs with a common charge sign and a third ion of opposite sign are contained within the `cflib.psi` dict. The function for each specific triplet interaction gets its own field within the dict. The fields are named as `<ion0>-<ion1>-<ion2>`, with the order of the ions obeying the following rules, given here in order of precedence:

  1. Cations before anions;

  1. In alphabetical order.

The ionic names should match those described [for an input file](../io/#pytzeriogetmols).

Some examples:

```python
cflib.psi['Ca-Mg-Cl']  = <Ca-Mg-Cl interaction coefficients function>
cflib.psi['Mg-Na-SO4'] = <Mg-Na-SO4 interaction coefficients function>
cflib.psi['Na-Cl-SO4'] = <Na-Cl-SO4 interaction coefficients function>
```

### Neutral interactions

Functions that evaluate the $\lambda$, $\zeta$ and $\mu$ coefficients for the interactions between a neutral solute and an ion or other neutral ($\lambda$), the three-way between a neutral, cation and anion ($\zeta$) and the three-way between three neutrals of the same kind ($\mu$) are contained within `cflib.lambd`, `cflib.eta` and `cflib.mu` respectively.

The field names obey the rules, in order of precedence:

  2. Neutrals first, then cations, then anions;

  2. In alphabetical order.

Assigning functions is exactly the same as described for the other interaction types.


### Unsymmetrical mixing terms

A function to evaluate the J and J' equations are contained in `cflib.jfunc`. Unlike the other fields within the **cflib**, only one function is provided, so this field directly contains the relevant function, rather than storing it in a dict.

Different options for the functions needed here can be found in **pytzer.jfuncs**.


<hr />


## Modify an existing cflib

The functions within an existing cflib can easily be switched by reassignment. For example, if you wanted to use the Møller (1988) model, but replace only the Na-Cl interaction equations with the model of Archer (1992), you could write:

```python
import pytzer as pz

# Get Møller (1988) cflib
cflib = pz.cflibs.M88

# Update Na-Cl interaction function to Archer (1992)
cflib.bC['Na-Cl'] = pz.coeffs.bC_Na_Cl_A92ii
```

Note that the statement to get the **cflib** (`cflib = pz.cflibs.M88`) only references, not copies, from **pytzer.cflibs**. To copy, and make changes without modifying the original, use:

```python
import pytzer as pz
from copy import deepcopy

# Get Møller (1988) cflib
cflib = deepcopy(pz.cflibs.M88)
cflib.name = 'M88-modified' # so we know it's been changed

# Update Na-Cl interaction function to Archer (1992)
cflib.bC['Na-Cl'] = pz.coeffs.bC_Na_Cl_A92ii
```

<hr />

## Build your own

You can also construct your own **cflib** from scratch. In the example below, we initialise a `cflib` using the `pytzer.cflibs.CoeffLib` class. We add functions from `pytzer.coeffs` for the system Na-Ca-Cl using functions from Møller (1988). Finally, we use the method `add_zeros` to fill out any interactions that we have neglected to provide functions for with zeros.

```python
import pytzer as pz
import numpy as np

# Initialise
mycflib = pz.cflibs.CoeffLib()

# Debye-Hueckel limiting slope
mycflib.dh['Aosm'] = coeffs.Aosm_M88

# Cation-anion interactions (betas and Cs)
mycflib.bC['Ca-Cl' ] = coeffs.bC_Ca_Cl_M88
mycflib.bC['Na-Cl' ] = coeffs.bC_Na_Cl_M88

# Cation-cation and anion-anion interactions (theta)
# c-c'
mycflib.theta['Ca-Na' ] = coeffs.theta_Ca_Na_M88

# Unsymmetrical mixing functions
mycflib.jfunc = jfuncs.Harvie

# Triplet interactions (psi)
# c-c'-a
mycflib.psi['Ca-Na-Cl' ] = coeffs.psi_Ca_Na_Cl_M88

# Fill missing functions with zeros (none in this instance)
mycflib.add_zeros(np.array(['Na','Ca','Cl']))
```

To explicitly assign zeros to any interaction (i.e. the interaction is ignored by the model), you can use the appropriate zero-functions from **pytzer.coeffs**:

```python
mycflib.bC['Ba-SO4']   = coeffs.bC_zero    # ignore Ba-SO4 interactions
mycflib.bC['H-Na']     = coeffs.theta_zero # ignore H-Na interactions
mycflib.psi['H-Mg-OH'] = coeffs.psi_zero   # ignore H-Mg-OH interactions
```

## Print out coefficients

You can use the function **CoeffLib.print_coeffs** to create a file containing every coefficient, evaluated at a single input temperature and pressure of your choice. For example:

```python
mycflib.print_coeffs(298.15,'myCoeffs.txt')
```

would evaluate every coefficient at 298.15 K and print the results to the file **myCoeffs.txt**.
