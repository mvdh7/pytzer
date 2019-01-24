# Introduction

**pytzer.coeffs** contains functions for ionic interaction coefficients.

There are six different types of coefficient functions, each representing a different interaction type. The function names begin with:

  1. `bC_`, for cation-anion interactions (*β*, *C*, *α* and *ω* coefficients);

  1. `theta_`, for interactions between pairs of same-charge ions (*θ*);

  1. `psi_`, for three-way interactions between a cation and two anions, or an anion and two cations (*ψ*);

  1. `lambd_`, for interactions between one neutral solute and one ion (*λ*);

  1. `eta_`, for interactions between one neutral solute, one cation and one anion (*η*); and

  1. `mu_`, for interactions between three neutral solutes of the same type (*μ*).

We do not provide a full list of the functions currently at this point (but would like to in the future).

A number of the functions in **pytzer.coeffs** are not used by any of the ready-made **cfdicts**.

## Philosophy

The main principle followed when constructing these functions is to maintain a **single source of truth**. This manifests itself in several ways.

Coefficient functions are only created for studies that report something new. If a new study reports coefficient equations or values but is simply copying them exactly from an older study, the newer study does not get a new function. Rather, when constructing a **CoefficientDictionary** to represent the newer study, the older study's function should be selected.

If a new study takes an older study's coefficients and modifies them in some way, this will be represented in the code: the function for the new study will call the older study's function to get the original values, and then modify them, rather than re-declaring the original values itself. See for example...

Where an equation is provided to evaluate coefficients, rather than a single value, this equation will be written out in its own function... See for example...

## Syntax

Within the function titles, **neutral**, **cation**, **anion** and **source** should be replaced with appropriate values, as described in the [naming conventions](../../name-conventions).

The input `T` is always a **numpy.vstack** of temperature values (in K), for example as output by **pytzer.io.getmols**.

Some of these functions are defined as a function of pressure as well as temperature. These components of the equations are retained within the functions, but for now, the standard atmospheric pressure (i.e. 0.101325 MPa) is used in every case, as defined by `COEFFS_PRESSURE` in **pytzer.coeffs**.

The outputs are the coefficient(s') value(s), and a logical array (`valid`) indicating whether the input temperature(s) fell within the function's validity range. *The logical array is not currently used.*

Output formats...

### `bC_` functions

The calculations in **pytzer.model** use a Pitzer model with separate *C*<sub>0</sub> and *C*<sub>1</sub> values, rather than a single *C*<sub><i>ϕ</i></sub>. Where coefficients are originally given in terms of *C*<sub><i>ϕ</i></sub>, this is converted into the equivalent *C*<sub>0</sub> value within the `bC_` function (and *C*<sub>0</sub> set to zero).

Each `bC_` function also returns values for the *α*<sub>1</sub>, *α*<sub>2</sub> and *ω* coefficients that accompany each *β*<sub>1</sub>, *β*<sub>2</sub> and *C*<sub>1</sub>. This is necessary because these auxiliary coefficients differ between electrolytes and between studies.

```python
def bC_cation_anion_source(T):

    b0 = <b0 equation/value>
    b1 = <b1 equation/value>
    b2 = <b2 equation/value>
    C0 = <C0 equation/value>
    C1 = <C1 equation/value>

    alph1 = <alph1 value>
    alph2 = <alph2 value>
    omega = <omega value>

    valid = <logical expression evaluating temperature validity>

    return b0,b1,b2,C0,C1, alph1,alph2,omega, valid
```

### Other functions

All the other coefficients do not have auxiliary parameters and consequently have much simpler functions:

```python
def coefficient_ion1_ion2[_ion3]_source(T):

    coeff = <coefficient equation/value>
    valid = <logical expression evaluating temperature validity>

    return coeff, valid
```
