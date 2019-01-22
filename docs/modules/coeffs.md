# Introduction

**pytzer.coeffs** contains functions for ionic interaction coefficients.

There are three different types of coefficient functions, each representing a different interaction type:

  1. `bC_` functions

  1. `theta_` functions

  1. `psi_` functions

## Syntax

The functions use the following formats. Within the function titles, **cation**, **anion** and **source** should be replaced with appropriate values. The input `T` is a numpy vstack of temperature values (in K), for example as produced by **pytzer.io.getmols**.

### `bC_` function syntax

```python
def bC_cation_anion_source(T):

    b0 = <beta0 value>
    b1 = <beta1 value>
    b2 = <beta2 value>
    C0 = <C0 value>
    C1 = <C1 value>

    alph1 = <alpha1 value>
    alph2 = <alpha2 value>
    omega = <omega value>

    valid = <logical indicating temperature validity>

    return b0,b1,b2,C0,C1, alph1,alph2,omega, valid
```

If the source has a Cphi value, instead of C0 and C1, conversion takes place within the bC_ function as follows:

```python
def bC_cation_anion_source(T):

    b0 = <beta0 value>
    b1 = <beta1 value>
    b2 = <beta2 value>

    Cphi = <Cphi value>

    zC = <cation charge>
    zA = <anion charge>
    C0 = Cphi / (2 * sqrt(abs(zC*zA)))

    C1 = <C1 value>

    alph1 = <alpha1 value>
    alph2 = <alpha2 value>
    omega = <omega value>

    valid = <logical indicating temperature validity>

    return b0,b1,b2,C0,C1, alph1,alph2,omega, valid
```

In either case, for missing b or C coefficients, the value should be set to `full_like(T)`, and the corresponding `alph2` or `omega` value set to -9.

### `theta_` function syntax

```python
def theta_ion0_ion1_source(T):

    theta = <theta value>
    valid = <logical indicating temperature validity>

    return theta, valid
```

### `psi_` function syntax

```python
def psi_ion0_ion1_ion2_source(T):

    psi   = <theta value>
    valid = <logical indicating temperature validity>

    return psi, valid
```
