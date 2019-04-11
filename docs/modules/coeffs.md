<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML' async></script>

# Coefficient functions

**pytzer.coeffs** contains functions for ionic interaction coefficients.

There are six different types of coefficient functions, each representing a different interaction type. The function names begin with:

  1. `bC_`, for cation-anion interactions ($\beta$, $C$, $\alpha$ and $\omega$ coefficients);

  1. `theta_`, for interactions between pairs of same-charge ions ($\theta$);

  1. `psi_`, for three-way interactions between a cation and two anions, or an anion and two cations ($\psi$);

  1. `lambd_`, for interactions between one neutral solute and either one ion or another neutral ($\lambda$);

  1. `zeta_`, for interactions between one neutral solute, one cation and one anion ($\zeta$); and

  1. `mu_`, for interactions between three neutral solutes of the same type ($\mu$).

There is not yet a full list of all the functions available. A number of them are not used by any ready-made **cflib**.


## Philosophy

The main principle followed when constructing these functions is to maintain a **single source of truth**. This manifests itself in several ways.

Coefficient functions are only created for studies that report something new. If a new study reports coefficient equations or values but is simply copying them exactly from an older study, the newer study does not get a new function. Rather, when constructing a **cflib** to represent the newer study, the older study's function should be selected.

If a new study takes an older study's coefficients and modifies them in some way, this will be represented in the code: the function for the new study will call the older study's function to get the original values, and then modify them, rather than re-declaring the original values itself.

Where an equation is provided to evaluate coefficients, rather than a single value, this equation will be written out once only, in its own function. All of the coefficient functions that use this equation will then simply pass the relevant coefficients into that function.

**pytzer** takes these steps because the literature is littered with typos. Papers that compile coefficients from multiple sources into a complex model are particularly problematic. Our approach should minimise the opportunity for these errors to occur, and better still, once an error is detected, it will only ever need to be fixed in one place. When we *do* identify a specific typo in a study, this will be noted in comments within the coefficient's function, and adjust the function to use the correct value, if it can be found. We keep a separate summary of all of these corrections, which will be published for reference in due course.


## Syntax

Within the function titles, **neutral**, **cation**, **anion** and **source** should be replaced with appropriate values, as described in the [naming conventions](../../name-conventions).

The input `T` is equivalent to `tempK`, and `P` to `pres`, elsewhere in **pytzer**: these are always arrays of temperature (in K) and pressure (in dbar), for example as output by **io.getmols**. The shorter `T` and `P` preferred here for clarity in the equations.

The outputs are the coefficient(s') value(s), and a logical array (`valid`) indicating whether the input temperature(s) fell within the function's validity range. *The validity array is not currently used.*

Temperature- or pressure-sensitive output coefficient variables each have the same shape as the temperature and pressure inputs, while those with constant values are returned as scalars.


### `bC_` functions

The calculations in **pytzer.model** use a Pitzer model with separate $C_0$ and $C_1$ values, rather than a single $C_\phi$. Where coefficients are originally given in terms of $C_\phi$, this is converted into the equivalent $C_0$ value within the `bC_` function (and $C_1$ set to zero).

Each `bC_` function also returns values for the $\alpha_1$, $\alpha_2$ and $\omega$ coefficients that accompany each $\beta_1$, $\beta_2$ and $C_1$. This is necessary because these auxiliary coefficients differ between electrolytes and between studies.

```python
def bC_cation_anion_source(T, P):
    b0 = <b0 equation/value>
    b1 = <b1 equation/value>
    b2 = <b2 equation/value>
    C0 = <C0 equation/value>
    C1 = <C1 equation/value>
    alph1 = <alph1 value>
    alph2 = <alph2 value>
    omega = <omega value>
    valid = <logical of temperature/pressure validity>
    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid
```

If any of the main five coefficients (i.e. `b0` to `C1`) is not used for any electrolyte, it will be set to zero. Unused `alph1`, `alph2` or `omega` coefficents are set to **−9**. (If a value of zero was used, it would cause problems going through **pytzer.model**, even though the problematic result is then multiplied by zero.)


### Other functions

All the other coefficients do not have auxiliary parameters and consequently have much simpler functions:

```python
def coefficient_ion1_ion2[_ion3]_source(T, P):
    coeff = <coefficient equation/value>
    valid = <logical expression evaluating temperature validity>
    return coeff, valid
```

Again, if the coefficient is explicitly defined as unused by the source, then the output `coeff` will be zero.
