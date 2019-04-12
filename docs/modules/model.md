<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
MathJax.Ajax.config.path["mhchem"] =
  "https://cdnjs.cloudflare.com/ajax/libs/mathjax-mhchem/3.3.2";
MathJax.Hub.Config({TeX: {extensions: ["[mhchem]/mhchem.js"]}});
</script><script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML' async></script>

# The Pitzer model

`.model` executes the Pitzer model to calculate solute and solvent activity and osmotic coefficients.

## Function inputs

Many of these functions have a common set of inputs: `mols`, `ions`, `tempK`, `pres`, `cflib` and `Izero`. The first four of these can be generated from an input CSV file; their formats are described in the [import/export documentation](../io/#getmols-import-csv-dataset). Throughout Pytzer, when we refer to `ions` we include any neutral species in the solution.

The next input `cflib` is a [coefficient library](../cflibs), which defines the set of interaction coefficients to use in the model. The [Seawater](../cflibs#Seawater) library is used by default.

`Izero` is another optional input. The default value of `False` executes a full Pitzer model. If `Izero` is instead changed to `True`, then only neutral interactions are evaluated: this is the setting to use for solutions with zero ionic strength. If you try to pass a zero-ionic-strength solution through the full model, a `nan` is returned along with lots of divide-by-zero warnings. You must split up your own input data and run the function twice, if you have both types of solution. The [black box function](../../quick-start/#running-pytzer-as-a-black-box) handles this split automatically.

All of the usage examples below assume that you have first imported Pytzer as `pz`:

```python
>>> import pytzer as pz
```

<hr />

## Excess Gibbs energy

From a physicochemical perspective, the excess Gibbs energy of a solution ($G_{ex}$) is the master variable from which many other properties are - literally - derived. Following [P91](../../references/#P91), Eqs. (59) and (F-5), and [CRP94](../../references/#CRP94), Eq. (AI1):

$$\frac{G_{ex}}{w_wRT} = f_G + 2 \sum_c \sum_a m_c m_a (B_{ca} + Z C_{ca}^{T})$$
$$+ \sum_i \sum_{i'} m_i m_{i'} \Bigl( 2 \Theta_{ii'} + \sum_j m_j \psi_{ii'j} \Bigr)$$
$$+ \sum_n m_n \Bigl(2 \sum_i m_i \lambda_{ni} + 2 \sum_{n'} m_{n'} \lambda_{nn'} + m_n \lambda_{nn} \Bigr)$$
$$+ \sum_n \sum_c \sum_a m_n m_c m_a \zeta_{nca} + \sum_n m_n^3 \mu_{nnn} $$

On the left hand side, we have the excess Gibbs energy ($G_{ex}$) divided by the universal gas constant ($R$ in J·mol·K<sup>−1</sup>, defined in [the 'constants' module](../constants)), temperature ($T$ in K), and mass of pure water ($w_w$ in kg, set to unity).

The first term on the right ($f_G$) is the Debye-Hückel approximation, a function of ionic strength evaluated by the function `fG` ([see below](#fg)). It includes the Debye-Hückel limiting slope $A_\phi$, which is evaluated using an [interaction coefficient function](../coeffs).

The Pitzer model then adds a series of corrections to this approximation for each dissolved cation ($c$), anion ($a$) and neutral component ($n$). In the equation above, $i$ refers to any ion and $j$ refers to an ion of the opposite sign charge. The term $i'$ refers to a different ion from $i$, but with the same sign charge. Similarly, $n'$ refers to a different neutral component from $n$.

The $m_x$ terms indicate the molality of component $x$, in mol·kg<sup>−1</sup>.

The $B_{ca}$ and $C_{ca}^T$ terms account for cation-anion interactions, and are evaluated by the functions `B` and `CT` ([see below](#b)). $B_{ca}$ depends on the empirical coefficients $\beta_0$, $\beta_1$, $\beta_2$, $\alpha_1$ and $\alpha_2$, and $C_{ca}^{T}$ on $C_0$, $C_1$ and $\omega$. $Z$ is defined by [P91](../../references/#P91), Eq. (66):

$$Z = \sum_i m_i |z_i|$$

where $z_i$ is the charge on ion $i$.

The $\Theta_{ii'}$ terms account for cation-cation and anion-anion interactions. As defined by [P91](../../references/#P91), Eq. (B-6):

$$\Theta_{ii'} = \theta_{ii'} + ^E\theta_{ii'}$$

where the $^E\theta_{ii'}$ term is a function of ionic strength, evaluated by `etheta` ([see below](#etheta)).

Finally, the "Greek letter terms" ($\beta_0$, $\beta_1$, $\beta_2$, $\alpha_1$, $\alpha_2$, $C_0$, $C_1$, $\omega$, $\theta_{ii'}$, $\psi_{ii'j}$, $\lambda_{nx}$, $\zeta_{nca}$ and $\mu_{nnn}$) are empirical coefficients, different for each combination of ions and neutral species, functions of temperature and sometimes pressure.

In Pytzer, the excess Gibbs energy is the only physicochemical equation that is actually explicitly written out. All other properties are determined by taking the appropriate differential of the excess Gibbs energy. These differentials are determined automatically by [Autograd](https://github.com/HIPS/autograd).

### `.Gex_nRT` - excess Gibbs energy

Evaluates $G_{ex}/w_wRT$, as defined above.

**Syntax:**

```python
>>> Gex_nRT = pz.model.Gex_nRT(mols, ions, tempK, pres, cflib=pz.cflibs.Seawater, Izero=False)
```

<hr />

## Activity and osmotic coefficients

The natural logarithm of the activity coefficient ($\ln \gamma_x$) of any dissolved component of a solution is the first differential of $G_{ex}/w_wRT$ with respect to the component's molality, following [P91](../../references/#P91), Eq. (34):

$$\ln \gamma_x = \frac{\partial (G_{ex}/w_wRT)}{\partial m_x} $$

The osmotic coefficient of a solution ($\phi$) is related to the first differential of $G_{ex}/w_wRT$ with respect to $w_w$, following [P91](../../references/#P91), Eq. (35):

$$\phi = 1 - \frac{\partial G_{ex}/\partial w_w}{RT} \sum_x m_x$$

where $x$ includes all ions and neutral components.

The solvent (i.e. water) activity ($a_w$) is related to the osmotic coefficient by [P91](../../references/#P91), Eq. (28):

$$\phi = - \frac{\ln a_w}{M_w \sum_x m_x}$$

where $M_w$ is the molar mass of water, in kg·mol<sup>−1</sup>.

To evaluate $\gamma_x$ and $\phi$, Pytzer evaluates the appropriate differentials of $G_{ex}/w_wRT$ by using Autograd to differentiate the `Gex_nRT` function.

<br />

### `.acfs` and `.ln_acfs` - solute activity coefficients

Returns a matrix of activity coefficients ($\gamma_x$, `acfs`) or their natural logarithm ($\ln \gamma_x$, `ln_acfs`) of the same size and shape as input `mols`. Each activity coefficient is for the same ion and solution composition as the corresponding input molality.

**Syntax:**

```python
>>> ln_acfs = pz.model.ln_acfs(mols, ions, tempK, pres, cflib=pz.cflibs.Seawater, Izero=False)
>>> acfs = pz.model.acfs(mols, ions, tempK, pres, cflib=pz.cflibs.Seawater, Izero=False)
```

<br />

### `.ln_acf2ln_acf_MX` - mean activity coefficient

Combines the natural logarithms of the activity coefficients of a cation ($\ln \gamma_c$, `ln_acfM`) and anion ($\ln \gamma_a$, `ln_acfX`) into a mean activity coefficient of an electrolyte ($\ln \gamma_\pm$, `ln_acf_MX`) with stoichiometric ratio between cation and anion of `nM`:`nX`.

**Syntax:**

```python
>>> ln_acf_MX = pz.model.ln_acf2ln_acf_MX(ln_acfM, ln_acfX, nM, nX)
```

<br />

### `.osm` - osmotic coefficient

Calculates the osmotic coefficient ($\phi$) for each input solution composition.

**Syntax:**

```python
>>> osm = pz.model.osm(mols, ions, tempK, pres, cflib=pz.cflibs.Seawater, Izero=False)
```

<br />

### `.aw` and `.lnaw` - water activity

Calculates the water activity ($a_w$) or its natural logarithm for each input solution composition.

**Syntax:**

```python
>>> lnaw = pz.model.lnaw(mols, ions, tempK, pres, cflib=pz.cflibs.Seawater, Izero=False)
>>> aw = pz.model.aw(mols, ions, tempK, pres, cflib=pz.cflibs.Seawater, Izero=False)
```

<br />

### `.osm2aw` - convert osmotic coefficient to water activity

Converts an osmotic coefficient ($\phi$, `osm`) into a water activity ($a_w$, `aw`).

**Syntax:**

```python
>>> aw = pz.model.osm2aw(mols, osm)
```

<br />

### `.aw2osm` - convert water activity to osmotic coefficient

Converts a water activity ($a_w$, `aw`) into an osmotic coefficient ($\phi$, `osm`).

**Syntax:**

```python
>>> osm = pz.model.aw2osm(mols, aw)
```

<hr />

## Pitzer model subfunctions

The full Pitzer model equation is broken down into some component subfunctions for clarity.

### `.Istr` - ionic strength

Calculates the ionic strength of the solution ($I$, `I`), following [P91](../../references/#P91), Eq. (11):

$$I = \frac{1}{2} \sum_i m_i z_i^2$$

**Syntax:**

```python
>>> I = pz.model.Istr(mols, zs)
```

Input `zs` is a list of the charge on each ion, which can be generated from `ions` using the [solute properties functions](../props).

<br />

### `.fG` - Debye-Hückel approximation

The Debye-Hückel approximation of the excess Gibbs energy. From [P91](../../references/#P91), Eq. (48):

$$f_G = \frac{4 I A_\phi}{-b} \ln (1 + b\sqrt{I})$$

where $b$, here and hereafter, is equal to 1.2 (mol·K<sup>−1</sup>)<sup>1/2</sup>.

**Syntax:**

```python
>>> fG = pz.model.fG(tempK, pres, I, cflib)
```
<br />

### `.g` - binary interaction subfunction

The function $g$, following [P91](../../references/#P91), Eq. (50):

$$g = 2[1 - (1 + x) \exp(-x)] / x^2$$

**Syntax:**

```python
>>> g = pz.model.g(x)
```

<br />

### `.h` - binary interaction subfunction

The function $h$, following [CRP94](../../references/#CRP94), Eq. (AI15):

$$h = \\{ 6 - [6 + x (6 + 3x + x^2)] \exp(-x) \\} / x^4$$

**Syntax:**

```python
>>> h = pz.model.h(x)
```

<br />

### `.B` - binary interaction subfunction

The function $B_{ca}$, following [P91](../../references/#P91), Eq. (49):

$$B_{ca} = \beta_0 + \beta_1 g(\alpha_1 \sqrt{I}) + \beta_2 g(\alpha_2 \sqrt{I})$$

where $\beta_0$, $\beta_1$, $\beta_2$, $\alpha_1$ and $\alpha_2$ take different values for each $ca$ combination, as defined by the functions in **pytzer.coeffs**.

**Syntax:**

```python
>>> B = pz.model.B(I, b0, b1, b2, alph1, alph2)
```

<br />

### `.CT` - binary interaction subfunction

The function $C_{ca}^T$, following [CRP94](../../references/#CRP94) Eq. (AI10):

$$C_{ca}^T = C_0 + 4 C_1 h(\omega \sqrt{I})$$

where $C_0$, $C_1$ and $\omega$ take different values for each $ca$ combination, as defined by the functions in **pytzer.coeffs**.

**Syntax:**

```python
>>> CT = pz.model.CT(I, C0, C1, omega)
```

<br />

### `.xij` - unsymmetrical mixing subfunction

The variable $x_{ii'}$, following [P91](../../references/#P91), Eq. (B-14):

$$x_{ii'} = 6 z_i z_{i'} A_\phi \sqrt{I}$$

**Syntax:**

```python
>>> xij = pz.model.xij(tempK, I, z0, z1, cflib)
```

where `z0` and `z1` are the charges on ions $i$ and $i'$ respectively (i.e. $z_i$ and $z_{i'}$).

<br />

### `.etheta` - unsymmetrical mixing term

The function $^E\theta_{ii'}$, following [P91](../../references/#P91), Eq. (B-15):

$$^E\theta_{ii'} = \frac{z_i z_i'}{4 I} \Bigl[J(x_{ii'}) - \frac{1}{2} J(x_{ii}) - \frac{1}{2} J(x_{i'i'})\Bigr]$$

**Syntax:**

```python
>>> etheta = pz.model.etheta(tempK, I, z0, z1, cflib)
```

This is only evaluated when $z_i \neq z_{i'}$. Otherwise, it is equal to zero.
