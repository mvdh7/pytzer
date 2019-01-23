# Introduction

**pytzer.model** executes the Pitzer model to calculate solute and solvent activity and osmotic coefficients.

## Function inputs

Many of these functions have a common set of inputs: `mols`, `ions`, `T` and `cfdict`.

The first three of these inputs can be generated from an input file by **pytzer.io.getmols** and their formats are described in [the relevant documentation](../io/#pytzeriogetmols).

The final input is a **cfdict** (coefficient dictionary), which defines the set of interaction coefficients to use in the model, as described on [the relevant page](../cfdicts).

All of the usage examples below assume that you have first imported **pytzer** as `pz`:

```python
import pytzer as pz
```

<hr />

# Excess Gibbs energy

From a physicochemical perspective, the excess Gibbs energy of a solution is the master variable from which many other properties are - literally - derived (Pitzer, 1991).

In **pytzer**, the excess Gibbs energy is the only physicochemical equation that is actually explicitly written out. All other properties are determined by taking the appropriate differential of the excess Gibbs energy. These differentials are determined automatically, by **autograd**.

## pytzer.model.Gex_nRT

```python
Gex_nRT = pz.model.Gex_nRT(mols,ions,T,cfdict)
```

Evaluates the excess Gibbs energy of the solution (per mol and divided by *RT*).

<hr />

# Activity and osmotic coefficients

## pytzer.model.acfs / pytzer.model.ln_acfs

```python
ln_acfs = pz.model.ln_acfs(mols,ions,T,cfdict)
acfs    = pz.model.acfs   (mols,ions,T,cfdict)
```

Returns a matrix of activity coefficients (`.acfs`) or their natural logarithm (`.ln_acfs`) of the same size and shape as input `mols`. Each activity coefficient is for the same ion and solution composition as the corresponding input molality.

## pytzer.model.ln_acf2ln_acf_MX

```python
ln_acf_MX = pz.model.ln_acf2ln_acf_MX(ln_acfM,ln_acfX,nM,nX)
```

Combines the natural logarithms of the activity coefficients of a cation (`ln_acfM`) and anion (`ln_acfM`) into a mean activity coefficient of an electrolyte (`ln_acf_MX`) with stoichiometric ratio between cation and anion of `nM`:`nX`.

## pytzer.model.osm

```python
osm = pz.model.osm(mols,ions,T,cfdict)
```

Calculates the osmotic coefficient for each input solution composition.

## pytzer.model.osm2aw

```python
aw = pz.model.osm2aw(mols,osm)
```

Converts an osmotic coefficient (`osm`) into a water activity (`aw`).

## pytzer.model.aw2osm

```python
aw = pz.model.aw2osm(mols,aw)
```

Converts a water activity (`aw`) into an osmotic coefficient (`osm`).

<hr />

# Pitzer model subfunctions

**pytzer.model** breaks down the full Pitzer model equation into some component subfunctions for clarity.

## pytzer.model.fG

The first line of Clegg et al. (1994) Eq. (AI1).

## pytzer.model.g

The function *g*, following Clegg et al. (1994) Eq. (AI13).

## pytzer.model.h

The function *h*, following Clegg et al. (1994) Eq. (AI15).

## pytzer.model.B

The function *B*<sub>ca</sub>, following Clegg et al. (1994) Eq. (AI7).

## pytzer.model.CT

The function *C*<sup>T</sup><sub>ca</sub>, following Clegg et al. (1994) Eq. (AI10).

## pytzer.model.xij

The variable <i>x<sub>ij</sub></i>, following Clegg et al. (1994) Eq. (AI23).

## pytzer.model.etheta

The function *θ*<sup>E</sup><i><sub>ij</sub></i>, following Clegg et al. (1994) Eq. (AI20).

<hr />

# References

Clegg, S. L., Rard, J. A., and Pitzer, K. S. (1994). Thermodynamic properties of 0–6 mol kg<sup>–1</sup> aqueous sulfuric acid from 273.15 to 328.15 K. <i>J. Chem. Soc., Faraday Trans.</i> 90, 1875–1894. <a href="https://doi.org/10.1039/FT9949001875">doi:10.1039/FT9949001875</a>.

Pitzer, K. S. (1991). “Ion Interaction Approach: Theory and Data Correlation,” in *Activity Coefficients in Electrolyte Solutions, 2nd Edition*, ed. K. S. Pitzer (CRC Press, Florida, USA), 75–153.
