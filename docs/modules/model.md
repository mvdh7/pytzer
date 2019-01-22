# Introduction

**pytzer.model** executes the Pitzer model to calculate solute and solvent activity and osmotic coefficients.

<hr />

# Excess Gibbs energy

From a physicochemical perspective, the excess Gibbs energy of a solution is the master variable from which many other properties are - literally - derived (Pitzer, 1991).

In **pytzer**, the excess Gibbs energy is the only physicochemical equation that is actually explicitly written out. All other properties are determined by taking the appropriate differential of the excess Gibbs energy. These differentials are determined automatically, by **autograd**.

## pytzer.model.Gex_nRT



<hr />

# Activity and osmotic coefficients

## pytzer.model.acfs



## pytzer.model.ln_acf2ln_acf_MX



## pytzer.model.osm



## pytzer.model.osm2aw



## pytzer.model.aw2osm



<hr />

# Pitzer model subfunctions

**pytzer.model** breaks down the full Pitzer model equation into some component subfunctions for clarity.

## pytzer.model.fG



## pytzer.model.g



## pytzer.model.h



## pytzer.model.B


## pytzer.model.CT



## pytzer.model.xij



## pytzer.model.etheta



<hr />

# References

Pitzer, K. S. (1991). “Ion Interaction Approach: Theory and Data Correlation,” in *Activity Coefficients in Electrolyte Solutions, 2nd Edition*, ed. K. S. Pitzer (CRC Press, Florida, USA), 75–153.
