# Unsymmetrical mixing functions

*The casual user has no need to explicitly call this module.*

**pytzer.jfuncs** provides different ways to evaluate the J function that appears in the [unsymmetrical mixing terms of the Pitzer model](../../modules/model/#etheta).

One of these functions must be contained within the [cfdict](../cfdicts) in order to execute the **pytzer.model** functions.

# jfunc options

## .P75_eq46

Evaluates J following Pitzer (1975) Eq. (46).

## .P75_eq47

Evaluates J following Pitzer (1975) Eq. (47).

## .Harvie

Evaluates J following "Harvie's method", as described by Pitzer (1991), pages 124 to 125.

## .numint

Uses a numerical integration method to evaluate J "exactly". *Not currently autograd-able.*

<hr />

# References

Pitzer, K. S. (1975). Thermodynamics of electrolytes. V. effects of higher-order electrostatic terms. *J. Solution Chem.* 4, 249–265. [doi:10.1007/BF00646562](https://doi.org/10.1007/BF00646562).

Pitzer, K. S. (1991). “Ion Interaction Approach: Theory and Data Correlation,” in *Activity Coefficients in Electrolyte Solutions, 2nd Edition*, ed. K. S. Pitzer (CRC Press, Florida, USA), 75–153.
