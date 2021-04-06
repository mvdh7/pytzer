# Debye-Hückel limiting slopes

*The casual user has no need to explicitly call this module.*

`.debyehueckel` contains functions to evaluate the Debye-Hückel limiting slope for the osmotic coefficient (i.e. $A_\phi$).

The syntax for the functions is similar to that for the [interaction parameters](../parameters): inputs are `tempK` and `pres`, in K and dbar respectively; outputs are the $A_\phi$ value `Aosm` and a logical validity indicator `valid`.

Several different calculation approaches are possible. As a function of temperature only, at a pressure of c. 1 atm:

  * `.Aosm_M88` - following [M88](../../refs/#M88);
  * `.Aosm_CRP94` - following [CRP94](../../refs/#CRP94);
  * `.Aosm_MarChemSpec` - following [CRP94](../../refs/#CRP94), with a constant correction to agree better with [AW90](../../refs/#AW90).

As a function of temperature and pressure:

  * `Aosm_AW90` - following [AW90](../../refs/#AW90), but with pure water density evaluated using the functions in the [TEOS-10 module](../teos10).
