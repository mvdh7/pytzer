<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML' async></script>

# Debye-Hückel limiting slopes

*The casual user has no need to explicitly call this module.*

`.debyehueckel` contains functions to evaluate the Debye-Hückel limiting slope for the osmotic coefficient (i.e. $A_\phi$).

The syntax for the functions is similar to that for the [interaction parameters](../parameters): inputs are `tempK` and `pres`, in K and dbar respectively; outputs are the $A_\phi$ value `Aosm` and a logical validity indicator `valid`.

Several different calculation approaches are possible. As a function of temperature only, at a pressure of c. 1 atm:

  * `.Aosm_CRP94` - following Clegg et al. (1994);
  * `.Aosm_MarChemSpec` - following Clegg et al. (1994), with a constant correction to agree better with Archer and Wang (1990);
  * `.Aosm_M88` - following Møller (1988).

As a function of temperature and pressure:

  * `Aosm_AW90` - following Archer and Wang (1990), but with pure water density evaluated using the functions in the [TEOS-10 module](../teos10).
