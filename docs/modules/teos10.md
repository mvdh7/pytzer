# TEOS-10 properties

`.teos10` contains the **specific Gibbs energy** master equation for pure water recommended by the International Association for the Properties of Water and Steam (IAPWS), along with functions to evaluate its derivative properties. All derivatives of the master equation are computed automatically by autograd.

The source for these equations is the [Supplementary Release on a Computationally Efficient Thermodynamic
Formulation for Liquid Water for Oceanographic Use](http://www.teos-10.org/pubs/IAPWS-2009-Supplementary.pdf) (IAWPS, 2009). All results appear to agree with the numerical check values given in their Table 6.

## Common syntax

Every function uses a common syntax:

```python
property = pz.teos10.property(tempK, pres)
```

**Inputs:**

  * `tempK` - water temperature in K;
  * `pres` - water pressure in dbar.

**Output:**

  * `property` - the specified property.

The following properties can be calculated, all for pure water:

  * `Gibbs` - specific Gibbs energy in J·kg<sup>−1</sup>;
  * `rho` - density in kg·m<sup>−3</sup>;
  * `s` - specific entropy in J·kg<sup>−1</sup>·K<sup>−1</sup>;
  * `cp` - specific isobaric heat capacity in J·kg<sup>−1</sup>·K<sup>−1</sup>;
  * `h` - specific enthalpy in J·kg<sup>−1</sup>;
  * `u` - specific internal energy in J·kg<sup>−1</sup>;
  * `f` - specific Helmholtz energy in J·kg<sup>−1</sup>;
  * `alpha` - thermal expansion coefficient in K<sup>−1</sup>;
  * `bs` - isentropic temperature-pressure coefficient/adiabatic lapse rate in K·Pa<sup>−1</sup>;
  * `kt` - isothermal compressibility in dbar<sup>−1</sup>;
  * `ks` - isentropic compressibility in dbar<sup>−1</sup>;
  * `w` - speed of sound in m·s<sup>−1</sup>.
