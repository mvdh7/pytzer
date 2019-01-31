# 0.2 (beta)

## 0.2.4

**Release date:** forthcoming

  * Testing activity coefficients against Prof D.R. Turner's and Prof S.L. Clegg's implementations (GIVAKT and FastPitz respectively):
    * Assembled new **CoefficientDictionary** WM13_MarChemSpec25 for testing;
    * Corrected `alph1` for **bC_Ca_OH_HMW84** to 2;
    * Fixed temperature units for **theta_H_Na_CMR93** and **theta_H_K_CMR93**;
    * Deleted duplicate MP98 functions in **coeffs**;
  * Continued adding coefficient functions to **cfdicts.MIAMI**:
    * All from PP82 (`Na-CO3` and `Na-HCO3` interactions);
  * Added equations for neutral-neutral interactions (lambda coefficient) to **model.Gex_nRT** and also to **cfdicts.add_zeros** method;
  * Changed to evaluating all neutral interactions (including with ions) even at zero ionic strength;
  * Updated nomenclature for consistency with P91: `eta` is now `zeta`;
  * Added functions with temporary values for 298.15 K for *λ*(tris,tris), *ζ*(tris,Na,Cl) and *μ*(tris,tris,tris) to **MarChemSpec25** **CoefficientDictionary**, for testing.


## 0.2.3

*The build of v0.2.3 had errors that prevented installation. These were fixed by v0.2.3.3, which is otherwise identical.*

**Release date:** 2019-01-30

  * Verified both **jfuncs.P75_eq47** and **jfuncs.Harvie**, and their differentials, against Prof D.R. Turner's and Prof S.L. Clegg's implementations;
  * Began assembling MIAMI **CoefficientDictionary** following MP98;
  * Added constant <i>A<sub>ϕ</sub></i> function for 25 °C following P91 to **coeffs.Aosm_25** for MarChemSpec project testing.

## 0.2.2

**Release date:** 2019-01-28

  * Added temperature to the file produced by **print_coeffs**;
  * Improved documentation:
    * Added neutrals, and made other updates, for **pytzer.coeffs**;
    * Updated for **pytzer.cfdicts**;
    * Added **pytzer.meta** page;
    * Updated installation instructions;
    * Documented new `Izero=False` input to **model.Gex_nRT**;
  * Fixed zero ionic strength error for **model.Gex_nRT**;
  * Removed `Jp` outputs from all **jfuncs**, and set up correct differentiation for `Harvie`;
  * Corrected functions in **CoefficientDictionary** `WM13` following intercomparison with Prof S.L. Clegg's Fortran implementation:
    * `bC_Mg_SO4_PP86ii`: eliminated *difference of two large numbers* error, by substitution;
    * `bC_Na_HSO4_HPR93`: fixed incorrect charge for HSO<sub>4</sub><sup>−</sup>;
    * `bC_Na_OH_PP87i`: fixed typos in coefficients;
  * Switched **MarChemSpec** **CoefficientDictionary** to *not* use **GT17simopt** for `Na-Cl` interaction (stick with M88 instead);
  * Fixed function for `H-Na-HSO4` interaction in **WM13** **CoefficientDictionary**;
  * Added external package requirements (i.e. **numpy** and **autograd**) to the **setup.py**;
  * Fixed **CoefficientDictionary.get_contents** to include neutral species;
  * Added input `Izero=False` to **model.Gex_nRT** and its derivatives, to allow calculations for solutions with zero ionic strength.


## 0.2.1

**Release date:** 2019-01-24

  * Fixed fatal indexing error in **model.Gex_nRT**;
  * Added **CoefficientDictionary** `WM13` for the Waters and Millero (2013) model;
  * Added **CoefficientDictionary** `MarChemSpec` for MarChemSpec project testing;
  * Added *some* coefficient values from Gallego-Urrea and Turner (2017);
  * Added equations for neutral solute interactions to **model.Gex_nRT**, along with supporting functions in **pytzer.cfdicts**;
  * Added **CoefficientDictionary** methods:
    * **print_coeffs**, to evaluate all model coefficients at a given temperature, and print them to file;
    * **get_contents**, to generate lists of all ions and all references within the dictionary;
  * Updated documentation on coefficient dictionaries to reflect these changes;
  * Added **meta** module as single-source-of-truth for package version;
  * Updated documentation with basic information on neutral solutes.


## 0.2.0

**Release date:** 2019-01-23

The first beta release, including full documentation on [pytzer.readthedocs.io](https://pytzer.readthedocs.io).


# 0.1 (alpha)

## 0.1.X

Versions 0.1.X were used for alpha development. Changes are not documented.
