# 0.2 (beta)


## 0.2.2

**Release date:** forthcoming

  * Added temperature to the file produced by **print_coeffs**;
  * Improved documentation:
    * Added neutrals, and made other updates, for **pytzer.coeffs**;
    * Updated for **pytzer.cfdicts**;
  * Fixed zero ionic strength error for **model.Gex_nRT**;
  * Removed `Jp` outputs from all **jfuncs**;
  * Corrected functions in **CoefficientDictionary** `WM13` following intercomparison with Prof S.L. Clegg:
    * `bC_Na_OH_PP87i`: typos in coefficients;
    * `bC_Mg_SO4_PP86ii`: eliminated 'difference of two large numbers' error by substitution;
  * Added external package requirements (i.e. **numpy** and **autograd**) to the **setup.py**.


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
