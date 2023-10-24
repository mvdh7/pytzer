# Version history

## 0.5 [beta]

Switches from Autograd to [JAX](https://jax.readthedocs.io/en/latest/) for faster automatic differentiation and JIT compilation.

### 0.5.3 (24 Oct 2023)

!!! new-version "Changes in v0.5.3"
    * Created new `Humphreys22` parameter library and validated its calculations against [HWT22](../refs/#h).
    * Created new `Clegg22` parameter library (but its calculations against [CHW22](../refs/#c) still need validating).
    * Created new `Clegg23` parameter library and validated its calculations against [CWTD23](../refs/#c).
    * Renamed `io` module as `get`.

### 0.5.2 (31 Aug 2021)

!!! new-version "Changes in v0.5.2"
    * Fixed calculation error in `convert.solvent_to_solution`.
    * Added more ions in `properties.ion_to_mass.`

### 0.5.1 (17 Aug 2021)

!!! new-version "Changes in v0.5.1"
    * Fixed `pz.convert.solvent_to_solution` to take and return p*K* values instead of *K* values, and to replace any NaNs in the input with 0.
    * Added `pz.equilibrate.stoichiometric.get_constants()` to calculate stoichiometric equilibrium constants for the carbonic acid equilibria when total dissolved inorganic carbon is zero.

### 0.5.0 (9 Aug 2021)

!!! new-version "Changes in v0.5.0"
    * Switched from Autograd to [JAX](https://jax.readthedocs.io/en/latest/) for faster automatic differentiation and JIT compilation.
    * Renamed and reorganised virtually everything.

## 0.4 [beta]

Adds solvers to determine the equilibrium composition of aqueous solutions.  This version was demonstrated at the AGU/ASLO/TOS Ocean Sciences Meeting (San Diego, Feb 2020).

### 0.4.3 (11 Sep 2019)

!!! new-version "Changes in v0.4.3"
    * Added `osm` function for osmotic coefficient to **matrix** module;
    * Added **io** module functions to roughly convert between molinity (mol/kg-solution) and molality (mol/kg-H<sub>2</sub>O);
    * Added `salinity2mols` function in **io** module to estimate molalities from salinity for [MZF93](../refs/#m) artificial seawater;
    * Switched `assemble` function in **matrix** module to use the MarChemSpec parameter library by default.

### 0.4.2 (6 Sep 2019)

!!! new-version "Changes in v0.4.2"
    * Removed pressure input from thermodynamic equilibrium constant functions;
    * Converted copies from shallow to deep in parameter library modules;
    * Added **plot** module with some meta-analytical visualisation tools (not yet documented).

### 0.4.1 (13 Aug 2019)

!!! new-version "Changes in v0.4.1"
    * Added carbonic acid and boric acid equilibrium solving;
    * Eliminated module **tables**;
    * Updated functions and parameter libraries noting that `['Cl', 'CO3']` gets sorted to `['CO3', 'Cl']`;
    * Fixed incorrect `alph1` values for all HMW 1:(>1) electrolytes;
    * Added several new interaction parameter functions;
    * Testing MIAMI parameter library against PM16's model:
        * Adjusted numerical values to fix errors in some parameter functions:
            * `bC_Ca_BOH4_SRM87`;
            * `bC_K_OH_MP98`;
            * `bC_Mg_BOH4_SRM87`;
            * `bC_Na_Br_MP98`;
            * `bC_Na_F_MP98`;
            * `bC_Na_HSO4_MP98`;
            * `bC_Sr_Cl_MP98`;
            * `theta_K_Sr_MP98`;
        * Added new MP98 functions until numerical conflicts can be resolved:
        * `bC_K_CO3_MP98` should follow SRG87;
        * `theta_H_K_MP98` and `theta_H_Na_MP98` should follow CMR93.

### 0.4.0 (3 Aug 2019)

!!! new-version "Changes in v0.4.0"
    * Module **dissociation** added with functions to evaluate thermodynamic dissociation constants;
    * Module **equilibrate** added with an equilibrium solver and associated functions;
    * Added **blackbox_equilibrate** function to demonstrate equilibration procedure;
    * Fixed indexing error in **matrix.assemble** for the *zeta* coefficient matrix;
    * Module **jfuncs** renamed as **unsymmetrical**;
    * As a general terminology change, the Pitzer model interaction *coefficients* are now instead referred to as *parameters*, for better consistency with the published literature. As a consequence:
        * Module **coefficients** renamed as **parameters**;
        * Module **cflibs** renamed as **libraries**;
        * Class **CoefficientLibrary** renamed as **ParameterLibrary**, and its abbreviation **cflib** to **prmlib**;
    * Parameter libraries can now be stored in separate script files.
    * Added complete [HMW84](../refs/#h) parameter library;
    * Added all interaction coefficients to [MIAMI](../refs/#m) parameter library - but its calculation results have not yet been tested;
    * Fixed multiplier error for `Cphi` values in `PM73_TableVI` in **tables** module;
    * Added presumably missing `*1e-3` multiplier to `Cphi` values in all SRRJ87 binary interaction functions;
    * Added meta-analysis functions in **meta** module.

## 0.3 [beta]

Adds pressure as an input variable.  However, most of the interaction coefficients are still not yet pressure-sensitive.

### 0.3.1 (16 Jul 2019)

!!! new-version "Changes in v0.3.1"
    * Module **props** renamed as **properties**;
    * Module **coeffs** renamed as **coefficients**;
    * Class **CoeffLib** renamed as **CoefficientLibrary**;
    * Added docstrings to all functions in **coefficients** module;
    * Added **matrix** module with alternative, matrix-based Pitzer model implementation;
    * Added **none** function to **jfuncs** module to ignore unsymmetrical mixing;
    * Fixed <i>A<sub>ϕ</sub></i> bug in **CoefficientLibrary** method `print_coeffs`;
    * In preparation for equilibration calculations:
        * Added `gettots` function to **io** module.

### 0.3.0 (12 Apr 2019)

!!! new-version "Changes in v0.3.0"
    * Moved Debye-Hückel functions (for <i>A<sub>ϕ</sub></i>) from **coeffs** into new module **debyehueckel**;
    * Added Archer and Wang (1990) calculation of <i>A<sub>ϕ</sub></i>;
    * Added pressure input (`pres` in dbar) to all relevant functions in **coeffs**, **debyehueckel**, **model**, **io** and **blackbox** functions;
    * Added **teos10** module to calculate various properties of pure water;
    * Added **Seawater** coefficient library - like **MarChemSpec**, but with pressure-dependent <i>A<sub>ϕ</sub></i> term (following Archer and Wang, 1990), and other general improvements;
    * Made `cflib` input optional for all **model** functions (default = Seawater);
    * Fixed loop index correction for n-n' interactions;
    * Eliminated unnecessary arrays for constant values in **coeffs**;
    * Adjusted **model.B** and **model.CT** function inputs to reduce number of **cflib** function calls;
    * Rewrote CRP94 `Aosm` function to make it autograd-able;
    * Added docstrings throughout (except for functions in **coeffs**).

## 0.2 [beta]

Includes calculation of solute activity coefficients, water activity and osmotic coefficients, as a function of temperature and composition, at a constant pressure of 1 atmosphere.

### 0.2.7 (4 Mar 2019)

!!! new-version "Changes in v0.2.7"
    * Transposed all inputs to the excess Gibbs energy function and its derivatives, giving about a 5× speed-up;
    * For consistency with **Pitzer.jl**:
        * Simplified osmotic coefficient calculation approach;
        * Added direct calculation of water activity by differentiation;
        * Changed both `T` and `temp` to `tempK` in input files and all functions;
        * Renamed coefficient "dictionary" as "library":
            * **CoefficientDictionary** class becomes **CoeffLib**;
            * **cfdict** function inputs become **cflib**;
        * Updated documentation to reflect these changes;
    * Added **io.saveall** function to create CSV file for results;
    * Switched **CoeffLib.get_contents()** to use dict key names, not function names, to find ions.

### 0.2.6 (20 Feb 2019)

!!! new-version "Changes in v0.2.6"
    * Continuing to build MIAMI (i.e. MP98) coefficient dictionary:
        * Added `bC_` functions for borate interactions from SRM87;
        * Added `bC_` functions for bisulfide interactions from HPM88;
        * Added new tables of PM73 data, and a new type of coefficient function in `bC_PM73(T,iset)` to use them (not yet compatible with all **CoefficientDictionary** methods);
    * Updated **pytzer.blackbox**:
        * Changed output filename extension to **\_py**;
        * Saving results to file now optional (with default behaviour to save);
        * Can also optionally use a different **CoefficientDictionary** (default remains **MarChemSpec**);
    * Added numerical integral approach for unsymmetrical mixing *J* function to **jfuncs** (not yet differentiable by autograd).

### 0.2.5 (6 Feb 2019)

!!! new-version "Changes in v0.2.5"
    * <u>Testing **MarChemSpec05** against **FastPitz** at 278.15 K - perfect agreement to 6 significant figures</u>;
    * Added `theta_Ca_H_MarChemSpec` to combine RGO82 and WM13 equations;
    * Added new module **pytzer.tables** to store long lists of coefficients found in tables;
    * Added basic **tables** page to documentation, and new ions to the name conventions lists;
    * Updated `bC_` functions for `Ca-SO4`, `Ca-HSO4` and `K-HSO4`:
        * Used data from new **pytzer.tables** module;
        * Relabelled from `P91` to `WM13`, to better reflect their provenance;
    * Added `Aosm_MarChemSpec`, which adds a small offset to `Aosm_CRP94` to make it consistent with AW90 and FastPitz;
    * Created new **MarChemSpec** **CoefficientDictionary**, with temperature-varying `Aosm_MarChemSpec`;
    * Added equations and usage examples to documentation for **pytzer.model**;
    * Wrote **pytzer.blackbox** function and quick-start documentation, for easy end-user testing.

### 0.2.4 (31 Jan 2019)

!!! new-version "Changes in v0.2.4"
    * Testing activity coefficients against Prof D.R. Turner's and Prof S.L. Clegg's implementations (GIVAKT and FastPitz respectively):
        * Assembled new **CoefficientDictionary** WM13_MarChemSpec25 for testing;
        * Corrected `alph1` for **bC_Ca_OH_HMW84** to 2;
        * Fixed temperature units for **theta_H_Na_CMR93** and **theta_H_K_CMR93**;
        * Deleted duplicate MP98 functions in **coeffs**;
        * Added functions with temporary values for 298.15 K for *λ*(tris, tris), *ζ*(tris, Na, Cl) and *μ*(tris, tris, tris) to **MarChemSpec25** **CoefficientDictionary**;
        * <u>**pytzer** activity and osmotic coefficients agree perfectly (to 6 significant figures) with FastPitz at 298.15 K with the **MarChemSpec25** **CoefficientDictionary**</u>;
    * Continued adding coefficient functions to **cfdicts.MIAMI**:
        * All from PP82 (`Na-CO3` and `Na-HCO3` interactions);
    * Added equations for neutral-neutral interactions (lambda coefficient) to **model.Gex_nRT** and also to **cfdicts.add_zeros** method;
    * Changed to evaluating all neutral interactions (including with ions) even at zero ionic strength;
    * Updated nomenclature for consistency with P91, MP98 and others: `eta` (*η*) is now `zeta` (*ζ*).

### 0.2.3 (30 Jan 2019)

*The build of v0.2.3 had errors that prevented installation. These were fixed by v0.2.3.3, which is otherwise identical.*

!!! new-version "Changes in v0.2.3"
    * Verified both **jfuncs.P75_eq47** and **jfuncs.Harvie**, and their derivatives, against Prof D.R. Turner's and Prof S.L. Clegg's implementations (GIVAKT and FastPitz respectively):
        * <u>**P75_eq47** returns identical results (to >10 significant figures) in every case</u>;
        * <u>**Harvie** agrees perfectly (to >10 significant figures) with GIVAKT, but FastPitz differs from both by up to 5%</u>;
    * Began assembling MIAMI **CoefficientDictionary** following MP98;
    * Added constant <i>A<sub>ϕ</sub></i> function for 25 °C following P91 to **coeffs.Aosm_25** for MarChemSpec project testing.

### 0.2.2 (28 Jan 2019)

!!! new-version "Changes in v0.2.2"
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

### 0.2.1 (24 Jan 2019)

!!! new-version "Changes in v0.2.1"
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

### 0.2.0 (23 Jan 2019)

The first beta release, including full documentation on [pytzer.readthedocs.io](https://pytzer.readthedocs.io).

## 0.1 [alpha]

### 0.1.X

Versions 0.1.X were used for alpha development. Changes are not documented.
