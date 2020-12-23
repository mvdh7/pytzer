# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2020  Matthew Paul Humphreys  (GNU GPLv3)
"""Solve for equilibrium."""
import jax
from jax import numpy as np


# def _varmols(eqstate, tots1, fixmols1, eles, fixions, fixcharges):
#     """Calculate variable molalities from solver targets."""
#     # Seems long-winded, but it's necessary for Autograd
#     q = 0
#     if len(eles) > 0:
#         for e, ele in enumerate(eles):
#             if ele == "t_HSO4":
#                 tHSO4 = tots1[e]
#                 if tHSO4 > 0:
#                     aHSO4 = _sig01(eqstate[q])
#                     mHSO4 = tHSO4 * aHSO4
#                     mSO4 = tHSO4 - mHSO4
#                 else:
#                     mHSO4 = 0.0
#                     mSO4 = 0.0
#                 q += 1
#             elif ele == "t_Mg":
#                 tMg = tots1[e]
#                 if tMg > 0:
#                     aMgOH = _sig01(eqstate[q])
#                     mMg = tMg * aMgOH
#                     mMgOH = tMg - mMg
#                 else:
#                     mMg = 0.0
#                     mMgOH = 0.0
#                 q += 1
#             elif ele == "t_trisH":
#                 ttrisH = tots1[e]
#                 if ttrisH > 0:
#                     atrisH = _sig01(eqstate[q])
#                     mtrisH = ttrisH * atrisH
#                     mtris = ttrisH - mtrisH
#                 else:
#                     mtrisH = 0.0
#                     mtris = 0.0
#                 q += 1
#             elif ele == "t_H2CO3":
#                 tH2CO3 = tots1[e]
#                 if tH2CO3 > 0:
#                     aH2CO3 = _sig01(eqstate[q])
#                     aHCO3 = _sig01(eqstate[q + 1])
#                     mCO2 = tH2CO3 * aH2CO3
#                     mHCO3 = (tH2CO3 - mCO2) * aHCO3
#                     mCO3 = tH2CO3 - mCO2 - mHCO3
#                 else:
#                     mCO2 = 0.0
#                     mHCO3 = 0.0
#                     mCO3 = 0.0
#                 q += 2
#             elif ele == "t_BOH3":
#                 tBOH3 = tots1[e]
#                 if tBOH3 > 0:
#                     aBOH3 = _sig01(eqstate[q])
#                     mBOH3 = tBOH3 * aBOH3
#                     mBOH4 = tBOH3 - mBOH3
#                 else:
#                     mBOH3 = 0.0
#                     mBOH4 = 0.0
#                 q += 1
#         if "t_HSO4" not in eles:
#             mHSO4 = 0.0
#             mSO4 = 0.0
#         if "t_Mg" not in eles:
#             mMg = 0.0
#             mMgOH = 0.0
#         if "t_trisH" not in eles:
#             mtrisH = 0.0
#             mtris = 0.0
#         if "t_H2CO3" not in eles:
#             mCO2 = 0.0
#             mHCO3 = 0.0
#             mCO3 = 0.0
#         if "t_BOH3" not in eles:
#             mBOH3 = 0.0
#             mBOH4 = 0.0
#     else:
#         mHSO4 = 0.0
#         mSO4 = 0.0
#         mMg = 0.0
#         mMgOH = 0.0
#         mtrisH = 0.0
#         mtris = 0.0
#         mCO2 = 0.0
#         mHCO3 = 0.0
#         mCO3 = 0.0
#         mBOH3 = 0.0
#         mBOH4 = 0.0
#     zbHSO4 = -mHSO4 - 2 * mSO4
#     zbMg = 2 * mMg + mMgOH
#     zbtrisH = mtrisH
#     zbH2CO3 = -(mHCO3 + 2 * mCO3)
#     zbBOH3 = -mBOH4
#     if len(fixcharges) == 0:
#         zbfixed = 0.0
#     else:
#         zbfixed = np.sum(fixmols1 * fixcharges)
#     zbalance = zbfixed + zbHSO4 + zbMg + zbtrisH + zbH2CO3 + zbBOH3
#     if len(eqstate) == q + 1:
#         dissociatedH2O = np.exp(-eqstate[-1])
#     elif len(eqstate) == q:
#         dissociatedH2O = 0.0
#     else:
#         print("WARNING: eqstate and tots1 dimensions are not compatible!")
#     mOH = (zbalance + np.sqrt(zbalance ** 2 + dissociatedH2O)) / 2
#     mH = mOH - zbalance
#     return (
#         mH,
#         mOH,
#         mHSO4,
#         mSO4,
#         mMg,
#         mMgOH,
#         mtris,
#         mtrisH,
#         mCO2,
#         mHCO3,
#         mCO3,
#         mBOH3,
#         mBOH4,
#     )


@jax.jit
def Gibbs_H2O(ln_aw, m_H, ln_acf_H, m_OH, ln_acf_OH, ln_kH2O):
    """Evaluate the Gibbs energy for water dissocation."""
    return ln_acf_H + np.log(m_H) + ln_acf_OH + np.log(m_OH) - ln_aw - ln_kH2O


# def Gibbs_HSO4(mH, lnacfH, mSO4, lnacfSO4, mHSO4, lnacfHSO4, lnkHSO4):
#     """Evaluate the Gibbs energy for the bisulfate-sulfate equilibrium."""
#     return lnacfH + np.log(mH) + lnacfSO4 + np.log(mSO4) - lnacfHSO4 - np.log(mHSO4) - lnkHSO4


# def Gibbs_Mg(mMg, lnacfMg, mMgOH, lnacfMgOH, mOH, lnacfOH, lnkMg):
#     """Evaluate the Gibbs energy for the magnesium-MgOH+ equilibrium."""
#     return lnacfMg + np.log(mMg) + lnacfOH + np.log(mOH) - lnacfMgOH - np.log(mMgOH) + lnkMg


# def Gibbs_trisH(mH, lnacfH, mtris, lnacftris, mtrisH, lnacftrisH, lnktrisH):
#     """Evaluate the Gibbs energy for the tris-trisH+ equilibrium."""
#     return (
#         lnacftris + np.log(mtris) - lnacftrisH - np.log(mtrisH) + lnacfH + np.log(mH) - lnktrisH
#     )


def Gibbs_H2CO3(lnaw, mH, lnacfH, mHCO3, lnacfHCO3, mCO2, lnacfCO2, lnkH2CO3):
    """Evaluate the Gibbs energy for the H2CO3-bicarbonate equilibrium."""
    return (
        lnacfH
        + np.log(mH)
        + lnacfHCO3
        + np.log(mHCO3)
        - lnacfCO2
        - np.log(mCO2)
        - lnaw
        - lnkH2CO3
    )


def Gibbs_HCO3(mH, lnacfH, mHCO3, lnacfHCO3, mCO3, lnacfCO3, lnkHCO3):
    """Evaluate the Gibbs energy for the bicarbonate-carbonate equilibrium."""
    return lnacfH + np.log(mH) + lnacfCO3 + np.log(mCO3) - lnacfHCO3 - np.log(mHCO3) - lnkHCO3


# def Gibbs_BOH3(lnaw, lnacfBOH4, mBOH4, lnacfBOH3, mBOH3, lnacfH, mH, lnkBOH3):
#     """Evaluate the Gibbs energy for the boric acid equilibrium."""
#     return (
#         lnacfBOH4
#         + np.log(mBOH4)
#         + lnacfH
#         + np.log(mH)
#         - lnacfBOH3
#         - np.log(mBOH3)
#         - lnaw
#         - lnkBOH3
#     )


# def Gibbs_components(
#     eqstate,
#     tots1,
#     fixmols1,
#     eles,
#     allions,
#     fixions,
#     fixcharges,
#     allmxs,
#     lnks,
#     ideal=False,
# ):
#     """Evaluate the Gibbs energy for each component equilibrium."""
#     (
#         mH,
#         mOH,
#         mHSO4,
#         mSO4,
#         mMg,
#         mMgOH,
#         mtris,
#         mtrisH,
#         mCO2,
#         mHCO3,
#         mCO3,
#         mBOH3,
#         mBOH4,
#     ) = _varmols(eqstate, tots1, fixmols1, eles, fixions, fixcharges)
#     allmols = deepcopy(fixmols1)
#     # The order in which molality values are added to allmols within each
#     # equilibrium in the following statements MUST comply with the order in
#     # which the corresponding ions were added to allions by the function
#     # pytzer.properties.getallions, which is defined by their order in the
#     # entries of the pytzer.properties._eq2ions dict.
#     q = 0
#     for ele in eles:
#         if ele == "t_HSO4":
#             allmols = [*allmols, mHSO4, mSO4]
#             lnkHSO4 = lnks[q]
#             q += 1
#         elif ele == "t_Mg":
#             allmols = [*allmols, mMg, mMgOH]
#             lnkMg = lnks[q]
#             q += 1
#         elif ele == "t_trisH":
#             allmols = [*allmols, mtrisH, mtris]
#             lnktrisH = lnks[q]
#             q += 1
#         elif ele == "t_H2CO3":
#             allmols = [*allmols, mCO2, mHCO3, mCO3]
#             lnkH2CO3 = lnks[q]
#             lnkHCO3 = lnks[q + 1]
#             q += 2
#         elif ele == "t_BOH3":
#             allmols = [*allmols, mBOH3, mBOH4]
#             lnkBOH3 = lnks[q]
#             q += 1
#     allmols = array([[*allmols, mH, mOH]])
#     solveH2O = len(eqstate) == q + 1
#     # Get activities:
#     if ideal:
#         lnaw = 0.0
#         lnacfH = 0.0
#         lnacfOH = 0.0
#         lnacfHSO4 = 0.0
#         lnacfSO4 = 0.0
#         lnacfMg = 0.0
#         lnacfMgOH = 0.0
#         lnacftris = 0.0
#         lnacftrisH = 0.0
#         lnacfCO2 = 0.0
#         lnacfHCO3 = 0.0
#         lnacfCO3 = 0.0
#         lnacfBOH3 = 0.0
#         lnacfBOH4 = 0.0
#     else:
#         lnaw = matrix.lnaw(allmols, allmxs)
#         lnacfs = matrix.ln_acfs(allmols, allmxs)
#         lnacfH = lnacfs[allions == "H"]
#         lnacfOH = lnacfs[allions == "OH"]
#         lnacfHSO4 = lnacfs[allions == "HSO4"]
#         lnacfSO4 = lnacfs[allions == "SO4"]
#         lnacfMg = lnacfs[allions == "Mg"]
#         lnacfMgOH = lnacfs[allions == "MgOH"]
#         lnacftris = lnacfs[allions == "tris"]
#         lnacftrisH = lnacfs[allions == "trisH"]
#         lnacfCO2 = lnacfs[allions == "CO2"]
#         lnacfHCO3 = lnacfs[allions == "HCO3"]
#         lnacfCO3 = lnacfs[allions == "CO3"]
#         lnacfBOH3 = lnacfs[allions == "BOH3"]
#         lnacfBOH4 = lnacfs[allions == "BOH4"]
#     # Evaluate equilibrium states:
#     if solveH2O:
#         lnkH2O = lnks[-1]
#         gH2O = Gibbs_H2O(lnaw, mH, lnacfH, mOH, lnacfOH, lnkH2O)
#     else:
#         gH2O = 0.0
#     if len(eles) > 0:
#         if tots1[eles == "t_HSO4"] > 0:
#             gHSO4 = Gibbs_HSO4(mH, lnacfH, mSO4, lnacfSO4, mHSO4, lnacfHSO4, lnkHSO4)
#         else:
#             gHSO4 = 0.0
#         if tots1[eles == "t_Mg"] > 0:
#             gMg = Gibbs_Mg(mMg, lnacfMg, mMgOH, lnacfMgOH, mOH, lnacfOH, lnkMg)
#         else:
#             gMg = 0.0
#         if tots1[eles == "t_trisH"] > 0:
#             gtrisH = Gibbs_trisH(
#                 mH, lnacfH, mtris, lnacftris, mtrisH, lnacftrisH, lnktrisH
#             )
#         else:
#             gtrisH = 0.0
#         if tots1[eles == "t_H2CO3"] > 0:
#             gH2CO3 = Gibbs_H2CO3(
#                 lnaw, mH, lnacfH, mHCO3, lnacfHCO3, mCO2, lnacfCO2, lnkH2CO3
#             )
#             gHCO3 = Gibbs_HCO3(mH, lnacfH, mHCO3, lnacfHCO3, mCO3, lnacfCO3, lnkHCO3)
#         else:
#             gH2CO3 = 0.0
#             gHCO3 = 0.0
#         if tots1[eles == "t_BOH3"] > 0:
#             gBOH3 = Gibbs_BOH3(
#                 lnaw, lnacfBOH4, mBOH4, lnacfBOH3, mBOH3, lnacfH, mH, lnkBOH3
#             )
#         else:
#             gBOH3 = 0.0
#     else:
#         gHSO4 = 0.0
#         gMg = 0.0
#         gtrisH = 0.0
#         gH2CO3 = 0.0
#         gHCO3 = 0.0
#         gBOH3 = 0.0
#     return gH2O, gHSO4, gMg, gtrisH, gH2CO3, gHCO3, gBOH3


# def Gibbs_equilibria(
#     eqstate,
#     tots1,
#     fixmols1,
#     eles,
#     allions,
#     fixions,
#     fixcharges,
#     allmxs,
#     lnks1,
#     ideal=False,
# ):
#     """Evaluate the total Gibbs energy to be minimised for all equilibria."""
#     gH2O, gHSO4, gMg, gtrisH, gH2CO3, gHCO3, gBOH3 = Gibbs_components(
#         eqstate,
#         tots1,
#         fixmols1,
#         eles,
#         allions,
#         fixions,
#         fixcharges,
#         allmxs,
#         lnks1,
#         ideal,
#     )
#     return (
#         gH2O ** 2
#         + gHSO4 ** 2
#         + gMg ** 2
#         + gtrisH ** 2
#         + gH2CO3 ** 2
#         + gHCO3 ** 2
#         + gBOH3 ** 2
#     )


# _GibbsGrad = egrad(_Gibbs)


# def solve(
#     eqstate_guess, tots1, fixmols1, eles, allions, fixions, allmxs, lnks1, ideal=False
# ):
#     """Solve for the solution's equilibrium state."""
#     fixcharges = transpose(properties.charges(fixions)[0])
#     Gargs = (tots1, fixmols1, eles, allions, fixions, fixcharges, allmxs, lnks1, ideal)
#     eqstate = minimize(
#         lambda eqstate: Gibbs_equilibria(eqstate, *Gargs),
#         eqstate_guess,
#         method="BFGS",
#         jac=lambda eqstate: Gibbs_Grad(eqstate, *Gargs),
#     )
#     return eqstate


# def solvequick(eqstate_guess, tots1, fixmols1, eles, allions, fixions, allmxs, lnks1):
#     """Solve ideal case first to speed up computation."""
#     Sargs = (tots1, fixmols1, eles, allions, fixions, allmxs, lnks1)
#     eqstate_ideal = solve(eqstate_guess, *Sargs, ideal=True)["x"]
#     eqstate = solve(eqstate_ideal, *Sargs, ideal=False)
#     return eqstate


# def _oosetup(eqstate_guess, tots, eles, fixions, tempK, pres, prmlib):
#     """Prepare for solveloop and solvepool."""
#     eqstates = full((len(tempK), len(eqstate_guess)), 0.0)
#     allions = properties.getallions(eles, fixions)
#     allmols = full((len(tempK), len(allions)), 0.0)
#     lnks = full((len(eqstate_guess), len(tempK)), nan)
#     q = 0
#     for ele in eles:
#         if ele == "t_HSO4":
#             lnks[q] = prmlib.lnk["HSO4"](tempK)
#             q += 1
#         elif ele == "t_Mg":
#             lnks[q] = prmlib.lnk["MgOH"](tempK)
#             q += 1
#         elif ele == "t_trisH":
#             lnks[q] = prmlib.lnk["trisH"](tempK)
#             q += 1
#         elif ele == "t_H2CO3":
#             lnks[q] = prmlib.lnk["H2CO3"](tempK)
#             lnks[q + 1] = prmlib.lnk["HCO3"](tempK)
#             q += 2
#         elif ele == "t_BOH3":
#             lnks[q] = prmlib.lnk["BOH3"](tempK)
#             q += 1
#     if len(eqstate_guess) == q + 1:
#         lnks[-1] = prmlib.lnk["H2O"](tempK)
#     return eqstates, allmols, allions, lnks


# def solveloop(
#     eqstate_guess, tots, fixmols, eles, fixions, tempK, pres, prmlib=libraries.Seawater
# ):
#     """Run solver through a loop of input data."""
#     eqstates, allmols, allions, lnks = _oosetup(
#         eqstate_guess, tots, eles, fixions, tempK, pres, prmlib
#     )
#     for L in range(len(tempK)):
#         print("Solving {} of {}...".format(L + 1, len(tempK)))
#         allmxs = matrix.assemble(allions, array([tempK[L]]), array([pres[L]]), prmlib)
#         if len(eles) > 0:
#             tots1 = tots[:, L]
#         else:
#             tots1 = tots
#         if len(fixions) > 0:
#             fixmols1 = fixmols[:, L]
#         else:
#             fixmols1 = fixmols
#         lnks1 = lnks[:, L]
#         Largs = (eqstate_guess, tots1, fixmols1, eles, allions, fixions, allmxs, lnks1)
#         if L == 0:
#             eqstates[L] = solvequick(*Largs)["x"]
#         else:
#             eqstates[L] = solve(*Largs)["x"]
#         eqstate_guess = eqstates[L]
#         allmols[L] = eqstate2mols(eqstates[L], tots1, fixmols1, eles, fixions)[0]
#     print("Solving complete!")
#     return transpose(allmols), allions, eqstates


# def eqstate2mols(eqstate, tots1, fixmols1, eles, fixions):
#     """Convert eqstate solution to arrays required for Pytzer functions."""
#     fixcharges = transpose(properties.charges(fixions)[0])
#     (
#         mH,
#         mOH,
#         mHSO4,
#         mSO4,
#         mMg,
#         mMgOH,
#         mtris,
#         mtrisH,
#         mCO2,
#         mHCO3,
#         mCO3,
#         mBOH3,
#         mBOH4,
#     ) = _varmols(eqstate, tots1, fixmols1, eles, fixions, fixcharges)
#     allions = properties.getallions(eles, fixions)
#     allmols = full_like(allions, 0.0, dtype="float64")
#     for i, ion in enumerate(allions):
#         if len(fixions) > 0:
#             if ion in fixions:
#                 allmols[i] = fixmols1[fixions == ion]
#         if ion not in fixions:
#             if ion == "H":
#                 allmols[i] = mH
#             elif ion == "OH":
#                 allmols[i] = mOH
#             elif ion == "HSO4":
#                 allmols[i] = mHSO4
#             elif ion == "SO4":
#                 allmols[i] = mSO4
#             elif ion == "Mg":
#                 allmols[i] = mMg
#             elif ion == "MgOH":
#                 allmols[i] = mMgOH
#             elif ion == "tris":
#                 allmols[i] = mtris
#             elif ion == "trisH":
#                 allmols[i] = mtrisH
#             elif ion == "CO2":
#                 allmols[i] = mCO2
#             elif ion == "HCO3":
#                 allmols[i] = mHCO3
#             elif ion == "CO3":
#                 allmols[i] = mCO3
#             elif ion == "BOH3":
#                 allmols[i] = mBOH3
#             elif ion == "BOH4":
#                 allmols[i] = mBOH4
#     return allmols, allions
