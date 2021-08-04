# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
from . import ParameterLibrary
from .. import debyehueckel, dissociation as k, parameters as prm, unsymmetrical

Millero98 = ParameterLibrary(name="Millero98")
Millero98.update_Aphi(debyehueckel.Aosm_M88)
Millero98.assign_func_J(unsymmetrical.Harvie)
# Table A1
Millero98.update_ca("Na", "Cl", prm.bC_Na_Cl_M88)
Millero98.update_ca("K", "Cl", prm.bC_K_Cl_GM89)
Millero98.update_ca("K", "SO4", prm.bC_K_SO4_GM89)
Millero98.update_ca("Ca", "Cl", prm.bC_Ca_Cl_GM89)
Millero98.update_ca("Ca", "SO4", prm.bC_Ca_SO4_M88)
Millero98.update_ca("Ca", "SO3", prm.bC_Ca_SO3_MP98)
Millero98.update_ca("Sr", "SO4", prm.bC_Sr_SO4_MP98)
# Table A2
Millero98.update_ca("Mg", "Cl", prm.bC_Mg_Cl_PP87i)
Millero98.update_ca("Mg", "SO4", prm.bC_Mg_SO4_PP86ii)
# Table A3
Millero98.update_ca("Na", "HSO4", prm.bC_Na_HSO4_MP98)
Millero98.update_ca("Na", "HCO3", prm.bC_Na_HCO3_MP98)
Millero98.update_ca("Na", "SO4", prm.bC_Na_SO4_HPR93)
Millero98.update_ca("Na", "CO3", prm.bC_Na_CO3_MP98)
Millero98.update_ca(
    "Na", "BOH4", prm.bC_Na_BOH4_MP98
)  # Should be SRRJ87, conflicts with MP98.
Millero98.update_ca("Na", "HS", prm.bC_Na_HS_HPM88)
Millero98.update_ca("Na", "SCN", prm.bC_Na_SCN_SP78)
Millero98.update_ca("Na", "SO3", prm.bC_Na_SO3_MHJZ89)
Millero98.update_ca("Na", "HSO3", prm.bC_Na_HSO3_MHJZ89)
# Table A4
Millero98.update_ca("K", "HCO3", prm.bC_K_HCO3_RGWW83)
Millero98.update_ca(
    "K", "CO3", prm.bC_K_CO3_MP98
)  # Should be SRG87, but conflicts with MP98.
Millero98.update_ca(
    "K", "BOH4", prm.bC_K_BOH4_MP98
)  # Should be SRRJ87, but conflicts with MP98.
Millero98.update_ca("K", "HS", prm.bC_K_HS_HPM88)
Millero98.update_ca("K", "H2PO4", prm.bC_K_H2PO4_SP78)
Millero98.update_ca("K", "SCN", prm.bC_K_SCN_SP78)
# Table A5
Millero98.update_ca("Mg", "Br", prm.bC_Mg_Br_SP78)
Millero98.update_ca("Mg", "BOH4", prm.bC_Mg_BOH4_SRM87)
Millero98.update_ca("Mg", "ClO4", prm.bC_Mg_ClO4_SP78)
Millero98.update_ca("Ca", "Br", prm.bC_Ca_Br_SP78)
Millero98.update_ca("Ca", "BOH4", prm.bC_Ca_BOH4_SRM87)
Millero98.update_ca("Ca", "ClO4", prm.bC_Ca_ClO4_SP78)
# Table A6
Millero98.update_ca("Sr", "Br", prm.bC_Sr_Br_SP78)
Millero98.update_ca("Sr", "Cl", prm.bC_Sr_Cl_SP78)  # Not in table but in text §4.6
Millero98.update_ca("Sr", "NO3", prm.bC_Sr_NO3_SP78)
Millero98.update_ca("Sr", "ClO4", prm.bC_Sr_ClO4_SP78)
# Millero98.update_ca("Sr", "HSO3", prm.bC_Sr_HSO3_MP98) # Interaction also appears in Table A8?!
Millero98.update_ca("Sr", "BOH4", prm.bC_Sr_BOH4_MP98)
# Table A7
Millero98.update_ca("Na", "I", prm.bC_Na_I_MP98)
Millero98.update_ca("Na", "Br", prm.bC_Na_Br_MP98)
Millero98.update_ca("Na", "F", prm.bC_Na_F_MP98)
Millero98.update_ca("K", "Br", prm.bC_K_Br_MP98)
Millero98.update_ca("K", "F", prm.bC_K_F_MP98)
Millero98.update_ca("K", "OH", prm.bC_K_OH_MP98)
Millero98.update_ca("K", "I", prm.bC_K_I_MP98)
Millero98.update_ca("Na", "ClO3", prm.bC_Na_ClO3_MP98)
Millero98.update_ca("K", "ClO3", prm.bC_K_ClO3_MP98)
Millero98.update_ca("Na", "ClO4", prm.bC_Na_ClO4_MP98)
Millero98.update_ca("Na", "BrO3", prm.bC_Na_BrO3_MP98)
Millero98.update_ca("K", "BrO3", prm.bC_K_BrO3_MP98)
Millero98.update_ca("Na", "NO3", prm.bC_Na_NO3_MP98)
Millero98.update_ca("K", "NO3", prm.bC_K_NO3_MP98)
Millero98.update_ca("Mg", "NO3", prm.bC_Mg_NO3_MP98)
Millero98.update_ca("Ca", "NO3", prm.bC_Ca_NO3_MP98)
Millero98.update_ca("H", "Br", prm.bC_H_Br_MP98)
Millero98.update_ca("Sr", "Cl", prm.bC_Sr_Cl_MP98)
Millero98.update_ca("NH4", "Cl", prm.bC_NH4_Cl_MP98)
Millero98.update_ca("NH4", "Br", prm.bC_NH4_Br_MP98typo)
Millero98.update_ca("NH4", "F", prm.bC_NH4_F_MP98)
# Table A8
Millero98.update_ca("Sr", "I", prm.bC_Sr_I_PM73)
Millero98.update_ca("Na", "NO2", prm.bC_Na_NO2_PM73)
Millero98.update_ca("Na", "H2PO4", prm.bC_Na_H2PO4_PM73)
Millero98.update_ca("Na", "HPO4", prm.bC_Na_HPO4_PM73)
Millero98.update_ca("Na", "PO4", prm.bC_Na_PO4_PM73)
Millero98.update_ca("Na", "H2AsO4", prm.bC_Na_H2AsO4_PM73)
Millero98.update_ca("K", "H2AsO4", prm.bC_K_H2AsO4_PM73)
Millero98.update_ca("Na", "HAsO4", prm.bC_Na_HAsO4_PM73)
Millero98.update_ca("Na", "AsO4", prm.bC_Na_AsO4_PM73)
Millero98.update_ca("Na", "acetate", prm.bC_Na_acetate_PM73)
Millero98.update_ca(
    "K", "HSO4", prm.bC_K_HSO4_MP98
)  # MP98 mis-citation based on PM16 code.
Millero98.update_ca("K", "NO2", prm.bC_K_NO2_PM73)
Millero98.update_ca("K", "HPO4", prm.bC_K_HPO4_PM73)
Millero98.update_ca("K", "PO4", prm.bC_K_PO4_PM73)
Millero98.update_ca("K", "HAsO4", prm.bC_K_HAsO4_PM73)
Millero98.update_ca(
    "K", "AsO4", prm.bC_K_AsO4_PM73
)  # Not in table, but presumably should be?
Millero98.update_ca("K", "acetate", prm.bC_K_acetate_PM73)
Millero98.update_ca(
    "Mg", "HSO4", prm.bC_Mg_HSO4_MP98
)  # MP98 mis-citation based on PM16 code.
Millero98.update_ca("Mg", "HCO3", prm.bC_Mg_HCO3_MP98)
Millero98.update_ca("Mg", "HS", prm.bC_Mg_HS_HPM88)
Millero98.update_ca("Mg", "I", prm.bC_Mg_I_PM73)
Millero98.update_ca("Mg", "HSO3", prm.bC_Mg_HSO3_RZM91)
Millero98.update_ca("Mg", "SO3", prm.bC_Mg_SO3_RZM91)
Millero98.update_ca("Ca", "HSO4", prm.bC_Ca_HSO4_HMW84)
Millero98.update_ca("Ca", "HCO3", prm.bC_Ca_HCO3_HMW84)
Millero98.update_ca("Ca", "HSO3", prm.bC_Ca_HSO3_MP98)
Millero98.update_ca("Ca", "HS", prm.bC_Ca_HS_HPM88)
Millero98.update_ca("Ca", "OH", prm.bC_Ca_OH_HMW84)
Millero98.update_ca("Ca", "I", prm.bC_Ca_I_PM73)
Millero98.update_ca("Sr", "HSO4", prm.bC_Sr_HSO4_MP98)
Millero98.update_ca("Sr", "HCO3", prm.bC_Sr_HCO3_MP98)
Millero98.update_ca("Sr", "HSO3", prm.bC_Sr_HSO3_MP98)
Millero98.update_ca("Sr", "OH", prm.bC_Sr_OH_MP98)
Millero98.update_ca("NH4", "SO4", prm.bC_NH4_SO4_PM73)
Millero98.update_ca("MgOH", "Cl", prm.bC_MgOH_Cl_HMW84)
# Table A9
Millero98.update_ca("H", "Cl", prm.bC_H_Cl_CMR93)
Millero98.update_ca("H", "SO4", prm.bC_H_SO4_MP98)  # Cited paper not published.
Millero98.update_ca("H", "HSO4", prm.bC_H_HSO4_MP98)
# ^ Cited paper not published, & no equation provided, but this interaction is too
# critical to skip?
Millero98.update_ca("Na", "OH", prm.bC_Na_OH_PP87i)
# ^ Not in MP98 tables (I think!) but should be, based on PM code
# Table A10
Millero98.update_cc("H", "Sr", prm.theta_H_Sr_RGRG86)
Millero98.update_cc(
    "H", "Na", prm.theta_H_Na_MP98
)  # Should be CMR93, but conflicts with MP98
Millero98.update_cc(
    "H", "K", prm.theta_H_K_MP98
)  # Should be CMR93, but conflicts with MP98
Millero98.update_cc(
    "H", "Mg", prm.theta_H_Mg_MP98
)  # Should be RGB80, but has not temp. term
Millero98.update_cc(
    "Ca", "H", prm.theta_Ca_H_MP98
)  # Should be RGO81, see notes in parameters
Millero98.update_cc("K", "Na", prm.theta_K_Na_GM89)
Millero98.update_cc("Mg", "Na", prm.theta_Mg_Na_PP87ii)
Millero98.update_cc("Ca", "Na", prm.theta_Ca_Na_M88)
Millero98.update_cc("K", "Mg", prm.theta_K_Mg_PP87ii)
Millero98.update_cc("Ca", "K", prm.theta_Ca_K_GM89)
Millero98.update_aa("Cl", "SO4", prm.theta_Cl_SO4_M88)
Millero98.update_aa("CO3", "Cl", prm.theta_CO3_Cl_PP82)
Millero98.update_aa("Cl", "HCO3", prm.theta_Cl_HCO3_PP82)
Millero98.update_aa(
    "BOH4", "Cl", prm.theta_BOH4_Cl_MP98typo
)  # Typo in PM code replicated here
Millero98.update_aa("CO3", "HCO3", prm.theta_CO3_HCO3_MP98)
Millero98.update_aa("HSO4", "SO4", prm.theta_HSO4_SO4_MP98)  # Cited paper not published
Millero98.update_aa("Cl", "OH", prm.theta_Cl_OH_MP98)
# Table A11
Millero98.update_cc(
    "Na", "Sr", prm.theta_Na_Sr_MP98
)  # Unclear where MP98 got this from
Millero98.update_cc("K", "Sr", prm.theta_K_Sr_MP98)  # Unclear where MP98 got this from
Millero98.update_cc("Ca", "Mg", prm.theta_Ca_Mg_HMW84)
Millero98.update_aa("Cl", "F", prm.theta_Cl_F_MP98)
Millero98.update_aa("CO3", "SO4", prm.theta_CO3_SO4_HMW84)
Millero98.update_aa("HCO3", "SO4", prm.theta_HCO3_SO4_HMW84)
Millero98.update_aa("BOH4", "SO4", prm.theta_BOH4_SO4_FW86)
Millero98.update_aa("Cl", "HSO4", prm.theta_Cl_HSO4_HMW84)
Millero98.update_aa("OH", "SO4", prm.theta_OH_SO4_HMW84)  # MP98 incorrect citation.
Millero98.update_aa("Br", "OH", prm.theta_Br_OH_PK74)
Millero98.update_aa("Cl", "NO3", prm.theta_Cl_NO3_PK74)
Millero98.update_aa("Cl", "H2PO4", prm.theta_Cl_H2PO4_HFM89)
Millero98.update_aa("Cl", "HPO4", prm.theta_Cl_HPO4_HFM89)
Millero98.update_aa("Cl", "PO4", prm.theta_Cl_PO4_HFM89)
Millero98.update_aa(
    "Cl", "H2AsO4", prm.theta_Cl_H2AsO4_M83
)  # Shouldn't have unsymmetrical term.
Millero98.update_aa(
    "Cl", "HAsO4", prm.theta_Cl_HAsO4_M83
)  # Shouldn't have unsymmetrical term.
Millero98.update_aa(
    "AsO4", "Cl", prm.theta_AsO4_Cl_M83
)  # Shouldn't have unsymmetrical term.
Millero98.update_aa("Cl", "SO3", prm.theta_Cl_SO3_MHJZ89)
Millero98.update_aa(
    "acetate", "Cl", prm.theta_acetate_Cl_M83
)  # Shouldn't have unsymm. term.
# Table A10
Millero98.update_cca("K", "Na", "Cl", prm.psi_K_Na_Cl_GM89)
Millero98.update_cca("K", "Na", "SO4", prm.psi_K_Na_SO4_GM89)
Millero98.update_cca("Mg", "Na", "Cl", prm.psi_Mg_Na_Cl_PP87ii)
Millero98.update_cca("Ca", "Na", "Cl", prm.psi_Ca_Na_Cl_M88)
Millero98.update_cca("Ca", "Na", "SO4", prm.psi_Ca_Na_SO4_M88)
Millero98.update_cca("K", "Mg", "Cl", prm.psi_K_Mg_Cl_PP87ii)
Millero98.update_cca(
    "Ca", "K", "Cl", prm.psi_Ca_K_Cl_MP98typo
)  # Typo in PM code replicated here.
Millero98.update_cca("Ca", "K", "SO4", prm.psi_Ca_K_SO4_GM89)
Millero98.update_caa("Na", "Cl", "SO4", prm.psi_Na_Cl_SO4_M88)
Millero98.update_caa(
    "K", "Cl", "SO4", prm.psi_K_Cl_SO4_MP98
)  # Should be GM89, conflicts with MP98.
Millero98.update_caa("Ca", "Cl", "SO4", prm.psi_Ca_Cl_SO4_M88)
Millero98.update_caa("Na", "CO3", "Cl", prm.psi_Na_CO3_Cl_TM82)
Millero98.update_caa(
    "Na", "Cl", "HCO3", prm.psi_Na_Cl_HCO3_PP82
)  # MP98 incorrect citation.
Millero98.update_caa("Na", "BOH4", "Cl", prm.psi_Na_BOH4_Cl_MP98)
Millero98.update_caa("Mg", "BOH4", "Cl", prm.psi_Mg_BOH4_Cl_MP98)
Millero98.update_caa("Ca", "BOH4", "Cl", prm.psi_Ca_BOH4_Cl_MP98)
Millero98.update_cca(
    "H", "Sr", "Cl", prm.psi_H_Sr_Cl_MP98
)  # Cites M85 book but can't find it there.
Millero98.update_cca(
    "H", "Mg", "Cl", prm.psi_H_Mg_Cl_MP98
)  # Should be RGB80, but that doesn't
# include MP98 temperature term.
Millero98.update_cca(
    "Ca", "H", "Cl", prm.psi_Ca_H_Cl_MP98
)  # Should be RGO81, but that doesn't
# include MP98 temperature term.
Millero98.update_caa(
    "Na", "HSO4", "SO4", prm.psi_Na_HSO4_SO4_MP98
)  # Cited paper not published.
Millero98.update_caa("Na", "CO3", "HCO3", prm.psi_Na_CO3_HCO3_MP98)
Millero98.update_caa("K", "CO3", "HCO3", prm.psi_K_CO3_HCO3_MP98)
# Table A11
Millero98.update_cca(
    "Na", "Sr", "Cl", prm.psi_Na_Sr_Cl_MP98
)  # Couldn't find in PK74 as cited.
Millero98.update_cca("K", "Sr", "Cl", prm.psi_K_Sr_Cl_MP98)
Millero98.update_cca("K", "Na", "Br", prm.psi_K_Na_Br_PK74)
Millero98.update_cca("Mg", "Na", "SO4", prm.psi_Mg_Na_SO4_HMW84)
Millero98.update_cca("K", "Mg", "SO4", prm.psi_K_Mg_SO4_HMW84)
Millero98.update_cca("Ca", "Mg", "Cl", prm.psi_Ca_Mg_Cl_HMW84)
Millero98.update_cca("Ca", "Mg", "SO4", prm.psi_Ca_Mg_SO4_HMW84)
Millero98.update_cca("H", "Na", "Cl", prm.psi_H_Na_Cl_PMR97)
Millero98.update_cca("H", "Na", "SO4", prm.psi_H_Na_SO4_PMR97)
Millero98.update_cca("H", "Na", "Br", prm.psi_H_Na_Br_PK74)
Millero98.update_cca("H", "K", "Cl", prm.psi_H_K_Cl_HMW84)
Millero98.update_cca("H", "K", "SO4", prm.psi_H_K_SO4_HMW84)
Millero98.update_cca(
    "H", "K", "Br", prm.psi_H_K_Br_MP98
)  # No function in HMW84 (no Br!).
Millero98.update_cca(
    "H", "Mg", "Br", prm.psi_H_Mg_Br_MP98
)  # Couldn't find in PK74 as cited.
Millero98.update_cca("Mg", "MgOH", "Cl", prm.psi_Mg_MgOH_Cl_HMW84)
Millero98.update_caa("Mg", "Cl", "SO4", prm.psi_Mg_Cl_SO4_HMW84)
Millero98.update_caa("Mg", "Cl", "HCO3", prm.psi_Mg_Cl_HCO3_HMW84)
Millero98.update_caa("Na", "Cl", "F", prm.psi_Na_Cl_F_MP98)
Millero98.update_caa("Na", "CO3", "SO4", prm.psi_Na_CO3_SO4_HMW84)
Millero98.update_caa("K", "CO3", "SO4", prm.psi_K_CO3_SO4_HMW84)
Millero98.update_caa("Na", "HCO3", "SO4", prm.psi_Na_HCO3_SO4_HMW84)
Millero98.update_caa("Mg", "HCO3", "SO4", prm.psi_Mg_HCO3_SO4_HMW84)
Millero98.update_caa("Na", "Cl", "HSO4", prm.psi_Na_Cl_HSO4_HMW84)
Millero98.update_caa("K", "HSO4", "SO4", prm.psi_K_HSO4_SO4_HMW84)
Millero98.update_caa(
    "Na", "Cl", "OH", prm.psi_Na_Cl_OH_HMW84
)  # Ref. presumably mislabelled by MP98.
Millero98.update_caa(
    "K", "Cl", "OH", prm.psi_K_Cl_OH_HMW84
)  # Ref. presumably mislabelled by MP98.
Millero98.update_caa(
    "Ca", "Cl", "OH", prm.psi_Ca_Cl_OH_HMW84
)  # Ref. presum. mislabelled by MP98.
Millero98.update_caa(
    "Na", "OH", "SO4", prm.psi_Na_OH_SO4_HMW84
)  # Ref. presumably mislabelled by MP98.
Millero98.update_caa(
    "K", "OH", "SO4", prm.psi_K_OH_SO4_HMW84
)  # Ref. presumably mislabelled by MP98.
Millero98.update_caa("Na", "Br", "OH", prm.psi_Na_Br_OH_PK74)
Millero98.update_caa("K", "Br", "OH", prm.psi_K_Br_OH_PK74)
Millero98.update_caa("Na", "Cl", "NO3", prm.psi_Na_Cl_NO3_PK74)
Millero98.update_caa("K", "Cl", "NO3", prm.psi_K_Cl_NO3_PK74)
Millero98.update_caa("Na", "Cl", "H2PO4", prm.psi_Na_Cl_H2PO4_HFM89)
Millero98.update_caa(
    "K", "Cl", "H2PO4", prm.psi_K_Cl_H2PO4_MP98
)  # Can't find cited PS76 paper.
Millero98.update_caa("Na", "Cl", "HPO4", prm.psi_Na_Cl_HPO4_HFM89)
Millero98.update_caa("Na", "Cl", "PO4", prm.psi_Na_Cl_PO4_HFM89)
Millero98.update_caa("Na", "Cl", "H2AsO4", prm.psi_Na_Cl_H2AsO4_M83)
Millero98.update_caa("Na", "Cl", "HAsO4", prm.psi_Na_Cl_HAsO4_M83)
Millero98.update_caa("Na", "AsO4", "Cl", prm.psi_Na_AsO4_Cl_M83)
Millero98.update_caa("Na", "Cl", "SO3", prm.psi_Na_Cl_SO3_MHJZ89)
# Lambdas all from Table A12
Millero98.update_nc("CO2", "Na", prm.lambd_CO2_Na_HM93)
Millero98.update_nc("CO2", "K", prm.lambd_CO2_K_HM93)
Millero98.update_nc("CO2", "Ca", prm.lambd_CO2_Ca_HM93)
Millero98.update_nc("CO2", "Mg", prm.lambd_CO2_Mg_HM93)
Millero98.update_na("CO2", "Cl", prm.lambd_CO2_Cl_HM93)
Millero98.update_na("CO2", "SO4", prm.lambd_CO2_SO4_HM93)
Millero98.update_nc("BOH3", "Na", prm.lambd_BOH3_Na_FW86)
Millero98.update_nc("BOH3", "K", prm.lambd_BOH3_K_FW86)
Millero98.update_na("BOH3", "Cl", prm.lambd_BOH3_Cl_FW86)
Millero98.update_na("BOH3", "SO4", prm.lambd_BOH3_SO4_FW86)
Millero98.update_nc("NH3", "Na", prm.lambd_NH3_Na_CB89)
Millero98.update_nc(
    "NH3", "K", prm.lambd_NH3_K_CB89
)  # Also includes CB89 temperature term.
Millero98.update_nc("NH3", "Mg", prm.lambd_NH3_Mg_CB89)
Millero98.update_nc("NH3", "Ca", prm.lambd_NH3_Ca_CB89)
Millero98.update_nc("NH3", "Sr", prm.lambd_NH3_Sr_CB89)
Millero98.update_nc("H3PO4", "H", prm.lambd_H3PO4_H_PS76)
Millero98.update_nc("H3PO4", "K", prm.lambd_H3PO4_K_PS76)
Millero98.update_na("SO2", "Cl", prm.lambd_SO2_Cl_MHJZ89)
Millero98.update_nc("SO2", "Na", prm.lambd_SO2_Na_MHJZ89)
Millero98.update_nc("SO2", "Mg", prm.lambd_SO2_Mg_RZM91)
Millero98.update_na("HF", "Cl", prm.lambd_HF_Cl_MP98)
Millero98.update_nc("HF", "Na", prm.lambd_HF_Na_MP98)
# Zetas all from Table A12
Millero98.update_nca("CO2", "H", "Cl", prm.zeta_CO2_H_Cl_HM93)
Millero98.update_nca("CO2", "Na", "Cl", prm.zeta_CO2_Na_Cl_HM93)
Millero98.update_nca("CO2", "K", "Cl", prm.zeta_CO2_K_Cl_HM93)
Millero98.update_nca("CO2", "Ca", "Cl", prm.zeta_CO2_Ca_Cl_HM93)
Millero98.update_nca("CO2", "Mg", "Cl", prm.zeta_CO2_Mg_Cl_HM93)
Millero98.update_nca("CO2", "Na", "SO4", prm.zeta_CO2_Na_SO4_HM93)
Millero98.update_nca("CO2", "K", "SO4", prm.zeta_CO2_K_SO4_HM93)
Millero98.update_nca("CO2", "Mg", "SO4", prm.zeta_CO2_Mg_SO4_HM93)
Millero98.update_nca("BOH3", "Na", "SO4", prm.zeta_BOH3_Na_SO4_FW86)
Millero98.update_nca("NH3", "Ca", "Cl", prm.zeta_NH3_Ca_Cl_CB89)
Millero98.update_nca(
    "H3PO4", "Na", "Cl", prm.zeta_H3PO4_Na_Cl_MP98
)  # PS76 don't have this term...
# Dissociation constants
Millero98.update_equilibrium("BOH3", k.BOH3_M79)
Millero98.update_equilibrium("MgOH", k.MgOH_MP98)
# Millero98.update_equilibrium("H2O", k.H2O_M79)  # maybe?
Millero98.update_equilibrium("H2O", k.H2O_M88)
Millero98.update_equilibrium("H2CO3", k.H2CO3_MP98)
Millero98.update_equilibrium("HCO3", k.HCO3_MP98)
Millero98.update_equilibrium("HF", k.HF_MP98)
Millero98.update_equilibrium("H3PO4", k.H3PO4_MP98)
Millero98.update_equilibrium("H2PO4", k.H2PO4_MP98)
Millero98.update_equilibrium("HPO4", k.HPO4_MP98)
Millero98.update_equilibrium("H2S", k.H2S_MP98)
Millero98.update_equilibrium("HSO4", k.HSO4_CRP94)
Millero98.update_equilibrium("NH4", k.NH4_MP98)
Millero98.update_equilibrium("CaCO3", k.CaCO3_MP98_MR97)
Millero98.update_equilibrium("MgCO3", k.MgCO3_MP98_MR97)
Millero98.update_equilibrium("SrCO3", k.SrCO3_MP98_MR97)
Millero98.update_equilibrium("CaH2PO4", k.CaH2PO4_MP98_MR97)
Millero98.update_equilibrium("MgH2PO4", k.MgH2PO4_MP98_MR97)
Millero98.update_equilibrium("CaHPO4", k.CaHPO4_MP98_MR97)
Millero98.update_equilibrium("MgHPO4", k.MgHPO4_MP98_MR97)
Millero98.update_equilibrium("CaPO4", k.CaPO4_MP98_MR97)
Millero98.update_equilibrium("MgPO4", k.MgPO4_MP98_MR97)
Millero98.update_equilibrium("CaF", k.CaF_MP98_MR97)
Millero98.update_equilibrium("MgF", k.MgF_MP98_MR97)
