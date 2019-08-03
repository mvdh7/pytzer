# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
from .. import parameters as prm
from .. import debyehueckel, unsymmetrical
name = 'MIAMI'
dh = {'Aosm': debyehueckel.Aosm_M88}
jfunc = unsymmetrical.Harvie
bC = {
# Table A1
    'Na-Cl': prm.bC_Na_Cl_M88,
    'K-Cl': prm.bC_K_Cl_GM89,
    'K-SO4': prm.bC_K_SO4_GM89,
    'Ca-Cl': prm.bC_Ca_Cl_GM89,
    'Ca-SO4': prm.bC_Ca_SO4_M88,
    'Ca-SO3': prm.bC_Ca_SO3_MP98,
    'Sr-SO4': prm.bC_Sr_SO4_MP98,
# Table A2
    'Mg-Cl': prm.bC_Mg_Cl_PP87i,
    'Mg-SO4': prm.bC_Mg_SO4_PP86ii,
# Table A3
    'Na-HSO4': prm.bC_Na_HSO4_MP98,
    'Na-HCO3': prm.bC_Na_HCO3_PP82,
    'Na-SO4': prm.bC_Na_SO4_HPR93,
    'Na-CO3': prm.bC_Na_CO3_PP82,
    'Na-BOH4': prm.bC_Na_BOH4_SRRJ87,
    'Na-HS': prm.bC_Na_HS_HPM88,
    'Na-SCN': prm.bC_Na_SCN_SP78,
    'Na-SO3': prm.bC_Na_SO3_MHJZ89,
    'Na-HSO3': prm.bC_Na_HSO3_MHJZ89,
# Table A4
    'K-HCO3': prm.bC_K_HCO3_RGWW83,
    'K-CO3': prm.bC_K_CO3_SRG87,
    'K-BOH4': prm.bC_K_BOH4_SRRJ87,
    'K-HS': prm.bC_K_HS_HPM88,
    'K-H2PO4': prm.bC_K_H2PO4_SP78,
    'K-SCN': prm.bC_K_SCN_SP78,
# Table A5
    'Mg-Br': prm.bC_Mg_Br_SP78,
    'Mg-BOH4': prm.bC_Mg_BOH4_SRM87,
    'Mg-ClO4': prm.bC_Mg_ClO4_SP78,
    'Ca-Br': prm.bC_Ca_Br_SP78,
    'Ca-BOH4': prm.bC_Ca_BOH4_SRM87,
    'Ca-ClO4': prm.bC_Ca_ClO4_SP78,
# Table A6
    'Sr-Br': prm.bC_Sr_Br_SP78,
    'Sr-Cl': prm.bC_Sr_Cl_SP78, # not in table but in text §4.6
    'Sr-NO3': prm.bC_Sr_NO3_SP78,
    'Sr-ClO4': prm.bC_Sr_ClO4_SP78,
#    'Sr-HSO3': prm.bC_Sr_HSO3_MP98, # interaction also appears in Table A8?!
    'Sr-BOH4': prm.bC_Sr_BOH4_MP98,
# Table A7
    'Na-I': prm.bC_Na_I_MP98,
    'Na-Br': prm.bC_Na_Br_MP98,
    'Na-F': prm.bC_Na_F_MP98,
    'K-Br': prm.bC_K_Br_MP98,
    'K-F': prm.bC_K_F_MP98,
    'K-OH': prm.bC_K_OH_MP98,
    'K-I': prm.bC_K_I_MP98,
    'Na-ClO3': prm.bC_Na_ClO3_MP98,
    'K-ClO3': prm.bC_K_ClO3_MP98,
    'Na-ClO4': prm.bC_Na_ClO4_MP98,
    'Na-BrO3': prm.bC_Na_BrO3_MP98,
    'K-BrO3': prm.bC_K_BrO3_MP98,
    'Na-NO3': prm.bC_Na_NO3_MP98,
    'K-NO3': prm.bC_K_NO3_MP98,
    'Mg-NO3': prm.bC_Mg_NO3_MP98,
    'Ca-NO3': prm.bC_Ca_NO3_MP98,
    'H-Br': prm.bC_H_Br_MP98,
    'Sr-Cl': prm.bC_Sr_Cl_MP98,
    'NH4-Cl': prm.bC_NH4_Cl_MP98,
    'NH4-Br': prm.bC_NH4_Br_MP98,
    'NH4-F': prm.bC_NH4_F_MP98,
# Table A8
    'Sr-I': prm.bC_Sr_I_PM73,
    'Na-NO2': prm.bC_Na_NO2_PM73,
    'Na-H2PO4': prm.bC_Na_H2PO4_PM73,
    'Na-HPO4': prm.bC_Na_HPO4_PM73,
    'Na-PO4': prm.bC_Na_PO4_PM73,
    'Na-H2AsO4': prm.bC_Na_H2AsO4_PM73,
    'K-H2AsO4': prm.bC_K_H2AsO4_PM73,
    'Na-HAsO4': prm.bC_Na_HAsO4_PM73,
    'Na-AsO4': prm.bC_Na_AsO4_PM73,
    'Na-acetate': prm.bC_Na_acetate_PM73,
    'K-HSO4': prm.bC_K_HSO4_HMW84,
    'K-NO2': prm.bC_K_NO2_PM73,
    'K-HPO4': prm.bC_K_HPO4_PM73,
    'K-PO4': prm.bC_K_PO4_PM73,
    'K-HAsO4': prm.bC_K_HAsO4_PM73,
    'K-AsO4': prm.bC_K_AsO4_PM73, # not in table, but presumably should be?
    'K-acetate': prm.bC_K_acetate_PM73,
    'Mg-HSO4': prm.bC_Mg_HSO4_HMW84,
    'Mg-HCO3': prm.bC_Mg_HCO3_MP98,
    'Mg-HS': prm.bC_Mg_HS_HPM88,
    'Mg-I': prm.bC_Mg_I_PM73,
    'Mg-HSO3': prm.bC_Mg_HSO3_RZM91,
    'Mg-SO3': prm.bC_Mg_SO3_RZM91,
    'Ca-HSO4': prm.bC_Ca_HSO4_HMW84,
    'Ca-HCO3': prm.bC_Ca_HCO3_HMW84,
    'Ca-HSO3': prm.bC_Ca_HSO3_MP98,
    'Ca-HS': prm.bC_Ca_HS_HPM88,
    'Ca-OH': prm.bC_Ca_OH_HMW84,
    'Ca-I': prm.bC_Ca_I_PM73,
    'Sr-HSO4': prm.bC_Sr_HSO4_MP98,
    'Sr-HCO3': prm.bC_Sr_HCO3_MP98,
    'Sr-HSO3': prm.bC_Sr_HSO3_MP98,
    'Sr-OH': prm.bC_Sr_OH_MP98,
    'NH4-SO4': prm.bC_NH4_SO4_PM73,
    'MgOH-Cl': prm.bC_MgOH_Cl_HMW84,
# Table A9
    'H-Cl': prm.bC_H_Cl_CMR93,
    'H-SO4': prm.bC_H_SO4_MP98, # cited paper not published
    'H-HSO4': prm.bC_H_HSO4_MP98, # cited paper not published, & no equation
                                  # provided, but this interaction is too
                                  # critical to skip?
} # end of bC dict
theta = {
# Table A10
    'H-Sr': prm.theta_H_Sr_RGRG86,
    'H-Na': prm.theta_H_Na_CMR93,
    'H-K': prm.theta_H_K_CMR93,
    'H-Mg': prm.theta_H_Mg_RGB80, # doesn't include temperature term from MP98
    'Ca-H': prm.theta_Ca_H_RGO81, # doesn't include temperature term from MP98
    'K-Na': prm.theta_K_Na_GM89,
    'Mg-Na': prm.theta_Mg_Na_PP87ii,
    'Ca-Na': prm.theta_Ca_Na_M88,
    'K-Mg': prm.theta_K_Mg_PP87ii,
    'Ca-K': prm.theta_Ca_K_GM89,
    'Cl-SO4': prm.theta_Cl_SO4_M88,
    'Cl-CO3': prm.theta_Cl_CO3_PP82,
    'Cl-HCO3': prm.theta_Cl_HCO3_PP82,
    'BOH4-Cl': prm.theta_BOH4_Cl_MP98,
    'CO3-HCO3': prm.theta_CO3_HCO3_MP98,
    'HSO4-SO4': prm.theta_HSO4_SO4_MP98, # cited paper not published
    'Cl-OH': prm.theta_Cl_OH_MP98,
# Table A11
    'Na-Sr': prm.theta_Na_Sr_MP98, # unclear where MP98 got this from
    'K-Sr': prm.theta_K_Sr_MP98, # unclear where MP98 got this from
    'Ca-Mg': prm.theta_Ca_Mg_HMW84,
    'Cl-F': prm.theta_Cl_F_MP98,
    'CO3-SO4': prm.theta_CO3_SO4_HMW84,
    'HCO3-SO4': prm.theta_HCO3_SO4_HMW84,
    'BOH4-SO4': prm.theta_BOH4_SO4_FW86,
    'Cl-HSO4': prm.theta_Cl_HSO4_HMW84,
    'OH-SO4': prm.theta_OH_SO4_HMW84, # MP98 incorrect citation
    'Br-OH': prm.theta_Br_OH_PK74,
    'Cl-NO3': prm.theta_Cl_NO3_PK74,
    'Cl-H2PO4': prm.theta_Cl_H2PO4_HFM89,
    'Cl-HPO4': prm.theta_Cl_HPO4_HFM89,
    'Cl-PO4': prm.theta_Cl_PO4_HFM89,
    'Cl-H2AsO4': prm.theta_Cl_H2AsO4_M83, # shouldn't have unsymmetrical term
    'Cl-HAsO4': prm.theta_Cl_HAsO4_M83, # shouldn't have unsymmetrical term
    'AsO4-Cl': prm.theta_AsO4_Cl_M83, # shouldn't have unsymmetrical term
    'Cl-SO3': prm.theta_Cl_SO3_MHJZ89,
    'acetate-Cl': prm.theta_acetate_Cl_M83, # shouldn't have unsymmetrical term
} # end of theta dict
psi = {
# Table A10
    'K-Na-Cl': prm.psi_K_Na_Cl_GM89,
    'K-Na-SO4': prm.psi_K_Na_SO4_GM89,
    'Mg-Na-Cl': prm.psi_Mg_Na_Cl_PP87ii,
    'Ca-Na-Cl': prm.psi_Ca_Na_Cl_M88,
    'Ca-Na-SO4': prm.psi_Ca_Na_SO4_M88,
    'K-Mg-Cl': prm.psi_K_Mg_Cl_PP87ii,
    'Ca-K-Cl': prm.psi_Ca_K_Cl_GM89,
    'Ca-K-SO4': prm.psi_Ca_K_SO4_GM89,
    'Na-Cl-SO4': prm.psi_Na_Cl_SO4_M88,
    'K-Cl-SO4': prm.psi_K_Cl_SO4_GM89,
    'Ca-Cl-SO4': prm.psi_Ca_Cl_SO4_M88,
    'Na-Cl-CO3': prm.psi_Na_Cl_CO3_TM82,
    'Na-Cl-HCO3': prm.psi_Na_Cl_HCO3_PP82, # MP98 incorrect citation
    'Na-BOH4-Cl': prm.psi_Na_BOH4_Cl_MP98,
    'Mg-BOH4-Cl': prm.psi_Mg_BOH4_Cl_MP98,
    'Ca-BOH4-Cl': prm.psi_Ca_BOH4_Cl_MP98,
    'H-Sr-Cl': prm.psi_H_Sr_Cl_MP98, # cites M85 book but can't find it there
    'H-Mg-Cl': prm.psi_H_Mg_Cl_RGB80, # doesn't include MP98 temperature term
    'Ca-H-Cl': prm.psi_Ca_H_Cl_RGO81, # doesn't include MP98 temperature term
    'Na-HSO4-SO4': prm.psi_Na_HSO4_SO4_MP98, # cited paper not published
    'Na-CO3-HCO3': prm.psi_Na_CO3_HCO3_MP98,
    'K-CO3-HCO3': prm.psi_K_CO3_HCO3_MP98,
# Table A11
    'Na-Sr-Cl': prm.psi_Na_Sr_Cl_MP98, # couldn't find in PK74 as cited
    'K-Sr-Cl': prm.psi_K_Sr_Cl_MP98,
    'K-Na-Br': prm.psi_K_Na_Br_PK74,
    'Mg-Na-SO4': prm.psi_Mg_Na_SO4_HMW84,
    'K-Mg-SO4': prm.psi_K_Mg_SO4_HMW84,
    'Ca-Mg-Cl': prm.psi_Ca_Mg_Cl_HMW84,
    'Ca-Mg-SO4': prm.psi_Ca_Mg_SO4_HMW84,
    'H-Na-Cl': prm.psi_H_Na_Cl_PMR97,
    'H-Na-SO4': prm.psi_H_Na_SO4_PMR97,
    'H-Na-Br': prm.psi_H_Na_Br_PK74,
    'H-K-Cl': prm.psi_H_K_Cl_HMW84,
    'H-K-SO4': prm.psi_H_K_SO4_HMW84,
    'H-K-Br': prm.psi_H_K_Br_MP98, # no function in HMW84 (no Br!)
    'H-Mg-Br': prm.psi_H_Mg_Br_MP98, # couldn't find in PK74 as cited
    'Mg-MgOH-Cl': prm.psi_Mg_MgOH_Cl_HMW84,
    'Mg-Cl-SO4': prm.psi_Mg_Cl_SO4_HMW84,
    'Mg-Cl-HCO3': prm.psi_Mg_Cl_HCO3_HMW84,
    'Na-Cl-F': prm.psi_Na_Cl_F_MP98,
    'Na-CO3-SO4': prm.psi_Na_CO3_SO4_HMW84,
    'K-CO3-SO4': prm.psi_K_CO3_SO4_HMW84,
    'Na-HCO3-SO4': prm.psi_Na_HCO3_SO4_HMW84,
    'Mg-HCO3-SO4': prm.psi_Mg_HCO3_SO4_HMW84,
    'Na-Cl-HSO4': prm.psi_Na_Cl_HSO4_HMW84,
    'K-HSO4-SO4': prm.psi_K_HSO4_SO4_HMW84,
    'Na-Cl-OH': prm.psi_Na_Cl_OH_HMW84, # ref. presumably mislabelled by MP98
    'K-Cl-OH': prm.psi_K_Cl_OH_HMW84, # ref. presumably mislabelled by MP98
    'Ca-Cl-OH': prm.psi_Ca_Cl_OH_HMW84, # ref. presumably mislabelled by MP98
    'Na-OH-SO4': prm.psi_Na_OH_SO4_HMW84, # ref. presumably mislabelled by MP98
    'K-OH-SO4': prm.psi_K_OH_SO4_HMW84, # ref. presumably mislabelled by MP98
    'Na-Br-OH': prm.psi_Na_Br_OH_PK74,
    'K-Br-OH': prm.psi_K_Br_OH_PK74,
    'Na-Cl-NO3': prm.psi_Na_Cl_NO3_PK74,
    'K-Cl-NO3': prm.psi_K_Cl_NO3_PK74,
    'Na-Cl-H2PO4': prm.psi_Na_Cl_H2PO4_HFM89,
    'K-Cl-H2PO4': prm.psi_K_Cl_H2PO4_MP98, # can't find cited PS76 paper
    'Na-Cl-HPO4': prm.psi_Na_Cl_HPO4_HFM89,
    'Na-Cl-PO4': prm.psi_Na_Cl_PO4_HFM89,
    'Na-Cl-H2AsO4': prm.psi_Na_Cl_H2AsO4_M83,
    'Na-Cl-HAsO4': prm.psi_Na_Cl_HAsO4_M83,
    'Na-AsO4-Cl': prm.psi_Na_AsO4_Cl_M83,
    'Na-Cl-SO3': prm.psi_Na_Cl_SO3_MHJZ89,
} # end of psi dict
lambd = { # all from Table A12
    'CO2-Na': prm.lambd_CO2_Na_HM93,
    'CO2-K': prm.lambd_CO2_K_HM93,
    'CO2-Ca': prm.lambd_CO2_Ca_HM93,
    'CO2-Mg': prm.lambd_CO2_Mg_HM93,
    'CO2-Cl': prm.lambd_CO2_Cl_HM93,
    'CO2-SO4': prm.lambd_CO2_SO4_HM93,
    'BOH3-Na': prm.lambd_BOH3_Na_FW86,
    'BOH3-K': prm.lambd_BOH3_K_FW86,
    'BOH3-Cl': prm.lambd_BOH3_Cl_FW86,
    'BOH3-SO4': prm.lambd_BOH3_SO4_FW86,
    'NH3-Na': prm.lambd_NH3_Na_CB89,
    'NH3-K': prm.lambd_NH3_K_CB89, # also includes CB89 temperature term
    'NH3-Mg': prm.lambd_NH3_Mg_CB89,
    'NH3-Ca': prm.lambd_NH3_Ca_CB89,
    'NH3-Sr': prm.lambd_NH3_Sr_CB89,
    'H3PO4-H': prm.lambd_H3PO4_H_PS76,
    'H3PO4-K': prm.lambd_H3PO4_K_PS76,
    'SO2-Cl': prm.lambd_SO2_Cl_MHJZ89,
    'SO2-Na': prm.lambd_SO2_Na_MHJZ89,
    'SO2-Mg': prm.lambd_SO2_Mg_RZM91,
    'HF-Cl': prm.lambd_HF_Cl_MP98,
    'HF-Na': prm.lambd_HF_Na_MP98,
} # end of lambd dict
zeta = { # all from Table A12
    'CO2-H-Cl': prm.zeta_CO2_H_Cl_HM93,
    'CO2-Na-Cl': prm.zeta_CO2_Na_Cl_HM93,
    'CO2-K-Cl': prm.zeta_CO2_K_Cl_HM93,
    'CO2-Ca-Cl': prm.zeta_CO2_Ca_Cl_HM93,
    'CO2-Mg-Cl': prm.zeta_CO2_Mg_Cl_HM93,
    'CO2-Na-SO4': prm.zeta_CO2_Na_SO4_HM93,
    'CO2-K-SO4': prm.zeta_CO2_K_SO4_HM93,
    'CO2-Mg-SO4': prm.zeta_CO2_Mg_SO4_HM93,
    'BOH3-Na-SO4': prm.zeta_BOH3_Na_SO4_FW86,
    'NH3-Ca-Cl': prm.zeta_NH3_Ca_Cl_CB89,
    'H3PO4-Na-Cl': prm.zeta_H3PO4_Na_Cl_MP98, # PS76 don't have this term...
} # end of zeta dict
