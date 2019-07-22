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
#    'Sr-HSO3': prm.bC_Sr_HSO3_SP78,
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
#    'Na-acetate': prm.bC_Na_acetate_PM73,
    'K-HSO4': prm.bC_K_HSO4_HMW84,
    'K-NO2': prm.bC_K_NO2_PM73,
    'K-HPO4': prm.bC_K_HPO4_PM73,
    'K-PO4': prm.bC_K_PO4_PM73,
    'K-HAsO4': prm.bC_K_HAsO4_PM73,
    'K-AsO4': prm.bC_K_AsO4_PM73, # not in table, but presumably should be?
#    'K-acetate': prm.bC_K_acetate_PM73,
    'Mg-HSO4': prm.bC_Mg_HSO4_HMW84,
#    'Mg-HCO3': prm.bC_Mg_HCO3_TM82,
    'Mg-HS': prm.bC_Mg_HS_HPM88,
    'Mg-I': prm.bC_Mg_I_PM73,
#    'Mg-HSO3': prm.bC_Mg_HSO3_HPM88,
#    'Mg-SO3': prm.bC_Mg_SO3_HPM88,
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
#    'H-SO4': prm.bC_H_SO4_Pierrot,
} # end of bC dict

theta = {
# Table A10
    'Cl-CO3': prm.theta_Cl_CO3_PP82,
    'Cl-HCO3': prm.theta_Cl_HCO3_PP82,
    'Cl-SO3': prm.theta_Cl_SO3_MHJZ89,
} # end of theta dict

psi = {
# Table A11
    'Na-Cl-SO3': prm.psi_Na_Cl_SO3_MHJZ89,
} # end of psi dict
