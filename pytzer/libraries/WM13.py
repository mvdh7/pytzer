# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
from .. import parameters as prm
from .. import debyehueckel, unsymmetrical
# Waters and Millero (2013). Mar. Chem. 149, 8-22,
#  doi:10.1016/j.marchem.2012.11.003
name = 'WM13'
dh = {'Aosm': debyehueckel.Aosm_M88}
jfunc = unsymmetrical.Harvie
bC = {}
theta = {}
psi = {}
# Table A1: Na salts
bC['Na-Cl'] = prm.bC_Na_Cl_M88
bC['Na-SO4'] = prm.bC_Na_SO4_HM86
bC['Na-HSO4'] = prm.bC_Na_HSO4_HPR93
bC['Na-OH'] = prm.bC_Na_OH_PP87i
# Table A2: Mg salts
bC['Mg-Cl'] = prm.bC_Mg_Cl_dLP83
bC['Mg-SO4'] = prm.bC_Mg_SO4_PP86ii
bC['Mg-HSO4'] = prm.bC_Mg_HSO4_RC99
# Table A3: Ca salts
bC['Ca-Cl'] = prm.bC_Ca_Cl_GM89
bC['Ca-SO4'] = prm.bC_Ca_SO4_WM13
bC['Ca-HSO4'] = prm.bC_Ca_HSO4_WM13
bC['Ca-OH'] = prm.bC_Ca_OH_HMW84
# Table A4: K salts
bC['K-Cl'] = prm.bC_K_Cl_GM89
bC['K-SO4'] = prm.bC_K_SO4_HM86
bC['K-HSO4'] = prm.bC_K_HSO4_WM13
bC['K-OH'] = prm.bC_K_OH_HMW84
# Table A5: H+ interactions
bC['H-Cl'] = prm.bC_H_Cl_CMR93
bC['H-SO4'] = prm.bC_H_SO4_CRP94
bC['H-HSO4'] = prm.bC_H_HSO4_CRP94
# Table A6: MgOH+ interactions
bC['MgOH-Cl'] = prm.bC_MgOH_Cl_HMW84
# Table A7: cation-cation interactions
theta['H-Na'] = prm.theta_H_Na_CMR93
theta['H-Mg'] = prm.theta_H_Mg_RGB80
theta['Ca-H'] = prm.theta_Ca_H_RGO81 # WM13 citation error
theta['H-K'] = prm.theta_H_K_CMR93
theta['Mg-Na'] = prm.theta_Mg_Na_HMW84
theta['Ca-Na'] = prm.theta_Ca_Na_HMW84
theta['K-Na'] = prm.theta_K_Na_HMW84
theta['Ca-Mg'] = prm.theta_Ca_Mg_HMW84
theta['K-Mg'] = prm.theta_K_Mg_HMW84
theta['Ca-K'] = prm.theta_Ca_K_HMW84
# Table A7: anion-anion interactions
theta['Cl-SO4'] = prm.theta_Cl_SO4_HMW84
theta['Cl-HSO4'] = prm.theta_Cl_HSO4_HMW84
theta['Cl-OH'] = prm.theta_Cl_OH_HMW84
theta['HSO4-SO4'] = prm.theta_HSO4_SO4_WM13
theta['OH-SO4'] = prm.theta_OH_SO4_HMW84
# Table A8: c-a-a' triplets
psi['H-Cl-SO4'] = prm.psi_H_Cl_SO4_WM13 # agrees with HMW84
psi['Na-Cl-SO4'] = prm.psi_Na_Cl_SO4_HMW84
psi['Mg-Cl-SO4'] = prm.psi_Mg_Cl_SO4_HMW84
psi['Ca-Cl-SO4'] = prm.psi_Ca_Cl_SO4_HMW84
psi['K-Cl-SO4'] = prm.psi_K_Cl_SO4_HMW84
psi['H-Cl-HSO4'] = prm.psi_H_Cl_HSO4_HMW84
psi['Na-Cl-HSO4'] = prm.psi_Na_Cl_HSO4_HMW84
psi['Mg-Cl-HSO4'] = prm.psi_Mg_Cl_HSO4_HMW84
psi['Ca-Cl-HSO4'] = prm.psi_Ca_Cl_HSO4_HMW84
psi['K-Cl-HSO4'] = prm.psi_K_Cl_HSO4_HMW84
psi['H-Cl-OH'] = prm.psi_H_Cl_OH_WM13 # agrees with HMW84
psi['Na-Cl-OH'] = prm.psi_Na_Cl_OH_HMW84
psi['Mg-Cl-OH'] = prm.psi_Mg_Cl_OH_WM13 # agrees with HMW84
psi['Ca-Cl-OH'] = prm.psi_Ca_Cl_OH_HMW84
psi['K-Cl-OH'] = prm.psi_K_Cl_OH_HMW84
psi['H-HSO4-SO4'] = prm.psi_H_HSO4_SO4_HMW84
psi['Na-HSO4-SO4'] = prm.psi_Na_HSO4_SO4_HMW84
psi['Mg-HSO4-SO4'] = prm.psi_Mg_HSO4_SO4_RC99
psi['Ca-HSO4-SO4'] = prm.psi_Ca_HSO4_SO4_WM13 # agrees with HMW84
psi['K-HSO4-SO4'] = prm.psi_K_HSO4_SO4_HMW84
psi['H-OH-SO4'] = prm.psi_H_OH_SO4_WM13 # agrees with HMW84
psi['Na-OH-SO4'] = prm.psi_Na_OH_SO4_HMW84
psi['Mg-OH-SO4'] = prm.psi_Mg_OH_SO4_WM13 # agrees with HMW84
psi['Ca-OH-SO4'] = prm.psi_Ca_OH_SO4_WM13 # agrees with HMW84
psi['K-OH-SO4'] = prm.psi_K_OH_SO4_HMW84
# Table A9: c-c'-a triplets
psi['H-Na-Cl'] = prm.psi_H_Na_Cl_HMW84
psi['H-Na-SO4'] = prm.psi_H_Na_SO4_WM13 # agrees with HMW84
psi['H-Na-HSO4'] = prm.psi_H_Na_HSO4_HMW84
psi['H-Mg-Cl'] = prm.psi_H_Mg_Cl_HMW84
psi['H-Mg-SO4'] = prm.psi_H_Mg_SO4_RC99
psi['H-Mg-HSO4'] = prm.psi_H_Mg_HSO4_RC99
psi['Ca-H-Cl'] = prm.psi_Ca_H_Cl_HMW84
psi['Ca-H-SO4'] = prm.psi_Ca_H_SO4_WM13 # agrees with HMW84
psi['Ca-H-HSO4'] = prm.psi_Ca_H_HSO4_WM13 # agrees with HMW84
psi['H-K-Cl'] = prm.psi_H_K_Cl_HMW84
psi['H-K-SO4'] = prm.psi_H_K_SO4_HMW84
psi['H-K-HSO4'] = prm.psi_H_K_HSO4_HMW84
psi['Mg-Na-Cl'] = prm.psi_Mg_Na_Cl_HMW84
psi['Mg-Na-SO4'] = prm.psi_Mg_Na_SO4_HMW84
psi['Mg-Na-HSO4'] = prm.psi_Mg_Na_HSO4_WM13 # agrees with HMW84
psi['Ca-Na-Cl'] = prm.psi_Ca_Na_Cl_HMW84
psi['Ca-Na-SO4'] = prm.psi_Ca_Na_SO4_HMW84
psi['Ca-Na-HSO4'] = prm.psi_Ca_Na_HSO4_WM13 # agrees with HMW84
psi['K-Na-Cl'] = prm.psi_K_Na_Cl_HMW84
psi['K-Na-SO4'] = prm.psi_K_Na_SO4_HMW84
psi['K-Na-HSO4'] = prm.psi_K_Na_HSO4_WM13 # agrees with HMW84
psi['Ca-Mg-Cl'] = prm.psi_Ca_Mg_Cl_HMW84
psi['Ca-Mg-SO4'] = prm.psi_Ca_Mg_SO4_HMW84
psi['Ca-Mg-HSO4'] = prm.psi_Ca_Mg_HSO4_WM13 # agrees with HMW84
psi['K-Mg-Cl'] = prm.psi_K_Mg_Cl_HMW84
psi['K-Mg-SO4'] = prm.psi_K_Mg_SO4_HMW84
psi['K-Mg-HSO4'] = prm.psi_K_Mg_HSO4_WM13 # agrees with HMW84
psi['Ca-K-Cl'] = prm.psi_Ca_K_Cl_HMW84
psi['Ca-K-SO4'] = prm.psi_Ca_K_SO4_WM13 # agrees with HMW84
psi['Ca-K-HSO4'] = prm.psi_Ca_K_HSO4_WM13 # agrees with HMW84
