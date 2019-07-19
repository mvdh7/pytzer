# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
from .. import parameters as prm
from .. import debyehueckel, unsymmetrical
# Waters and Millero (2013). Mar. Chem. 149, 8-22,
#  doi:10.1016/j.marchem.2012.11.003
from . import WM13
name = 'WM13_MarChemSpec25'
dh = {'Aosm': debyehueckel.Aosm_MarChemSpec25}
jfunc = unsymmetrical.P75_eq47
bC = WM13.bC
theta = WM13.theta
theta['H-Na'] = prm.theta_H_Na_MarChemSpec25
theta['H-K' ] = prm.theta_H_K_MarChemSpec25
theta['Ca-H'] = prm.theta_Ca_H_MarChemSpec
psi = WM13.psi
psi['Mg-MgOH-Cl'] = prm.psi_Mg_MgOH_Cl_HMW84
