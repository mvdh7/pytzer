from copy import deepcopy
import pytzer as pz

filename = 'testfiles/GenerateConcs.csv'

mols,ions,T = pz.io.getmols(filename)

cf = deepcopy(pz.cfdicts.WM13_MarChemSpec25)
cf.add_zeros(ions)

TEMP = 298.15

cf.print_coeffs(TEMP,'print_coeffs/WM13_MarChemSpec25.txt')

bC_Mg_Cl_name     = cf.bC['Mg-Cl'    ].__name__ # dLP83 ok
bC_MgOH_Cl_name   = cf.bC['MgOH-Cl'  ].__name__ # HMW84 ok
bC_Mg_HSO4_name   = cf.bC['Mg-HSO4'  ].__name__ # RC99
bC_MgOH_HSO4_name = cf.bC['MgOH-HSO4'].__name__
bC_Mg_SO4_name    = cf.bC['Mg-SO4'   ].__name__
bC_MgOH_SO4_name  = cf.bC['MgOH-SO4' ].__name__

theta_HSO4_SO4_name = cf.theta['HSO4-SO4'].__name__
theta_Mg_MgOH_name  = cf.theta['Mg-MgOH' ].__name__

psi_Mg_MgOH_Cl_name  = cf.psi['Mg-MgOH-Cl' ].__name__
psi_Mg_HSO4_SO4_name = cf.psi['Mg-HSO4-SO4'].__name__

bC_Mg_Cl = cf.bC['Mg-Cl'](TEMP)[0]
