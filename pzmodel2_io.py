import pytzer as pz
import numpy as np

mols, ions, tempK = pz.io.getmols('testfiles/pytzerPitzer.csv')

cf = pz.cflibs.MarChemSpec

cf.print_coeffs(298.15,'print_coeffs/test.txt')

#cf.bC['K-Cl'] = pz.coeffs.bC_K_Cl_GM89

#print(cf.ions)
cf.get_contents()
#print(cf.ions)
