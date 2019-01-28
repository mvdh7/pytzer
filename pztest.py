import pytzer as pz
import numpy as np 
#from autograd import elementwise_grad as egrad

filename = 'testfiles/GenerateConcs.csv'

mols,ions,T = pz.io.getmols(filename)

cf = pz.cfdicts.MIAMI

cf.print_coeffs(298.15,'print_coeffs/MIAMI.txt')

#cf.mu['tris-tris-tris'] = lambda T: [np.full_like(T,0.01),None]

# Cut out zero ionic strengths and do calculations

zs = pz.props.charges(ions)[0]
I = np.vstack(0.5 * (np.sum(mols * zs**2, 1)))

Gex_nRT = np.full_like(T   , np.nan)
osm     = np.full_like(T   , np.nan)
acfs    = np.full_like(mols, np.nan)

L = (I > 0).ravel()

nargsL  = (mols[ L,:], ions, T[ L], cf)
nargsLx = (mols[~L,:], ions, T[~L], cf)


#print('Calculating excess Gibbs energies...')
#Gex_nRT[ L] = pz.model.Gex_nRT(*nargsL)
#Gex_nRT[~L] = pz.model.Gex_nRT(*nargsLx, Izero=True)
#
#print('Calculating osmotic coefficients...')
#osm[ L] = pz.model.osm(*nargsL)
#osm[~L] = pz.model.osm(*nargsLx, Izero=True)
#
#print('Calculating activity coefficients...')
#acfs[ L,:] = pz.model.acfs(*nargsL)
#acfs[~L,:] = pz.model.acfs(*nargsLx, Izero=True)

## Test M88 NaCl - it's not me
#
##float_([-1.00588714e-1,
##        -1.80529413e-5,
##         8.61185543e00,
##         1.24880954e-2,
##         0            ,
##         3.41172108e-8,
##         6.83040995e-2,
##         2.93922611e-1]))
#
#dum = - 1.005188714e-1            \
#      - 1.80529413e-5 * T         \
#      + 8.61185543    / T         \
#      + 1.2488095e-2  * np.log(T) \
#      + 3.41172108e-8 * T**2      \
#      + 6.83040995e-2 / (680 - T) \
#      + 2.93922611e-1 / (T - 227)
#C0_Simon = 1e4 * dum[0] / 2e0
#
#dum2 = - 1.00588714e-1            \
#       - 1.80529413e-5 * T         \
#       + 8.61185543    / T         \
#       + 1.24880954e-2  * np.log(T) \
#       + 3.41172108e-8 * T**2      \
#       + 6.83040995e-2 / (680 - T) \
#       + 2.93922611e-1 / (T - 227)
#C0_Simon2 = 1e4 * dum2[0] / 2e0
#
#C0_M88 = 1e4 * pz.coeffs.bC_Na_Cl_M88(T)[3][0]
#
#print(C0_M88 == C0_Simon2)
