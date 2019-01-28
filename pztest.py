import pytzer as pz
import numpy as np 
#from autograd import elementwise_grad as egrad

filename = 'testfiles/PP87i Table 4.csv'

mols,ions,T = pz.io.getmols(filename)

T[:] = 273.15 + 350

cf = pz.cfdicts.MarChemSpec

ln_acfs = pz.model.ln_acfs(mols,ions,T,cf)

ln_acfMX = np.vstack(pz.model.ln_acf2ln_acf_MX(ln_acfs[:,0],ln_acfs[:,1],1,1))

acfMX = np.exp(ln_acfMX)

# Print out coefficient values to file

cf.print_coeffs(298.15,'print_coeffs/pytzer-v0-2-1-3_WM13_25.txt')

#acfMX_target = 

#osm = pz.model.osm(mols,ions,T,cf)
#aw = pz.model.osm2aw(mols,osm)



# Test jfunc differentiation Harvie

#x = np.array([1.5])
#
#J, Jp = pz.jfuncs._Harvie_raw(x)
#
#Jg = egrad(lambda x: pz.jfuncs.Harvie(x)[0])(x)
#
#print(Jp,Jg)
#

#
##cf.add_zeros(np.array(['Ba','Ca']))
#
# Cut out zero ionic strengths and do calculations

#zs = pz.props.charges(ions)[0]
#I = np.vstack(0.5 * (np.sum(mols * zs**2, 1)))
#
#Gex_nRT = np.full_like(T   , np.nan)
#osm     = np.full_like(T   , np.nan)
#acfs    = np.full_like(mols, np.nan)
#
#L = (I > 0).ravel()
#
#nargs = (mols[L,:], ions, T[L], cf)
#
#Gex_nRT[L  ] = pz.model.Gex_nRT(*nargs)
#
#import numba
#
#Gex_numba = numba.autojit(pz.model.Gex_nRT)
#
#Gex2 = Gex_numba(*nargs)

#osm    [L  ] = pz.model.osm    (*nargs)
#acfs   [L,:] = pz.model.acfs   (*nargs)

#I = np.zeros((1,1))
#T = np.full((1,1),298.15)
#
##fG = pz.model.fG(T,I,cf)
##
##B = pz.model.B(T,I,cf,'Na-Cl')
#g = pz.model.g(I)
##gp = pz.model.g_vjp(I)
#
#gp = egrad(pz.model.g)(I)

#
#CT = pz.model.CT(T,I,cf,'Na-Cl')
#h = pz.model.h(I)

#etheta = pz.model.etheta(T,I,+1,+2,cf)
#xij = pz.model.xij(T,I,+1,+2,cf)

#J_Harvie   = pz.jfuncs.Harvie  (xij)
#J_P75_eq46 = pz.jfuncs.P75_eq46(xij)
#J_P75_eq47  = pz.jfuncs.P75_eq47(xij)
#Jp_P75_eq47  = pz.jfuncs.P75_eq47_dx(xij)
#fx_JP_P75_eq47 = egrad(pz.jfuncs.P75_eq47)
#JP_P75_eq47 = fx_JP_P75_eq47(xij)
