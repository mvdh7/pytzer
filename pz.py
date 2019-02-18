import pytzer as pz
import numpy as np

mols = np.array([[1.,3.,1.]])

ions = np.array(['Na','Cl','Ca'])

T = np.array([[298.15]])

nreps = 1000
mols = np.tile(mols,(nreps,1))
T = np.tile(T,(nreps,1))

cf = pz.cfdicts.CoefficientDictionary()

cf.dh['Aosm'] = pz.coeffs.Aosm_MarChemSpec

cf.bC['Na-Cl'] = pz.coeffs.bC_Na_Cl_M88
cf.bC['Ca-Cl'] = pz.coeffs.bC_Ca_Cl_GM89

cf.theta['Ca-Na'] = pz.coeffs.theta_Ca_Na_HMW84

cf.psi['Ca-Na-Cl'] = pz.coeffs.psi_Ca_Na_Cl_HMW84

cf.add_zeros(ions)

cf.jfunc = pz.jfuncs.P75_eq47

zs = pz.props.charges(ions)[0]
I = pz.model.Istr(mols,zs)

# print(I)

# print(pz.model.fG(T,I,cf))

# print(cf.bC['Na-Cl'](T))
#
# print(pz.model.B(T,I,cf,'Na-Cl'))
#
# print(pz.model.g(1.5))
#
# print(pz.model.acfs(mols,ions,T,cf))
# %timeit pz.model.acfs(mols,ions,T,cf)
