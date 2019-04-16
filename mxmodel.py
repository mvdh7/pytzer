import pytzer as pz
from autograd import numpy as np
from autograd import elementwise_grad as egrad

# Define inputs
mols_mx = np.array([[1.5, 1.5, 0.5, 0.5]])
ions = np.array(['Na', 'Cl', 'Mg', 'SO4'])
tempK = np.array([298.15])
pres = np.array([10.1325])
cflib = pz.cflibs.CoeffLib()
cflib.dh['Aosm'] = pz.debyehueckel.Aosm_AW90
cflib.bC['Na-Cl'] = pz.coeffs.bC_Na_Cl_A92ii
cflib.bC['Na-SO4'] = pz.coeffs.bC_Na_SO4_HPR93
cflib.bC['Mg-SO4'] = pz.coeffs.bC_Mg_SO4_PP86ii
cflib.bC['Mg-Cl'] = pz.coeffs.bC_Mg_Cl_PP87i
cflib.jfunc = pz.jfuncs.none
cflib.add_zeros(ions)

# Calculate intermediates
mols_pz = np.vstack(mols_mx[0])
zs_pz, cations, anions, _ = pz.props.charges(ions)
zs_mx = np.transpose(zs_pz)

# Calculate ionic strength
Istr_pz = pz.model.Istr(mols_pz, zs_pz)
Istr_mx = pz.matrix.Istr(mols_mx, zs_mx)
Zstr_pz = pz.model.Zstr(mols_pz, zs_pz)
Zstr_mx = pz.matrix.Zstr(mols_mx, zs_mx)

# Assemble coefficient matrices
allmxs = pz.matrix.assemble(ions, tempK, pres, cflib)

# Calculate Debye-Hueckel term
fG_pz = pz.model.fG(tempK, pres, Istr_pz, cflib)
fG_mx = pz.matrix.fG(allmxs[1], Istr_mx)

# Calculate excess Gibbs energy
Gex_pz = pz.model.Gex_nRT(mols_pz, ions, tempK, pres, cflib)
Gex_mx = pz.matrix.Gex_nRT(mols_mx, *allmxs)

# Try molality derivative
facfs_pz = egrad(pz.model.Gex_nRT)
facfs_mx = egrad(pz.matrix.Gex_nRT)

acfs_pz = facfs_pz(mols_pz, ions, tempK, pres, cflib)
acfs_mx = facfs_mx(mols_mx, *allmxs)
