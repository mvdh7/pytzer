import pytzer as pz
from autograd import numpy as np

# Define inputs
mols_mx = np.array([[2.0, 3.5, 1.0, 0.5, 1.0, 0.5, 1.0, 2.0]])
ions = np.array(['Na', 'Cl', 'K', 'Mg', 'SO4', 'HSO4', 'Ca', 'tris'])
#mols_mx = np.array([[1.0,]])
#ions = np.array(['tris',])
tempK = np.array([298.15])
pres = np.array([10.1325])
cflib = pz.cflibs.Seawater

# Calculate intermediates
mols_pz = np.vstack(mols_mx[0])
zs_pz, cations, anions, _ = pz.properties.charges(ions)
zs_mx = np.transpose(zs_pz)

# Calculate ionic strength
Istr_pz = pz.model.Istr(mols_pz, zs_pz)
Izero = Istr_pz == 0
Istr_mx = pz.matrix.Istr(mols_mx, zs_mx)
Zstr_pz = pz.model.Zstr(mols_pz, zs_pz)
Zstr_mx = pz.matrix.Zstr(mols_mx, zs_mx)

# Assemble coefficient matrices
allmxs = pz.matrix.assemble(ions, tempK, pres, cflib)

# Calculate Debye-Hueckel term
fG_pz = pz.model.fG(tempK, pres, Istr_pz, cflib)
fG_mx = pz.matrix.fG(allmxs[1], Istr_mx)

# Calculate excess Gibbs energy
Gex_pz = pz.model.Gex_nRT(mols_pz, ions, tempK, pres, cflib, Izero=Izero)
Gex_mx = pz.matrix.Gex_nRT(mols_mx, allmxs)

# Try molality derivative
acfs_pz = pz.model.acfs(mols_pz, ions, tempK, pres, cflib, Izero=Izero)
acfs_mx = pz.matrix.acfs(mols_mx, allmxs)

# Water activity
aw_pz = pz.model.aw(mols_pz, ions, tempK, pres, cflib, Izero=Izero)
aw_mx = pz.matrix.aw(mols_mx, allmxs)

# Unsymmetrical mixing terms
Aosm = pz.debyehueckel.Aosm_MarChemSpec(tempK, pres)[0]
xij = pz.matrix.xij(Aosm, Istr_mx, np.array([zs_mx[zs_mx > 0]]))
xi = pz.matrix.xi(Aosm, Istr_mx, np.array([zs_mx[zs_mx > 0]]))
xj = pz.matrix.xj(Aosm, Istr_mx, np.array([zs_mx[zs_mx > 0]]))
Jxij = pz.matrix.jfunc(xij)
etheta = pz.matrix.etheta(Aosm, Istr_mx, np.array([zs_mx[zs_mx > 0]]))
