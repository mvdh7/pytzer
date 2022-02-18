from jax import numpy as np
from autograd import numpy as anp
import pytzer as pz
import pytzer4 as pz4

molalities = np.array([[1.0, 1.0, 1.0, 1.5, 1.5]])
charges = np.array([[+1, +1, -2, +3, -3]])

ionic_strength_model = pz.model.ionic_strength(molalities, charges)
ionic_strength = pz.matrix.ionic_strength(molalities, charges)
# print(ionic_strength, ionic_strength_model)

ionic_zfunc_model = pz.model.ionic_z(molalities, charges)
ionic_zfunc = pz.matrix.ionic_z(molalities, charges)
# print(ionic_zfunc, ionic_zfunc_model)

solutes = pz.odict()
solutes["Na"] = 1.0
solutes["Cl"] = 1.0
solutes["Mg"] = 1.0
solutes["SO4"] = 1.0

prmlib = pz.libraries.Seawater
mparams = prmlib.get_matrices(solutes)
params = prmlib.get_parameters(solutes)

print("GO")

n_cats_triu = np.triu_indices(2, k=1)
n_anis_triu = np.triu_indices(2, k=1)
gexnrt = pz.matrix.Gibbs_nRT(solutes, n_cats_triu, n_anis_triu, **mparams)
print(gexnrt)

gexnrt_model = pz.model.Gibbs_nRT(solutes, **params)
print(gexnrt_model)

gexnrt_v4 = pz4.model.Gex_nRT(
    anp.vstack([1.0, 1.0, 1.0, 1.0]), ["Na", "Cl", "Mg", "SO4"], 298.15, 10.10325
)[0]
print(gexnrt_v4)

#%%
m_cats = molalities
m_cats_cats = np.array(
    [(np.transpose(m_cats) @ m_cats)[np.triu_indices(len(m_cats[0]), k=1)]]
)

#%%
import jax

uns = pz.unsymmetrical.Harvie(3.2)
print(uns)

uns2 = jax.vmap(pz.unsymmetrical.Harvie, in_axes=0, out_axes=0)(np.array([3.2, 3.2]))
print(uns2)

uns3 = jax.vmap(jax.vmap(pz.unsymmetrical.Harvie))(np.array([[3.2, 3.2], [3.2, 3.2]]))
print(uns3)
