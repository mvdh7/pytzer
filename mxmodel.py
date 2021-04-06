import pytzer as pz
from autograd import numpy as np
from autograd.numpy import array, transpose

# Define inputs
mols_mx = np.array([[2.0, 3.5, 1.0, 0.5, 1.0, 0.5, 1.0, 2.0]])
ions = np.array(["Na", "Cl", "K", "Mg", "SO4", "HSO4", "Ca", "tris"])
# mols_mx = np.array([[1.0,]])
# ions = np.array(['tris',])
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
# etheta = pz.matrix.etheta(Aosm, Istr_mx, np.array([zs_mx[zs_mx > 0]]))

# =======================================================================
# The problem is that you can't autograd unsymmetrical mixing Jfunc(0.0)!
# =======================================================================

# Test autograd list comprehensioning - works fine
def Gtest(mols, allmxs):
    """Excess Gibbs energy of a solution."""
    (
        zs,
        Aosm,
        b0mx,
        b1mx,
        b2mx,
        C0mx,
        C1mx,
        alph1mx,
        alph2mx,
        omegamx,
        thetamx,
        lambdamx,
        psimxcca,
        psimxcaa,
        zetamx,
        mumx,
    ) = allmxs
    I = pz.matrix.Istr(mols, zs)
    cats = np.array([mols[zs > 0]])
    anis = np.array([mols[zs < 0]])
    zcats = array([zs[zs > 0]])
    zanis = array([zs[zs < 0]])
    zs_cats = array([[z if z > 0 else 0.0 for z in zs[0]]])
    zs_anis = array([[z if z < 0 else 0.0 for z in zs[0]]])
    ethetamx = pz.matrix.etheta(Aosm, I, zs_cats) + pz.matrix.etheta(Aosm, I, zs_anis)
    Gex_nRT0 = (
        mols @ thetamx @ transpose(mols)
        + cats @ pz.matrix.etheta(Aosm, I, zcats) @ transpose(cats)
        + anis @ pz.matrix.etheta(Aosm, I, zanis) @ transpose(anis)
    )[0]
    Gex_nRT1 = (mols @ (thetamx + ethetamx) @ transpose(mols))[0]
    return (
        Gex_nRT0,
        Gex_nRT1,
        ethetamx,
        thetamx,
        I,
        cats @ pz.matrix.etheta(Aosm, I, zcats) @ transpose(cats),
        mols @ pz.matrix.etheta(Aosm, I, zs_cats) @ transpose(mols),
        pz.matrix.etheta(Aosm, I, zcats),
        pz.matrix.etheta(Aosm, I, zs_cats),
        (pz.matrix.jfunc(pz.matrix.xij(Aosm, I, zcats))),
        #            - (pz.matrix.jfunc(pz.matrix.xi(Aosm, I, zcats))
        #            + pz.matrix.jfunc(pz.matrix.xj(Aosm, I, zcats)))/2),
        (pz.matrix.jfunc(pz.matrix.xij(Aosm, I, zs_cats))),
    )


#            - (pz.matrix.jfunc(pz.matrix.xi(Aosm, I, zs_cats))
#            + pz.matrix.jfunc(pz.matrix.xj(Aosm, I, zs_cats)))/2))

(Gt0, Gt1, Gethetamx, Gthetamx, I, Gbitcat, Gbitmol, Gwtf1, Gwtf2, Giz, Gizc) = Gtest(
    mols_mx, allmxs
)
from autograd import elementwise_grad as egrad

Gacf0 = np.exp(
    egrad(lambda mols_mx, allmxs: Gtest(mols_mx, allmxs)[0])(mols_mx, allmxs)
)
Gacf1 = np.exp(
    egrad(lambda mols_mx, allmxs: Gtest(mols_mx, allmxs)[1])(mols_mx, allmxs)
)

Gbitgradcat = egrad(lambda mols_mx, allmxs: Gtest(mols_mx, allmxs)[5])(mols_mx, allmxs)
Gbitgradmol = egrad(lambda mols_mx, allmxs: Gtest(mols_mx, allmxs)[6])(mols_mx, allmxs)
Ggradwtf1 = egrad(lambda mols_mx, allmxs: Gtest(mols_mx, allmxs)[7])(mols_mx, allmxs)
Ggradwtf2 = egrad(lambda mols_mx, allmxs: Gtest(mols_mx, allmxs)[8])(mols_mx, allmxs)
Ggradiz = egrad(lambda mols_mx, allmxs: Gtest(mols_mx, allmxs)[9])(mols_mx, allmxs)
Ggradizc = egrad(lambda mols_mx, allmxs: Gtest(mols_mx, allmxs)[10])(mols_mx, allmxs)

Jtest = pz.matrix.jfunc(6 * Aosm * np.sqrt(Istr_mx) * np.abs(transpose(zs_mx) @ zs_mx))

#%% Speed test index vs listcomp - index is faster by about 10x!
def spindex(mols, zs):
    return mols[zs > 0]


def splistc(mols, zs):
    return array([mol for i, mol in enumerate(mols[0]) if zs[0][i] > 0])


print(spindex(mols_mx, zs_mx))
print(splistc(mols_mx, zs_mx))
