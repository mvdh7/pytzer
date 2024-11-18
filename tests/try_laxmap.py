import jax
from jax import numpy as np
import pytzer as pz
from pytzer import convert
from pytzer.model import Gibbs_DH, B, CT, etheta


solutes = {
    "Na": 1.5,
    "Ca": 1.0,
    "H": 1.0,
    "Cl": 3.0,
    "SO4": 0.75,
    "CO2": 0.5,
}
temperature = 298.15
pressure = 10.1
prmlib = pz.libraries.Clegg23
prmlib.update_ca("Na", "Cl", pz.parameters.bC_Na_Cl_A92ii)  # for testing
prmlib.update_nn("CO2", "CO2", pz.parameters.theta_BOH4_Cl_CWTD23)  # for testing
prmlib.update_nnn("CO2", pz.parameters.mu_tris_tris_tris_LTA21)  # for testing
pz = prmlib.set_func_J(pz)


@jax.jit
def _Gex_nRT(solutes, temperature, pressure):
    # === Separate the names, molalities and charges of cations, anions and neutrals ===
    n_solutes = list(solutes.keys())
    n_cations = [s for s in n_solutes if s in convert.all_cations]
    n_anions = [s for s in n_solutes if s in convert.all_anions]
    n_neutrals = [s for s in n_solutes if s in convert.all_neutrals]
    m_cations = []
    z_cations = []
    for s in n_cations:
        m_cations.append(solutes[s])
        z_cations.append(convert.solute_to_charge[s])
    m_anions = []
    z_anions = []
    for s in n_anions:
        m_anions.append(solutes[s])
        z_anions.append(convert.solute_to_charge[s])
    m_neutrals = []
    for s in n_neutrals:
        m_neutrals.append(solutes[s])
    m_cations = np.array(m_cations)
    z_cations = np.array(z_cations)
    m_anions = np.array(m_anions)
    z_anions = np.array(z_anions)
    # === Start with terms that depend only on ionic strength ==========================
    if len(n_cations) > 0 or len(n_anions) > 0:
        I = 0.5 * (np.sum(m_cations * z_cations**2) + np.sum(m_anions * z_anions**2))
    else:
        I = 0.0
    sqrt_I = np.sqrt(I)
    tp = (temperature, pressure)
    Aphi = prmlib["Aphi"](*tp)[0]
    Gibbs = Gibbs_DH(Aphi, I)
    # === Add (neutral-)cation-anion interactions ======================================
    if len(n_cations) > 0 and len(n_anions) > 0:
        Z = np.sum(m_cations * z_cations) - np.sum(m_anions * z_anions)
        for c, cation in enumerate(n_cations):
            for a, anion in enumerate(n_anions):
                ca = prmlib["ca"][cation][anion](*tp)
                Gibbs = Gibbs + m_cations[c] * m_anions[a] * (
                    2 * B(sqrt_I, *ca[:3], *ca[5:7]) + Z * CT(sqrt_I, *ca[3:5], ca[7])
                )
                if len(n_neutrals) > 0:
                    for n, neutral in enumerate(n_neutrals):
                        try:
                            Gibbs = (
                                Gibbs
                                + m_neutrals[n]
                                * m_cations[c]
                                * m_anions[a]
                                * prmlib["nca"][neutral][cation][anion](*tp)[0]
                            )
                        except KeyError:
                            pass
    # === Add cation-cation(-anion) interactions =======================================
    if len(n_cations) > 1:
        for c0, cation0 in enumerate(n_cations):
            for _c1, cation1 in enumerate(n_cations[(c0 + 1) :]):
                c1 = c0 + _c1 + 1
                try:
                    Gibbs = Gibbs + 2 * m_cations[c0] * m_cations[c1] * (
                        prmlib["cc"][cation0][cation1](*tp)[0]
                        + etheta(
                            Aphi,
                            I,
                            z_cations[c0],
                            z_cations[c1],
                            func_J=prmlib["func_J"],
                        )
                    )
                except KeyError:
                    pass
                for a, anion in enumerate(n_anions):
                    try:
                        Gibbs = (
                            Gibbs
                            + m_cations[c0]
                            * m_cations[c1]
                            * m_anions[a]
                            * prmlib["cca"][cation0][cation1][anion](*tp)[0]
                        )
                    except KeyError:
                        pass
    # === Add (cation-)anion-anion interactions ========================================
    if len(n_anions) > 1:
        for a0, anion0 in enumerate(n_anions):
            for _a1, anion1 in enumerate(n_anions[(a0 + 1) :]):
                a1 = a0 + _a1 + 1
                try:
                    Gibbs = Gibbs + 2 * m_anions[a0] * m_anions[a1] * (
                        prmlib["aa"][anion0][anion1](*tp)[0]
                        + etheta(
                            Aphi, I, z_anions[a0], z_anions[a1], func_J=prmlib["func_J"]
                        )
                    )
                except KeyError:
                    pass
                for c, cation in enumerate(n_cations):
                    try:
                        Gibbs = (
                            Gibbs
                            + m_cations[c]
                            * m_anions[a0]
                            * m_anions[a1]
                            * prmlib["caa"][cation][anion0][anion1](*tp)[0]
                        )
                    except KeyError:
                        pass
    # === Add other neutral interactions ===============================================
    if len(n_neutrals) > 0:
        for n0, neutral0 in enumerate(n_neutrals):
            # Neutral-neutral (can be the same or different neutrals)
            for n1, neutral1 in enumerate(n_neutrals):
                try:
                    Gibbs = (
                        Gibbs
                        + m_neutrals[n0]
                        * m_neutrals[n1]
                        * prmlib["nn"][neutral0][neutral1](*tp)[0]
                    )
                except KeyError:
                    pass
            # Neutral-neutral-neutral (always the same neutral 3 times)
            try:
                Gibbs = Gibbs + m_neutrals[n0] ** 3 * prmlib["nnn"][neutral0](*tp)[0]
            except KeyError:
                pass
            # Neutral-cation
            if len(n_cations) > 0:
                for c, cation in enumerate(n_cations):
                    try:
                        Gibbs = (
                            Gibbs
                            + 2
                            * m_neutrals[n0]
                            * m_cations[c]
                            * prmlib["nc"][neutral0][cation](*tp)[0]
                        )
                    except KeyError:
                        pass
            # Neutral-anion
            if len(n_anions) > 0:
                for a, anion in enumerate(n_anions):
                    try:
                        Gibbs = (
                            Gibbs
                            + 2
                            * m_neutrals[n0]
                            * m_anions[a]
                            * prmlib["na"][neutral0][anion](*tp)[0]
                        )
                    except KeyError:
                        pass
    return Gibbs


# %%
def Gex_nRT(solutes, temperature, pressure):
    return _Gex_nRT(solutes, temperature, pressure)


Gargs = (solutes, temperature, pressure)
G = Gex_nRT(*Gargs)

fGsj = jax.jit(jax.grad(Gex_nRT))

Gs = jax.grad(Gex_nRT)(*Gargs)
Gsj = fGsj(*Gargs)
Gt = jax.grad(Gex_nRT, argnums=1)(*Gargs)
Gp = jax.grad(Gex_nRT, argnums=2)(*Gargs)

osolutes = pz.odict(solutes)
params = prmlib.get_parameters(
    solutes=osolutes, temperature=temperature, pressure=pressure
)
G_old = pz.model.Gibbs_nRT(osolutes, **params)
Gs_old = pz.model.log_activity_coefficients(osolutes, **params)

print(G)
print(G_old)

# %%
# solutes = {"Na": 2.0, "Mg": 1.0, "Cl": 4.0}
# charges = {"Na": 1, "Mg": 2, "Cl": -1}


# cations = {"Na", "Mg"}
# anions = {"Cl"}

# multipliers = {"Na": dict(Cl=1.5), "Mg": dict(Cl=1.2)}


# def iterm(cation_anion):
#     cation, anion = cation_anion
#     return solutes[cation] * solutes[anion] * multipliers[cation][anion]

# cation = "Na"
# anion = "Cl"
# it_Na = iterm((cation, anion))
# it_Mg = iterm(("Mg", "Cl"))

# pr = list(itertools.product(cations, anions))

# it_map = sum(map(iterm, pr))
# it_jaxmap = jax.lax.map(iterm, np.array(pr))


# %%
# def _ionic_strength(solute, charge):
#     return 0.5 * solute * charge ** 2


# def dict_sum(kv):
#     s = 0.0
#     for v in kv.values():
#         s += v
#     return s


# def ionic_strength(solutes, charges):
#     istrs = jax.tree.map(_ionic_strength, solutes, charges)
#     return dict_sum(istrs)


# def ionic_strength_2(solutes, charges):
#     s = 0.0
#     for k, v in solutes.items():
#         s += 0.5 * v * charges[k] ** 2
#     return s

# istr = ionic_strength(solutes, charges)
# istr2 = ionic_strength_2(solutes, charges)
