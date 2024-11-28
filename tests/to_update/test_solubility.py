import pytzer as pz
import numpy as np
from matplotlib import pyplot as plt

# Prepare parameter library
prmlib = pz.libraries.Clegg23
pz = prmlib.set_func_J(pz)

# %% Calculate activities across a range of conditions
temperature = 288.15
totals = pz.prepare.salinity_to_totals_MFWM08()
total_MgCa = totals["Mg"] + totals["Ca"]
total_CO2 = totals["CO2"]

fMgCa_sw = np.linspace(0.01, 0.99, num=21)
mCO2 = np.linspace(0.5, 1.5, num=21)

aMg = np.full((fMgCa_sw.size, mCO2.size), np.nan)
aCa = np.full((fMgCa_sw.size, mCO2.size), np.nan)
aCO3 = np.full((fMgCa_sw.size, mCO2.size), np.nan)

for i, f in enumerate(fMgCa_sw):
    for j, m in enumerate(mCO2):
        print(i, j)
        _totals = totals.copy()
        _totals["Mg"] = f * total_MgCa
        _totals["Ca"] = (1 - f) * total_MgCa
        _totals["CO2"] = m
        solutes, pks_constants = pz.solve(
            _totals, library=prmlib, temperature=temperature
        )
        params = prmlib.get_parameters(
            solutes=solutes, temperature=temperature, verbose=False
        )
        acfs = pz.activity_coefficients(solutes, **params)
        aMg[i, j] = (acfs["Mg"] * solutes["Mg"]).item()
        aCa[i, j] = (acfs["Ca"] * solutes["Ca"]).item()
        aCO3[i, j] = (acfs["CO3"] * solutes["CO3"]).item()

# %% Stoichiometric solubility product / ionic activity product calculated following
# Morse et al. (2006) eq. (2) --- but this is NOT in equilibrium!
# We would need to adjust Mg, Ca and CO3 to match iap.
xMgCa = 0.48
iap = aMg**xMgCa * aCa ** (1 - xMgCa) * aCO3
piap = -np.log10(iap)

# Visualise results
mm, ff = np.meshgrid(mCO2, fMgCa_sw)
fig, ax = plt.subplots(dpi=300)
cf = ax.contourf(ff, mm * total_CO2 * 1e3, piap, levels=128)
ax.scatter(totals["Mg"] / total_MgCa, total_CO2 * 1e3, marker="+", c="w")
ax.set_title("Mg/Ca in solid = {}".format(xMgCa))
ax.set_xlabel(r"$T_\mathrm{Mg}$ / ($T_\mathrm{Mg}$ + $T_\mathrm{Ca}$)")
ax.set_ylabel(r"$T_\mathrm{C}$ / mmol kg$^{-1}$")
plt.colorbar(cf, label="pIAP")
fig.tight_layout()
