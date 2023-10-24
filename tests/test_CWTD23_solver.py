"""Testing against CWTD23 SI#9 check values."""
import pandas as pd
import pytzer as pz
import numpy as np

# Select parameter library
prmlib = pz.libraries.Clegg23
pz = prmlib.set_func_J(pz)

# Solve and compare without equilibrating
data = pd.read_csv("tests/data/CWTD23 SI final table.csv")
data["aH2O_pz"] = np.nan
data["osm_pz"] = np.nan
data_diff = data[["temperature"]].copy()
for c in data.columns:
    if c.startswith("m"):
        data["y" + c[1:] + "_pz"] = np.nan
        data_diff["y" + c[1:]] = np.nan
data_diff.drop(columns="temperature", inplace=True)
for i, row in data.iterrows():
    solutes = pz.odict((s[1:], v) for s, v in row.items() if s.startswith("m"))
    params = prmlib.get_parameters(
        solutes=solutes, temperature=273.15 + row.temperature, verbose=False
    )
    aH2O = pz.activity_water(solutes, **params)
    data.loc[i, "aH2O_pz"] = aH2O
    osm = pz.osmotic_coefficient(solutes, **params)
    data.loc[i, "osm_pz"] = osm
    acfs = pz.activity_coefficients(solutes, **params)
    for s, v in acfs.items():
        data.loc[i, "y" + s + "_pz"] = v
        data_diff.loc[i, "y" + s] = 100 * (v - row["y" + s]) / row["y" + s]
dcols = list(data.columns)
dcols.sort()
data = data[dcols]

# Now with the equilibrium solver
data_eq = data.copy()
for c in data_eq.columns:
    if c.startswith("m"):
        data_eq[c + "_eq"] = np.nan
for i, row in data_eq.iterrows():
    totals = pz.odict(
        (
            ("BOH3", row.mBOH3 + row.mBOH4),
            ("Br", row.mBr),
            ("Ca", row.mCa + row.mCaCO3 + row.mCaF),
            ("Cl", row.mCl),
            ("CO2", row.mCO2 + row.mCO3 + row.mHCO3 + row.mCaCO3 + row.mMgCO3),
            ("F", row.mCaF + row.mF + row.mHF + row.mMgF),
            ("SO4", row.mHSO4 + row.mSO4),
            ("K", row.mK),
            ("Mg", row.mMg + row.mMgCO3 + row.mMgF + row.mMgOH),
            ("Na", row.mNa),
            ("Sr", row.mSr + row.mSrCO3),
        )
    )
    solutes_eq, pks_constants = pz.solve(
        totals,
        library=prmlib,
        temperature=273.15 + row.temperature,
    )
    for c in data_eq.columns:
        if c.endswith("_eq"):
            data_eq.loc[i, c] = solutes_eq[c[1:-3]]
data_eq["pH"] = -np.log10(data_eq["mH"])
data_eq["pH_eq"] = -np.log10(data_eq["mH_eq"])


def compeq(solute):
    if solute == "pH":
        return data_eq[[solute, solute + "_eq"]]
    else:
        return data_eq[["m" + solute, "m" + solute + "_eq"]]


def test_without_equilibrium():
    # Activity coefficients are all within 0.001% (i.e., 1e-3%) of CWTD23 values
    assert (data_diff.abs() < 1e-3).all().all()
    # Water activity within 1e-6 (absolute) of CWTD23 values
    assert ((data.aH2O_pz - data.aH2O).abs() < 1e-6).all()
    # Osmotic coefficient within 1e-5 (absolute) of CWTD23 values
    assert ((data.osm_pz - data.osm).abs() < 1e-5).all()


def test_with_equilibrium():
    for s in solutes:
        # Molalities all within 0.1% of CWTD23 values
        assert (
            100 * compeq(s).diff(axis=1)["m" + s + "_eq"].abs() / data_eq["m" + s]
        ).max() < 0.1


# test_without_equilibrium()
# test_with_equilibrium()
