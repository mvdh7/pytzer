from collections import OrderedDict
import pandas as pd, numpy as np
import pytzer as pz

# Select parameter library
prmlib = pz.libraries.Humphreys22
pz = prmlib.set_func_J(pz)

# Import data and run comparison, first without the solver (just using concentrations
# from in the table)
data = pd.read_excel("tests/data/Humphreys22-SI21.xlsx")
data["lnkHSO4_calc"] = prmlib["equilibria"]["HSO4"](
    data.temperature.to_numpy() + 273.15
)
data["lnkHSO4_td"] = np.nan
data["lnkHSO4_diff"] = np.nan
data["lnkH2O_calc"] = prmlib["equilibria"]["H2O"](data.temperature.to_numpy() + 273.15)
data["lnkH2O_td"] = np.nan
data["lnkH2O_diff"] = np.nan
data["lnkMgOH_calc"] = prmlib["equilibria"]["MgOH"](
    data.temperature.to_numpy() + 273.15
)
data["lnkMgOH_td"] = np.nan
data["lnkMgOH_diff"] = np.nan
data_ns_pz = data.copy()
data_ns_test = data.copy()
data_eq_pz = data.copy()
data_eq_test = data.copy()
data_eq_pct = data.copy()
for i, row in data.iterrows():
    solutes = OrderedDict((s[1:], v) for s, v in row.items() if s.startswith("m"))
    params = prmlib.get_parameters(
        solutes=solutes, temperature=273.15 + row.temperature, verbose=False
    )
    aH2O = pz.activity_water(solutes, **params)
    data_ns_pz.loc[i, "aH2O"] = aH2O
    data_ns_test.loc[i, "aH2O"] = np.round(aH2O - row.aH2O, decimals=5)
    phi = pz.osmotic_coefficient(solutes, **params)
    data_ns_pz.loc[i, "phi"] = phi
    data_ns_test.loc[i, "phi"] = np.round(phi - row.phi, decimals=5)
    acfs = pz.activity_coefficients(solutes, **params)
    for s, v in acfs.items():
        data_ns_pz.loc[i, "g" + s] = v
        data_ns_test.loc[i, "g" + s] = np.round(v - row["g" + s], decimals=5)

# Now do it again but with the equilibrium solver too
# Number 1
totals = OrderedDict(
    (
        ("Na", 0.4861818),
        ("Mg", 0.05474020),
        ("Ca", 0.01075004),
        ("K", 0.01058004),
        ("Cl", 0.5692021),
        ("SO4", 0.02927011),
    )
)
solutes_eq, pks_constants = pz.solve(
    totals,
    library=prmlib,
    temperature=278.15,
)
for s, v in solutes_eq.items():
    data_eq_pz.loc[0, "m" + s] = v
    if s in ["OH", "HSO4", "MgOH", "H"]:
        d = 12
    else:
        d = 5
    data_eq_test.loc[0, "m" + s] = np.round(v - data.loc[0, "m" + s], decimals=d)
    data_eq_pct.loc[0, "m" + s] = 100 * v / data.loc[0, "m" + s]
# Number 2
solutes_eq, pks_constants = pz.solve(
    totals,
    library=prmlib,
    temperature=298.15,
)
for s, v in solutes_eq.items():
    data_eq_pz.loc[1, "m" + s] = v
    if s in ["OH", "HSO4", "MgOH", "H"]:
        d = 12
    else:
        d = 5
    data_eq_test.loc[1, "m" + s] = np.round(v - data.loc[1, "m" + s], decimals=d)
    data_eq_pct.loc[1, "m" + s] = 100 * v / data.loc[1, "m" + s]
# Number 3
totals = OrderedDict(
    (
        ("Na", 0.4861818 - 0.04),
        ("Mg", 0.05474020),
        ("Ca", 0.01075004),
        ("K", 0.01058004),
        ("Cl", 0.5692021),
        ("SO4", 0.02927011),
    )
)
solutes_eq, pks_constants = pz.solve(
    totals,
    library=prmlib,
    temperature=278.15,
)
for s, v in solutes_eq.items():
    data_eq_pz.loc[2, "m" + s] = v
    if s in ["HSO4", "H"]:
        d = 8
    elif s in ["OH", "MgOH"]:
        d = 18
    else:
        d = 5
    data_eq_test.loc[2, "m" + s] = np.round(v - data.loc[2, "m" + s], decimals=d)
    data_eq_pct.loc[2, "m" + s] = 100 * v / data.loc[2, "m" + s]
# Number 4
solutes_eq, pks_constants = pz.solve(
    totals,
    library=prmlib,
    temperature=298.15,
)
for s, v in solutes_eq.items():
    data_eq_pz.loc[3, "m" + s] = v
    if s in ["HSO4", "H"]:
        d = 8
    elif s in ["OH", "MgOH"]:
        d = 18
    else:
        d = 5
    data_eq_test.loc[3, "m" + s] = np.round(v - data.loc[3, "m" + s], decimals=d)
    data_eq_pct.loc[3, "m" + s] = 100 * v / data.loc[3, "m" + s]

# Calculate activities and coefficients for equilibrated Pytzer data
for i, row in data_eq_pz.iterrows():
    solutes = OrderedDict((s[1:], v) for s, v in row.items() if s.startswith("m"))
    params = prmlib.get_parameters(
        solutes=solutes, temperature=273.15 + row.temperature, verbose=False
    )
    aH2O = pz.activity_water(solutes, **params)
    data_eq_test.loc[i, "aH2O"] = np.round(aH2O - row.aH2O, decimals=5)
    data_eq_pz.loc[i, "aH2O"] = aH2O
    phi = pz.osmotic_coefficient(solutes, **params)
    data_eq_test.loc[i, "phi"] = np.round(phi - row.phi, decimals=5)
    data_eq_pz.loc[i, "phi"] = phi
    acfs = pz.activity_coefficients(solutes, **params)
    for s, v in acfs.items():
        data_eq_test.loc[i, "g" + s] = np.round(v - row["g" + s], decimals=5)
        data_eq_pz.loc[i, "g" + s] = v

# Calculate thermodynamic Ks from HWT22 data with top-down approach (_td)
for df in [data, data_eq_pz]:
    df["lnkHSO4_td"] = np.log(df.mSO4 * df.gSO4 * df.mH * df.gH / (df.mHSO4 * df.gHSO4))
    df["lnkHSO4_diff"] = np.round(df.lnkHSO4_td - df.lnkHSO4_calc, decimals=8)
    df["lnkH2O_td"] = np.log(df.mH * df.gH * df.mOH * df.gOH / df.aH2O)
    df["lnkH2O_diff"] = np.round(df.lnkH2O_td - df.lnkH2O_calc, decimals=8)
    df["lnkMgOH_td"] = np.log(df.mMg * df.gMg * df.mOH * df.gOH / (df.mMgOH * df.gMgOH))
    df["lnkMgOH_diff"] = np.round(df.lnkMgOH_td - df.lnkMgOH_calc, decimals=8)

# Eliminate differences that have been identified and understood
# --------------------------------------------------------------
#
# The final digit of the gMgOH value at point 3 in HWT22 Table S21 is one away from
# the Pytzer value - so we assume this is a rounding error.
l = data.number == 3
data.loc[l, "gMgOH"] -= 1e-5
data_ns_test.loc[l, "gMgOH"] = np.round(
    data_ns_pz.loc[l, "gMgOH"] - data.loc[l, "gMgOH"], decimals=5
)

# Identify columns to test
test_cols_ns = [
    c
    for c in data.columns
    if c not in ["number", "temperature"]
    and not c.startswith("m")
    and not c.startswith("lnk")
]
test_cols_eq = [c for c in data.columns if c.startswith("m")]


def test_data_ns():
    """If we use the HWT22 concentrations and don't solve for speciation, do the Pytzer
    calculations of activity coefficients, osmotic coefficient and water activity agree
    with HWT22?
    """
    assert (data_ns_test[test_cols_ns] == 0).all().all()


def test_data_eq():
    """If we solve for equilibrium, are the Pytzer-calculated concentrations within 1.5%
    of the HWT22 values?
    """
    assert (
        ((data_eq_pct[test_cols_eq] > 98.5) & (data_eq_pct[test_cols_eq] < 101.5))
        .all()
        .all()
    )


def test_thermodynamic_Ks():
    """After solving for equilibrium with Pytzer, do the concentrations and activity
    coefficients agree with the target thermodynamic K values?
    """
    assert (data_eq_pz.lnkHSO4_diff == 0).all()
    assert (data_eq_pz.lnkH2O_diff == 0).all()
    assert (data_eq_pz.lnkMgOH_diff == 0).all()


# test_data_ns()
# test_data_eq()
# test_thermodynamic_Ks()
