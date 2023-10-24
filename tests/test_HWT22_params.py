import pandas as pd, numpy as np
import pytzer as pz

# Select parameter library
prmlib = pz.libraries.Humphreys22


def compare_ca(sheet_name, tempK):
    # Import coefficient values from paper
    ca = pd.read_excel("tests/data/Humphreys22-SI7.xlsx", sheet_name=sheet_name)
    for v in ["alpha1", "alpha2", "omega"]:
        ca[v] = ca[v].fillna(-9)
    ca = ca.fillna(0)
    # Get coefficient values from parameter library
    ca_pz = ca.copy()
    ca_test = ca.copy()
    for i, row in ca.iterrows():
        try:
            res = prmlib["ca"][row.cation][row.anion](T=tempK, P=1)[:-1]
        except KeyError:
            print(row.cation, row.anion, "not found in prmlib")
            res = 0, 0, 0, 0, 0, -9, -9, -9
        beta0, beta1, beta2, C0, C1, alpha1, alpha2, omega = res
        # Rounding
        beta0 = np.round(beta0, decimals=5)
        beta1 = np.round(beta1, decimals=5)
        beta2 = np.round(beta2, decimals=3)
        C0 = np.round(C0, decimals=7)
        C1 = np.round(C1, decimals=5)
        alpha1 = np.round(alpha1, decimals=5)
        # Fill parameter library values
        ca_pz.loc[i, "beta0"] = beta0
        ca_pz.loc[i, "beta1"] = beta1
        ca_pz.loc[i, "beta2"] = beta2
        ca_pz.loc[i, "C0"] = C0
        ca_pz.loc[i, "C1"] = C1
        ca_pz.loc[i, "alpha1"] = alpha1
        ca_pz.loc[i, "alpha2"] = alpha2
        ca_pz.loc[i, "omega"] = omega
        # Compute differences
        ca_test.loc[i, "beta0"] = np.round(beta0 - row.beta0, decimals=5)
        ca_test.loc[i, "beta1"] = np.round(beta1 - row.beta1, decimals=5)
        ca_test.loc[i, "beta2"] = np.round(beta2 - row.beta2, decimals=3)
        ca_test.loc[i, "C0"] = np.round(C0 - row.C0, decimals=7)
        ca_test.loc[i, "C1"] = np.round(C1 - row.C1, decimals=5)
        ca_test.loc[i, "alpha1"] = np.round(alpha1 - row.alpha1, decimals=5)
        ca_test.loc[i, "alpha2"] = alpha2 - row.alpha2
        ca_test.loc[i, "omega"] = omega - row.omega
    return ca, ca_pz, ca_test


def compare_cc(sheet_name, tempK):
    cc = pd.read_excel("tests/data/Humphreys22-SI7.xlsx", sheet_name=sheet_name)
    cc = cc.fillna(0)
    cc_pz = cc.copy()
    cc_test = cc.copy()
    for i, row in cc.iterrows():
        try:
            theta = prmlib["cc"][row.cation1][row.cation2](T=tempK, P=1)[0]
        except KeyError:
            print(row.cation1, row.cation2, "not found in prmlib")
            theta = 0
        psi_Cl = prmlib["cca"][row.cation1][row.cation2]["Cl"](T=tempK, P=1)[0]
        try:
            psi_HSO4 = prmlib["cca"][row.cation1][row.cation2]["HSO4"](T=tempK, P=1)[0]
        except KeyError:
            print(row.cation1, row.cation2, "HSO4", "not found in prmlib")
            psi_HSO4 = 0
        try:
            psi_SO4 = prmlib["cca"][row.cation1][row.cation2]["SO4"](T=tempK, P=1)[0]
        except KeyError:
            print(row.cation1, row.cation2, "SO4", "not found in prmlib")
            psi_SO4 = 0
        cc_pz.loc[i, "theta"] = np.round(theta, decimals=4)
        cc_pz.loc[i, "Cl"] = np.round(psi_Cl, decimals=4)
        cc_pz.loc[i, "HSO4"] = np.round(psi_HSO4, decimals=5)
        cc_pz.loc[i, "SO4"] = np.round(psi_SO4, decimals=5)
        cc_test.loc[i, "theta"] = np.round(theta - row.theta, decimals=4)
        cc_test.loc[i, "Cl"] = np.round(psi_Cl - row.Cl, decimals=4)
        cc_test.loc[i, "HSO4"] = np.round(psi_HSO4 - row.HSO4, decimals=5)
        cc_test.loc[i, "SO4"] = np.round(psi_SO4 - row.SO4, decimals=5)
    return cc, cc_pz, cc_test


def compare_aa():
    aa = pd.read_excel("tests/data/Humphreys22-SI7.xlsx", sheet_name="aa")
    aa = aa.fillna(0)
    aa_pz = aa.copy()
    aa_test = aa.copy()
    for i, row in aa.iterrows():
        try:
            theta = prmlib["aa"][row.anion1][row.anion2](T=298.15, P=1)[0]
        except KeyError:
            print(row.anion1, row.anion2, "not found in prmlib")
            theta = 0
        psi = {}
        for c in ["Ca", "H", "K", "Mg", "Na"]:
            try:
                psi[c] = prmlib["caa"][c][row.anion1][row.anion2](T=298.15, P=1)[0]
            except KeyError:
                print(c, row.anion1, row.anion2, "not found in prmlib")
                psi[c] = 0
        aa_pz.loc[i, "theta"] = np.round(theta, decimals=3)
        aa_test.loc[i, "theta"] = np.round(theta - row.theta, decimals=3)
        for c, ps in psi.items():
            aa_pz.loc[i, c] = np.round(ps, decimals=6)
            aa_test.loc[i, c] = np.round(ps - row[c], decimals=6)

    return aa, aa_pz, aa_test


# Run comparisons
ca5, ca5_pz, ca5_test = compare_ca("ca5", 278.15)
ca25, ca25_pz, ca25_test = compare_ca("ca25", 298.15)
cc5, cc5_pz, cc5_test = compare_cc("cc5", 278.15)
cc25, cc25_pz, cc25_test = compare_cc("cc25", 298.15)
aa, aa_pz, aa_test = compare_aa()
# Define columns to check are zero
ca_cols = ["beta0", "beta1", "beta2", "C0", "C1", "alpha1", "alpha2", "omega"]
cc_cols = ["theta", "Cl", "HSO4", "SO4"]
aa_cols = ["theta", "Ca", "H", "K", "Mg", "Na"]

# Eliminate differences that have been identified and understood
# --------------------------------------------------------------
#
# It looks like the C0 value for Na-Cl at 25 °C in HWT22 Table S17 is actually a Cphi
# value - if you divide it by 2, then it agrees with Pytzer.  Testing of the data in
# HWT22 Table S21 suggests that this is a fault only in Table S17 and thus that the
# HWT22 model code does correctly convert it to C0.
l = (ca25.cation == "Na") & (ca25.anion == "Cl")
ca25.loc[l, "C0"] /= 2
ca25_test.loc[l, "C0"] = np.round(ca25_pz.loc[l, "C0"] - ca25.loc[l, "C0"], decimals=6)
#
# The final digit of the C0 value for H-SO4 at 25 °C in HWT22 Table S17 is one away from
# the Pytzer value - we assume this is a rounding error.
l = (ca25.cation == "H") & (ca25.anion == "SO4")
ca25.loc[l, "C0"] -= 1e-7
ca25_test.loc[l, "C0"] = ca25_pz.loc[l, "C0"] - ca25.loc[l, "C0"]
#
# The psi value for Ca-Mg-SO4 is in the wrong place in HWT22 Table S19 - it is in the
# spot for the Ca-H-SO4 value, which should be zero.
l0 = (cc25.cation1 == "Ca") & (cc25.cation2 == "H")
l1 = (cc25.cation1 == "Ca") & (cc25.cation2 == "Mg")
cc25.loc[l1, "SO4"] = cc25.loc[l0, "SO4"].values
cc25.loc[l0, "SO4"] = 0
cc25_test.loc[l0, "SO4"] = cc25_pz.loc[l0, "SO4"] - cc25.loc[l0, "SO4"]
cc25_test.loc[l1, "SO4"] = cc25_pz.loc[l1, "SO4"] - cc25.loc[l1, "SO4"]


def test_ca5():
    """Does Pytzer reproduce the same betas and Cs as HWT22 at 5 °C?"""
    assert (ca5_test[ca_cols] == 0).all().all()


def test_ca25():
    """Does Pytzer reproduce the same betas and Cs as HWT22 at 25 °C?"""
    assert (ca25_test[ca_cols] == 0).all().all()


def test_cc5():
    """Does Pytzer reproduce the same cc thetas and cca psis as HWT22 at 5 °C?"""
    assert (cc5_test[cc_cols] == 0).all().all()


def test_cc25():
    """Does Pytzer reproduce the same cc thetas and cca psis as HWT22 at 25 °C?"""
    assert (cc25_test[cc_cols] == 0).all().all()


def test_aa():
    """Does Pytzer reproduce the same aa thetas and caa psis as HWT22?  These are all
    independent of temperature.
    """
    assert (aa_test[aa_cols] == 0).all().all()


# test_ca5()
# test_ca25()
# test_cc5()
# test_cc25()
# test_aa()
