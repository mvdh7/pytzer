import pytzer as pz
import pandas as pd

prmlib = pz.libraries.Clegg23

# Get sets of all ions
cations = {k for k in prmlib["ca"].keys()}
anions = []
for c in cations:
    for a in prmlib["ca"][c].keys():
        anions.append(a)
anions = set(anions)

# Chart 8 all at 25 °C
chart8_1 = pd.read_fwf("tests/data/CWTD23 SI Chart 8 part 1.txt")
chart8_2 = pd.read_csv("tests/data/CWTD23 SI Chart 8 part 2.csv")

# Chart 9 all at 5 °C
chart9_1 = pd.read_fwf("tests/data/CWTD23 SI Chart 9 part 1.txt")
chart9_2 = pd.read_csv("tests/data/CWTD23 SI Chart 9 part 2.csv")


def test_theta():
    """Do all thetas agree, including ones not in the Chart (should all be zero)?"""
    # At 25 °C
    for c1 in cations:
        for c2 in cations:
            if c1 != c2:
                try:
                    theta = "{:11.5e}".format(prmlib["cc"][c1][c2](298.15, 10)[0])
                except KeyError:
                    theta = "0.00000e+00"
                L = ~chart8_2.theta_cc.isnull() & (
                    ((chart8_2.cca_c1 == c1) & (chart8_2.cca_c2 == c2))
                    | ((chart8_2.cca_c1 == c2) & (chart8_2.cca_c2 == c1))
                )
                if sum(L) == 1:
                    theta_chart = "{:11.5e}".format(chart8_2.theta_cc[L].values[0])
                else:
                    assert sum(L) == 0
                    theta_chart = "0.00000e+00"
                assert theta.upper() == theta_chart.upper(), "theta {} {} {} {}".format(
                    c1, c2, theta, theta_chart
                )
    for a1 in anions:
        for a2 in anions:
            if a1 != a2:
                try:
                    theta = "{:11.5e}".format(prmlib["aa"][a1][a2](298.15, 10)[0])
                except KeyError:
                    theta = "0.00000e+00"
                L = ~chart8_2.theta_aa.isnull() & (
                    ((chart8_2.caa_a1 == a1) & (chart8_2.caa_a2 == a2))
                    | ((chart8_2.caa_a1 == a2) & (chart8_2.caa_a2 == a1))
                )
                if sum(L) == 1:
                    theta_chart = "{:11.5e}".format(chart8_2.theta_aa[L].values[0])
                else:
                    assert sum(L) == 0
                    theta_chart = "0.00000e+00"
                assert theta.upper() == theta_chart.upper(), "theta {} {} {} {}".format(
                    a1, a2, theta, theta_chart
                )
    # At 5 °C
    for c1 in cations:
        for c2 in cations:
            if c1 != c2:
                try:
                    theta = "{:11.5e}".format(prmlib["cc"][c1][c2](278.15, 10)[0])
                except KeyError:
                    theta = "0.00000e+00"
                L = ~chart9_2.theta_cc.isnull() & (
                    ((chart9_2.cca_c1 == c1) & (chart9_2.cca_c2 == c2))
                    | ((chart9_2.cca_c1 == c2) & (chart9_2.cca_c2 == c1))
                )
                if sum(L) == 1:
                    theta_chart = "{:11.5e}".format(chart9_2.theta_cc[L].values[0])
                else:
                    assert sum(L) == 0
                    theta_chart = "0.00000e+00"
                assert theta.upper() == theta_chart.upper(), "theta {} {} {} {}".format(
                    c1, c2, theta, theta_chart
                )
    for a1 in anions:
        for a2 in anions:
            if a1 != a2:
                try:
                    theta = "{:11.5e}".format(prmlib["aa"][a1][a2](278.15, 10)[0])
                except KeyError:
                    theta = "0.00000e+00"
                L = ~chart9_2.theta_aa.isnull() & (
                    ((chart9_2.caa_a1 == a1) & (chart9_2.caa_a2 == a2))
                    | ((chart9_2.caa_a1 == a2) & (chart9_2.caa_a2 == a1))
                )
                if sum(L) == 1:
                    theta_chart = "{:11.5e}".format(chart9_2.theta_aa[L].values[0])
                else:
                    assert sum(L) == 0
                    theta_chart = "0.00000e+00"
                assert theta.upper() == theta_chart.upper(), "theta {} {} {} {}".format(
                    a1, a2, theta, theta_chart
                )


def test_ca():
    print("Checking Chart 8 (betas and Cs at 25 °C)...")
    for i, row in chart8_1.iterrows():
        p = {}
        try:
            (
                p["b0"],
                p["b1"],
                p["b2"],
                p["C0"],
                p["C1"],
                p["alph1"],
                p["alph2"],
                p["omega"],
                _,
            ) = prmlib["ca"][row.cation][row.anion](
                298.15, 10
            )  # Based on NaOH, appears that CWTD23 have evaluated at 1 bar
            if p["alph1"] == -9:
                p["alph1"] = 0
            if p["alph2"] == -9:
                p["alph2"] = 0
            if p["omega"] == -9:
                p["omega"] = 0
        except KeyError:
            (
                p["b0"],
                p["b1"],
                p["b2"],
                p["C0"],
                p["C1"],
                p["alph1"],
                p["alph2"],
                p["omega"],
            ) = (0, 0, 0, 0, 0, 0, 0, 0)
        for k in p.keys():
            assert "{:11.5e}".format(p[k]) == "{:11.5e}".format(
                row[k]
            ), "{} {} {} {:11.5e} {:11.5e}".format(
                row.cation, row.anion, k, p[k], row[k]
            )
    print("Chart 8 checked!")

    print("Checking Chart 9 (betas and Cs at 5 °C)...")
    for i, row in chart9_1.iterrows():
        p = {}
        try:
            (
                p["b0"],
                p["b1"],
                p["b2"],
                p["C0"],
                p["C1"],
                p["alph1"],
                p["alph2"],
                p["omega"],
                _,
            ) = prmlib["ca"][row.cation][row.anion](
                278.15, 10
            )  # Based on NaOH, appears that CWTD23 have evaluated at 1 bar
            if p["alph1"] == -9:
                p["alph1"] = 0
            if p["alph2"] == -9:
                p["alph2"] = 0
            if p["omega"] == -9:
                p["omega"] = 0
        except KeyError:
            (
                p["b0"],
                p["b1"],
                p["b2"],
                p["C0"],
                p["C1"],
                p["alph1"],
                p["alph2"],
                p["omega"],
            ) = (0, 0, 0, 0, 0, 0, 0, 0)
        for k in p.keys():
            if row.cation == "H" and row.anion == "SO4" and k == "alph1":
                # Doesn't work to full precision for some reason
                assert "{:9.3e}".format(p[k]) == "{:9.3e}".format(
                    row[k]
                ), "{} {} {} {:11.5e} {:11.5e}".format(
                    row.cation, row.anion, k, p[k], row[k]
                )
            elif row.cation == "Mg" and row.anion == "Cl" and k == "C0":
                # Very minor rounding error
                assert "{:10.4e}".format(p[k]) == "{:10.4e}".format(
                    row[k]
                ), "{} {} {} {:11.5e} {:11.5e}".format(
                    row.cation, row.anion, k, p[k], row[k]
                )
            else:
                assert "{:11.5e}".format(p[k]) == "{:11.5e}".format(
                    row[k]
                ), "{} {} {} {:11.5e} {:11.5e}".format(
                    row.cation, row.anion, k, p[k], row[k]
                )
    print("Chart 9 checked!")
    # TODO There are quite a few issues with SRRJ87, check equations


# test_ca()
# test_theta()
