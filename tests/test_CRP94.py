import pandas as pd
import pytzer as pz

# Update unsymmetrical mixing function
pzlib = pz.libraries.Clegg94
pz = pzlib.set_func_J(pz)

# Import and solve
crp94 = pd.read_csv("tests/data/CRP94 Table 8.csv")
crp94["t_SO4"] = crp94.SO4
pz.solve_df(crp94, library=pzlib)

# Compare
crp94["alpha_pytzer"] = (crp94.SO4 / (crp94.SO4 + crp94.HSO4)).round(5)
crp94["alpha_diff"] = crp94.alpha - crp94.alpha_pytzer
# Close but not exact --- not sure why.  Excluding OH makes things much worse.


def test_CRP94_table8():
    """Are Pytzer's values close to CRP94 Table 8?"""
    assert (crp94.alpha_diff.abs() < 4e-5).all()


# test_CRP94_table8()
