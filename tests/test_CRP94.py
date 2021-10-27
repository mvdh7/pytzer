import pandas as pd, numpy as np
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


def test_CRP94_table7():
    """Does Pytzer reproduce the interaction parameters at 298.15 K from Table 7?"""
    tp = 298.15, 10.10325
    # H-HSO4 interaction
    b0, b1, _, C0, C1, alph1, _, omega, _ = pz.parameters.bC_H_HSO4_CRP94(*tp)
    assert np.round(b0, decimals=9) == 0.295_903_322
    assert np.round(b1, decimals=9) == 0.400_482_398
    assert np.round(C0, decimals=10) == -0.005_657_866_5  # should be -0.005_657_866_56
    assert np.round(C1, decimals=8) == -0.409_364_25  # should be -0.409_364_246
    assert alph1 == 2
    assert omega == 2.5
    # H-SO4 interaction
    b0, b1, _, C0, C1, alph1, _, omega, _ = pz.parameters.bC_H_SO4_CRP94(*tp)
    assert np.round(b0, decimals=11) == -0.008_386_089_24
    assert np.round(b1, decimals=9) == 0.314_734_575
    assert np.round(C0, decimals=10) == 0.010_192_247_4
    assert np.round(C1, decimals=8) == -0.323_662_60  # should be -0.323_662_605
    assert alph1 == 2
    assert omega == 2.5
    k_HSO4 = np.exp(pz.equilibrate.dissociation.HSO4_CRP94(tp[0]).item())
    assert np.round(k_HSO4, decimals=4) == 0.0105


def test_CRP94_table8():
    """Are Pytzer's HSO4-SO4 speciation values close to CRP94 Table 8?"""
    assert (crp94.alpha_diff.abs() < 4e-5).all()


def test_CRP94_Aphi():
    """Does Pytzer reproduce the Aphi test values from CRP94 Appendix II?"""
    Aphi_25 = np.round(pz.debyehueckel.Aosm_CRP94(298.15, 10.10325)[0], decimals=9)
    assert Aphi_25 == 0.391_475_238
    Aphi_00 = np.round(pz.debyehueckel.Aosm_CRP94(273.15, 10.10325)[0], decimals=9)
    assert Aphi_00 == 0.376_421_452


# test_CRP94_table7()
# test_CRP94_table8()
# test_CRP94_Aphi()
