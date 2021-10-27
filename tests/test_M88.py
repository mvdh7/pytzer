import pandas as pd, numpy as np
import pytzer as pz
from pytzer.libraries import Moller88

# Import data and prepare for tests
pz = Moller88.set_func_J(pz)
data = pd.read_csv("tests/data/M88 Table 4.csv").set_index("point")
m_cols = ["Na", "Ca", "Cl", "SO4"]
params = Moller88.get_parameters(solutes=m_cols, temperature=383.15)


def get_activity_water(data_row):
    dr = pz.odict(data_row[m_cols])
    return pz.activity_water(dr, **params).item()


data["a_H2O_pytzer"] = data.apply(get_activity_water, axis=1)


def test_M88_activity_water():
    """Can we reproduce the values from M88 Table 4?
    Note that we have corrected a presumed typo in row 4 (6.590 => 6.509 for Cl,
    based on an analysis of charge balance).
    """
    for i, row in data.iterrows():
        assert np.isclose(row["a_H2O"], row["a_H2O_pytzer"], rtol=0, atol=0.0001)


# test_M88_activity_water()
