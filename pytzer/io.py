# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
"""Import solution composition data, and export the results."""

from collections import OrderedDict
import numpy as np
from . import convert, dissociation, equilibrate as eq
from .libraries import Seawater


def solve_df(df, inplace=True, library=Seawater, verbose=False):
    """Solve all rows in a DataFrame for thermodynamic equilibrium."""
    if not inplace:
        df = df.copy()
    total_cols = {c: 0.0 for c in df.columns if c in convert.solute_to_charge}
    # Add empty rows to df to save results
    _ks_constants = dissociation.assemble(totals=total_cols)
    solutes = eq.components.find_solutes(total_cols, _ks_constants)
    for solute in solutes:
        if solute not in df:
            df[solute] = np.nan
    for ks_constant in _ks_constants:
        pks_constant = "pks_{}".format(ks_constant)
        if pks_constant not in df:
            df[pks_constant] = np.nan
    for i, row in df.iterrows():
        totals = OrderedDict(row[total_cols])
        if "temperature" in row:
            temperature = row.temperature
        else:
            temperature = 298.15
        if "pressure" in row:
            pressure = row.pressure
        else:
            pressure = 10.10325
        solutes, pks_constants = eq.solve(
            totals,
            library=library,
            temperature=temperature,
            pressure=pressure,
            verbose=verbose,
        )
        # Put results in df
        for solute, m_solute in solutes.items():
            df.loc[i, solute] = m_solute.item()
        for pks_constant, pks_value in pks_constants.items():
            df.loc[i, "pks_{}".format(pks_constant)] = pks_value
    if not inplace:
        return df
