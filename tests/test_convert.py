import numpy as np
import pytzer as pz
import pint

unit = pint.UnitRegistry()

molalities = dict(Na=1.5, Cl=1.5)
pks_constants = dict(H2O=14)
molinities, pks_out = pz.convert.solvent_to_solution(molalities, pks_constants)

# Manual approach - molalities
molality_NaCl = 1.5 * unit.mol / unit.kg
rmm_NaCl = 58.44 * unit.g / unit.mol
mass_H2O = 1 * unit.kg
amount_NaCl = molality_NaCl * mass_H2O
mass_NaCl = amount_NaCl * rmm_NaCl
mass_total = mass_H2O + mass_NaCl
content_NaCl = amount_NaCl / mass_total
content_NaCl_value = content_NaCl.magnitude  # mol / kg-solution

# Manual approach - pks_constants
mol_kg = unit.mol / unit.kg
pks_H2O_molality = 14
ks_H2O_molality = 10 ** -pks_H2O_molality * mol_kg ** 2
ks_H2O_content = ks_H2O_molality * mass_H2O ** 2 / mass_total ** 2
pks_H2O_content = -np.log10(ks_H2O_content / mol_kg ** 2).magnitude


def test_molinity_conversion():
    """Is the molality-to-molinity conversion function consistent with manual?"""
    assert np.isclose(content_NaCl.magnitude, molinities["Na"])
    assert np.isclose(content_NaCl.magnitude, molinities["Cl"])


def test_pK_conversions():
    """Is the pK conversion function consistent with manual?"""
    assert np.isclose(pks_H2O_content, pks_out["H2O"])


# test_molinity_conversion()
# test_pK_conversions()
