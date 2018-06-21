from . import coeffs

# Set up dict of coefficient functions
cf_M88 = {coeff:{} for coeff in ['bC', 'theta', 'psi', 'dissoc']}

# === MOLLER 1988 =============================================================

# Debye-Hueckel slope
cf_M88['Aosm'] = coeffs.Aosm_M88

# betas and Cs as cf['bC']['cation-anion']
cf_M88['bC']['Ca-Cl']  = coeffs.CaCl_M88
cf_M88['bC']['Ca-SO4'] = coeffs.CaSO4_M88
cf_M88['bC']['Na-Cl']  = coeffs.NaCl_M88
cf_M88['bC']['Na-SO4'] = coeffs.NaSO4_M88

# thetas as cf['theta']['cation1-cation2'] with cations in alph. order
cf_M88['theta']['Ca-Na']  = coeffs.CaNa_M88
cf_M88['theta']['Cl-SO4'] = coeffs.ClSO4_M88

## Unsymmetrical mixing terms
#cf_M88['etheta'] = 

# psis as cf['psi']['cation1-cation2-anion'] with cations in alph. order
#   or as cf['psi']['cation-anion1-anion2']  with anions  in alph. order
cf_M88['psi']['Ca-Na-Cl']  = coeffs.CaNaCl_M88
cf_M88['psi']['Ca-Na-SO4'] = coeffs.CaNaSO4_M88
cf_M88['psi']['Ca-Cl-SO4'] = coeffs.CaClSO4_M88
cf_M88['psi']['Na-Cl-SO4'] = coeffs.NaClSO4_M88

# Dissociation constants
cf_M88['dissoc']['Kw'] = coeffs.Kw_M88
