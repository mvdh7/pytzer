from . import coeffs

# === MOLLER 1988 =============================================================
M88 = {coeff:{} for coeff in ['bC', 'theta', 'psi', 'dissoc']}

# Debye-Hueckel slope
M88['Aosm'] = coeffs.Aosm_M88

# betas and Cs as cf['bC']['cation-anion']
M88['bC']['Ca-Cl' ] = coeffs.CaCl_M88
M88['bC']['Ca-SO4'] = coeffs.CaSO4_M88
M88['bC']['Na-Cl' ] = coeffs.NaCl_M88
M88['bC']['Na-SO4'] = coeffs.NaSO4_M88

# thetas as cf['theta']['cation1-cation2'] with cations in alph. order
# c-c'
M88['theta']['Ca-Na' ] = coeffs.CaNa_M88
# a-a'
M88['theta']['Cl-SO4'] = coeffs.ClSO4_M88

## Unsymmetrical mixing terms
#M88['etheta'] = 

# psis as cf['psi']['cation1-cation2-anion'] with cations in alph. order
#   or as cf['psi']['cation-anion1-anion2']  with anions  in alph. order
# c-c'-a
M88['psi']['Ca-Na-Cl' ] = coeffs.CaNaCl_M88
M88['psi']['Ca-Na-SO4'] = coeffs.CaNaSO4_M88
# c-a-a'
M88['psi']['Ca-Cl-SO4'] = coeffs.CaClSO4_M88
M88['psi']['Na-Cl-SO4'] = coeffs.NaClSO4_M88

# Dissociation constants
M88['dissoc']['Kw'] = coeffs.Kw_M88

# === GREENBERG & MOLLER 1989 =================================================
GM89 = {coeff:{} for coeff in ['bC', 'theta', 'psi', 'dissoc']}

# Debye-Hueckel slope
GM89['Aosm'] = coeffs.Aosm_M88

# betas and Cs as cf['bC']['cation-anion']
GM89['bC']['Ca-Cl' ] = coeffs.CaCl_GM89
GM89['bC']['Ca-SO4'] = coeffs.CaSO4_M88
GM89['bC']['K-Cl'  ] = coeffs.KCl_GM89
GM89['bC']['K-SO4' ] = coeffs.KSO4_GM89
GM89['bC']['Na-Cl' ] = coeffs.NaCl_M88
GM89['bC']['Na-SO4'] = coeffs.NaSO4_M88

# thetas as cf['theta']['cation1-cation2'] with cations in alph. order
# c-c'
GM89['theta']['Ca-K'  ] = coeffs.CaK_GM89
GM89['theta']['Ca-Na' ] = coeffs.CaNa_M88
GM89['theta']['K-Na'  ] = coeffs.KNa_GM89
# a-a'
GM89['theta']['Cl-SO4'] = coeffs.ClSO4_M88

## Unsymmetrical mixing terms
#GM89['etheta'] = 

# psis as cf['psi']['cation1-cation2-anion'] with cations in alph. order
#   or as cf['psi']['cation-anion1-anion2']  with anions  in alph. order
# c-c'-a
GM89['psi']['Ca-K-Cl'  ] = coeffs.CaKCl_GM89
GM89['psi']['Ca-K-SO4' ] = coeffs.CaKSO4_GM89
GM89['psi']['Ca-Na-Cl' ] = coeffs.CaNaCl_M88
GM89['psi']['Ca-Na-SO4'] = coeffs.CaNaSO4_M88
GM89['psi']['K-Na-Cl'  ] = coeffs.KNaCl_GM89
GM89['psi']['K-Na-SO4' ] = coeffs.KNaSO4_GM89
# c-a-a'
GM89['psi']['Ca-Cl-SO4'] = coeffs.CaClSO4_M88
GM89['psi']['K-Cl-SO4' ] = coeffs.KClSO4_GM89
GM89['psi']['Na-Cl-SO4'] = coeffs.NaClSO4_M88

# Dissociation constants
GM89['dissoc']['Kw'] = coeffs.Kw_M88
