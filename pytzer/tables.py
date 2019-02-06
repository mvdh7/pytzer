# pytzer: Pitzer model for chemical activities in aqueous solutions
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)

# The first set of tables from Pitzer (1991) Chapter 3 are all just
#  reproductions of the values listed by Silvester and Pitzer (1979).

# Pitzer (1991), Chapter 3, Table 12 [p109-p110]
# First temperature derivatives of b0, b1 and Cphi for 1-1 electrolytes
# 'max' gives maximum applicable molality in mol/kg
# INCOMPLETE: only includes values for H, Li, Na, K cations for now
P91_Ch3_T12 = {
    'info': 'First temperature derivatives of b0, b1 and Cphi ' \
          + 'for 1-1 electrolytes - INCOMPLETE',
    'H-Cl'   : {'b0': - 3.081e-4, 'b1':   1.419e-4, 'Cphi': - 6.213e-5, 'max': 4.5},
    'H-Br'   : {'b0': - 2.049e-4, 'b1':   4.467e-4, 'Cphi': - 5.685e-5, 'max': 6.0},
    'H-I'    : {'b0': - 0.230e-4, 'b1':   8.860e-4, 'Cphi': - 7.320e-5, 'max': 6.0},
    'H-ClO4' : {'b0':   4.905e-4, 'b1':  19.310e-4, 'Cphi': -11.770e-5, 'max': 6.0},
    'Li-Cl'  : {'b0': - 1.685e-4, 'b1':   5.366e-4, 'Cphi': - 4.520e-5, 'max': 6.4},
    'Li-Br'  : {'b0': - 1.891e-4, 'b1':   6.636e-4, 'Cphi': - 2.813e-5, 'max': 6.0},
    'Li-ClO4': {'b0':   0.386e-4, 'b1':   7.009e-4, 'Cphi': - 7.712e-5, 'max': 4.0},
    'Na-F'   : {'b0':   5.361e-4, 'b1':   8.700e-4, 'Cphi':   0       , 'max': 0.7},
    'Na-Cl'  : {'b0':   7.159e-4, 'b1':   7.005e-4, 'Cphi': -10.540e-5, 'max': 6.0},
    'Na-Br'  : {'b0':   7.692e-4, 'b1':  10.790e-4, 'Cphi': - 9.300e-5, 'max': 9.0},
    'Na-I'   : {'b0':   8.355e-4, 'b1':   8.280e-4, 'Cphi': - 8.350e-5, 'max': 6.0},
    'Na-OH'  : {'b0':   7.000e-4, 'b1':   1.340e-4, 'Cphi': -18.940e-5, 'max': 4.2},
    'Na-ClO3': {'b0':  10.350e-4, 'b1':  19.070e-4, 'Cphi': - 9.290e-5, 'max': 6.4},
    'Na-ClO4': {'b0':  12.960e-4, 'b1':  22.970e-4, 'Cphi': -16.230e-5, 'max': 6.0},
    'Na-BrO3': {'b0':   5.590e-4, 'b1':  34.370e-4, 'Cphi':   0       , 'max': 0.1},
    'Na-IO3' : {'b0':  20.660e-4, 'b1':  60.570e-4, 'Cphi':   0       , 'max': 0.1},
    'Na-SCN' : {'b0':   7.800e-4, 'b1':  20.000e-4, 'Cphi':   0       , 'max': 0.1},
    'Na-NO3' : {'b0':  12.660e-4, 'b1':  20.600e-4, 'Cphi': -23.160e-5, 'max': 2.2},
    'K-F'    : {'b0':   2.140e-4, 'b1':   5.440e-4, 'Cphi': - 5.950e-5, 'max': 5.9},
    'K-Cl'   : {'b0':   5.794e-4, 'b1':  10.710e-4, 'Cphi': - 5.095e-5, 'max': 4.5},
    'K-Br'   : {'b0':   7.390e-4, 'b1':  17.400e-4, 'Cphi': - 7.004e-5, 'max': 5.2},
    'K-I'    : {'b0':   9.914e-4, 'b1':  11.860e-4, 'Cphi': - 9.440e-5, 'max': 7.0},
    'K-ClO3' : {'b0':  19.870e-4, 'b1':  31.800e-4, 'Cphi':   0       , 'max': 0.1},
    'K-ClO4' : {'b0':   0.600e-4, 'b1': 100.700e-4, 'Cphi':   0       , 'max': 0.1},
    'K-SCN'  : {'b0':   6.870e-4, 'b1':  37.000e-4, 'Cphi':   0.430e-5, 'max': 3.1},
    'K-NO3'  : {'b0':   2.060e-4, 'b1':  64.500e-4, 'Cphi':  39.700e-5, 'max': 2.4},
    'K-H2PO4': {'b0':   6.045e-4, 'b1':  28.600e-4, 'Cphi': -10.110e-5, 'max': 1.8}}

# Pitzer (1991), Chapter 3, Table 13, Part I [p111]
# First temperature derivatives of b0, b1 and C0 for 2-1 and 1-2 electrolytes
# 'max' gives maximum applicable molality in mol/kg
P91_Ch3_T13_I = {
    'info': 'First temperature derivatives of b0, b1 and C0 ' \
          + 'for 2-1 and 1-2 electrolytes',
    'Mg-Cl'    : {'b0': -0.194e-3, 'b1':  2.78e-3, 'C0': -0.580e-4, 'max': 2.0},
    'Mg-Br'    : {'b0': -0.056e-3, 'b1':  3.86e-3, 'C0':  0       , 'max': 0.1},
    'Mg-ClO4'  : {'b0':  0.523e-3, 'b1':  4.50e-3, 'C0': -1.250e-4, 'max': 3.2},
    'Mg-NO3'   : {'b0':  0.515e-3, 'b1':  4.49e-3, 'C0':  0       , 'max': 0.1},
    'Ca-Cl'    : {'b0': -0.561e-3, 'b1':  2.66e-3, 'C0': -0.723e-4, 'max': 6.0},
    'Ca-Br'    : {'b0': -0.523e-3, 'b1':  6.04e-3, 'C0':  0       , 'max': 0.6},
    'Ca-NO3'   : {'b0':  0.530e-3, 'b1':  9.19e-3, 'C0':  0       , 'max': 0.1},
    'Ca-ClO4'  : {'b0':  0.830e-3, 'b1':  5.08e-3, 'C0': -1.090e-4, 'max': 4.0},
    'Sr-Cl'    : {'b0':  0.717e-3, 'b1':  2.84e-3, 'C0':  0       , 'max': 0.1},
    'Sr-Br'    : {'b0': -0.328e-3, 'b1':  6.53e-3, 'C0':  0       , 'max': 0.1},
    'Sr-NO3'   : {'b0':  0.177e-3, 'b1': 12.47e-3, 'C0':  0       , 'max': 0.2},
    'Sr-ClO4'  : {'b0':  1.143e-3, 'b1':  5.39e-3, 'C0': -1.100e-4, 'max': 3.0},
    'Ba-Cl'    : {'b0':  0.640e-3, 'b1':  3.23e-3, 'C0': -0.540e-4, 'max': 1.8},
    'Ba-Br'    : {'b0': -0.338e-3, 'b1':  6.78e-3, 'C0':  0       , 'max': 0.1},
    'Ba-NO3'   : {'b0': -2.910e-3, 'b1': 29.10e-3, 'C0':  0       , 'max': 0.1},
    'Mnjj-ClO4': {'b0':  0.397e-3, 'b1':  5.03e-3, 'C0': -1.180e-4, 'max': 4.0},
    'Cojj-ClO4': {'b0':  0.545e-3, 'b1':  5.36e-3, 'C0': -1.270e-4, 'max': 4.0},
    'Nijj-ClO4': {'b0':  0.666e-3, 'b1':  4.76e-3, 'C0': -1.350e-4, 'max': 4.0},
    'Cujj-Cl'  : {'b0': -2.710e-3, 'b1':  8.50e-3, 'C0':  0       , 'max': 0.6},
    'Znjj-ClO4': {'b0':  0.596e-3, 'b1':  5.09e-3, 'C0': -1.360e-4, 'max': 4.0},
    'Li-SO4'   : {'b0':  0.506e-3, 'b1':  1.41e-3, 'C0': -0.825e-4, 'max': 3.0},
    'Na-SO4'   : {'b0':  2.367e-3, 'b1':  5.63e-3, 'C0': -1.725e-4, 'max': 3.0},
    'K-SO4'    : {'b0':  1.440e-3, 'b1':  6.70e-3, 'C0':  0       , 'max': 0.1},
    'Rb-SO4'   : {'b0':  0.940e-3, 'b1':  8.64e-3, 'C0':  0       , 'max': 0.1},
    'Cs-SO4'   : {'b0': -0.893e-3, 'b1': 14.48e-3, 'C0':  0       , 'max': 0.1}}

# Pitzer (1991), Chapter 3, Table 13, Part II [p111]
# First temperature derivatives of b0, b1, b2 and C0 for 3-1 and 2-2 electrolytes
# 'max' gives maximum applicable molality in mol/kg
# alph1 = 1.4 for the M-SO4 b1's, alph1 = 2.0 otherwise
# alph2 = 12.0 throughout
#
# Na3-FeCN5 is a typo for Na3-FeCN6
P91_Ch3_T13_II = {
    'info': 'First temperature derivatives of b0, b1, b2 and C0 ' \
          + 'for 3-1 and 2-2 electrolytes',
    'La-Cl'      : {'b0':  0.253e-3, 'b1': 0.798e-2, 'b2':  0      , 'C0': -0.107e-3, 'max': 3.6 },
    'La-ClO4'    : {'b0':  0.152e-3, 'b1': 1.503e-2, 'b2':  0      , 'C0': -0.194e-3, 'max': 2.1 },
    'La-NO3'     : {'b0':  0.173e-3, 'b1': 1.095e-2, 'b2':  0      , 'C0': -0.130e-3, 'max': 2.2 },
    'Na-FejjjCN6': {'b0':  3.050e-3, 'b1': 1.520e-2, 'b2':  0      , 'C0':  0       , 'max': 0.1 },
    'K-FejjjCN6' : {'b0': -0.870e-3, 'b1': 3.150e-2, 'b2':  0      , 'C0':  0       , 'max': 0.1 },
    'K-FejjCN6'  : {'b0':  4.740e-3, 'b1': 3.920e-2, 'b2':  0      , 'C0':  0       , 'max': 0.2 },
    'Mg-SO4'     : {'b0': -0.690e-3, 'b1': 1.530e-2, 'b2': -2.53e-1, 'C0':  0.131e-3, 'max': 2.0 },
    'Ca-SO4'     : {'b0':  0       , 'b1': 5.460e-2, 'b2': -5.16e-1, 'C0':  0       , 'max': 0.02},
    'Cu-SO4'     : {'b0': -4.400e-3, 'b1': 2.380e-2, 'b2': -4.73e-1, 'C0':  1.200e-3, 'max': 1.0 },
    'Zn-SO4'     : {'b0': -3.660e-3, 'b1': 2.330e-2, 'b2': -3.33e-1, 'C0':  0.990e-3, 'max': 1.0 },
    'Cd-SO4'     : {'b0': -2.790e-3, 'b1': 1.710e-2, 'b2': -5.22e-1, 'C0':  0.650e-3, 'max': 1.0 }}
