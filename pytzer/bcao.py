import numpy as np

def NaCl_A92ii(T=np.array([298.15])):

    p = 0.101325 * np.ones(np.size(T))

    # Set NaCl p/t parameters from A92ii Table 2
    bC_b0 = np.array( \
            [ 0.242408292826506,
              0,
            - 0.162683350691532,
              1.38092472558595,
              0,
              0,
            -67.2829389568145,
              0,
              0.625057580755179,
            -21.2229227815693,
             81.8424235648693,
            - 1.59406444547912,
              0,
              0,
             28.6950512789644,
            -44.3370250373270,
              1.92540008303069,
            -32.7614200872551,
              0,
              0,
             30.9810098813807,
              2.46955572958185,
            - 0.725462987197141,
             10.1525038212526   ])

    bC_b1 = np.array( \
           [- 1.90196616618343,
              5.45706235080812,
              0,
            -40.5376417191367,
              0,
              0,
              4.85065273169753  * 1e2,
            - 0.661657744698137,
              0,
              0,
              2.42206192927009  * 1e2,
              0,
            -99.0388993875343,
              0,
              0,
            -59.5815563506284,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              0                  ])

    bC_b2 = np.zeros(np.size(bC_b0))

    bC_C0 = np.array( \
           [  0,
            - 0.0412678780636594,
              0.0193288071168756,
            - 0.338020294958017,      # typo in A92ii
              0,
              0.0426735015911910,
              4.14522615601883,
            - 0.00296587329276653,
              0,
              1.39697497853107,
            - 3.80140519885645,
              0.06622025084,          # typo in A92ii - "Rard's letter"
              0,
            -16.8888941636379,
            - 2.49300473562086,
              3.14339757137651,
              0,
              2.79586652877114,
              0,
              0,
              0,
              0,
              0,
            - 0.502708980699711   ])

    bC_C1 = np.array( \
           [  0.788987974218570,
            - 3.67121085194744,
              1.12604294979204,
              0,
              0,
            -10.1089172644722,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
             16.6503495528290      ])

    bCmx = np.stack((bC_b0, bC_b1, bC_b2, bC_C0, bC_C1)).transpose()

    # Set up p/t matrix
    pTmx = [np.ones(np.size(T))         , #  1
            10**-3 * T                  , #  2
            4e-6 * T**2                 , #  3
            1 / (T - 200)               , #  4
            1 / T                       , #  5
            100 / (T - 200)**2          , #  6 - typo in A92ii
            200 / T**2                  , #  7 - typo in A92ii
            8e-9 * T**3                 , #  8
            1 / (650 - T)**0.5          , #  9
            10**-5 * p                  , # 10
            2e-4 * p / (T - 225)        , # 11
            100 * p / (650 - T)**3      , # 12
            2e-8 * p * T                , # 13
            2e-4 * p / (650 - T)        , # 14
            10**-7 * p**2               , # 15
            2e-6 * p**2 / (T - 225)     , # 16
            p**2 / (650 - T)**3         , # 17
            2e-10 * p**2 * T            , # 18
            4e-13 * p**2 * T**2         , # 19
            0.04 * p / (T - 225)**2     , # 20
            4e-11 * p * T**2            , # 21
            2e-8 * p**3 / (T - 225)     , # 22
            0.01 * p**3 / (650 - T)**3  , # 23
            200 / (650 - T)**3          ] # 24

    # Alpha and omega values
    ao = np.tile([0, 2.0, 0, 0, 2.5],(np.size(T),1))

    # Validity range
    valid = np.logical_and(T >= 250, T <= 600)

    return np.matmul(np.transpose(pTmx),bCmx), ao, valid