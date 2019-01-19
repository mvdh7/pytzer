import pandas as pd

HMW84 = pd.read_excel('WM13 ref.xlsx', sheet_name='HMW84 T1 bC')

f = open('datasets/HMW84_funcs_bC.py','w')

for i in range(len(HMW84.index)):
    
    ions_fnc = '_'.join(HMW84.index[i])
    
    f.write('def bC_' + ions_fnc + '_HMW84(T):\n')
    
    f.write('# Coefficients from HMW84 Table 1\n')
            
    f.write('    b0   = np.full_like(T,' + str(HMW84.b0[i]) \
            + ', dtype=\'float64\')\n')
    f.write('    b1   = np.full_like(T,' + str(HMW84.b1[i]) \
            + ', dtype=\'float64\')\n')
    f.write('    b2   = np.full_like(T,' + str(HMW84.b2[i]) \
            + ', dtype=\'float64\')\n')
    f.write('    Cphi = np.full_like(T,' + str(HMW84.Cphi[i]) \
            + ', dtype=\'float64\')\n')
    
    f.write('    z' + HMW84.index[i][0] + ' = np.float_(' + str(HMW84.zM[i]) \
            + ')\n')
    f.write('    z' + HMW84.index[i][1] + ' = np.float_(' + str(HMW84.zX[i]) \
            + ')\n')    
    f.write('    C0 = Cphi / (2 * np.sqrt(np.abs(z' + HMW84.index[i][0] \
            + '*z' + HMW84.index[i][1] + ')))\n')
    
    f.write('    C1 = np.zeros_like(T)\n')
    
    if HMW84.b2[i] == 0:
        f.write('    alph1 = np.float_(2)\n')
        f.write('    alph2 = -9\n')
    else:
        f.write('    alph1 = np.float_(1.4)\n')
        f.write('    alph2 = np.float_(12)\n')
        
    f.write('    omega = -9\n')
    
    f.write('    valid = T == 298.15\n')
    f.write('    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid\n\n')

f.close()