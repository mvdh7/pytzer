import pandas as pd

pm73 = pd.read_excel('datasets/PM73.xlsx')

f = open('PM73_funcs.py','w')

for i in range(len(pm73.index)):
    
    cat = pm73.cation[i].replace('(','').replace(')','')
    ani = pm73.anion [i].replace('(','').replace(')','')

    f.write('# --- bC: ----------------------------- \n\n')
    
    f.write('def bC_' + cat + '_' + ani + '_PM73(T):\n\n')
    
    f.write('    # Coefficients from PM73 Table ' + pm73.Table[i] + '\n\n')
    
    f.write('    b0   = np.full_like(T,' + str(pm73.beta0[i]) + ' * ' \
            + str(pm73.Mbeta0[i]) + ', dtype=\'float64\')\n')
    
    f.write('    b1   = np.full_like(T,' + str(pm73.beta1[i]) + ' * ' \
            + str(pm73.Mbeta1[i]) + ', dtype=\'float64\')\n')
    
    f.write('    b2   = np.zeros_like(T)\n')
    
    f.write('    Cphi = np.full_like(T,' + str(pm73.Cphi[i]) + ' * ' \
            + str(pm73.MCphi[i]) + ', dtype=\'float64\')\n\n')
    
    f.write('    z' + cat + ' = np.float_(' + str(pm73.zC[i]) \
            + ')\n')
    f.write('    z' + ani + ' = np.float_(' + str(pm73.zA[i]) \
            + ')\n')
    f.write('    C0  = Cphi / (2 * np.sqrt(np.abs(z' + cat \
            + '*z' + ani + ')))\n')
    f.write('    C1   = np.zeros_like(T)\n\n')
    
    f.write('    alph1 = np.float_(2)\n')
    f.write('    alph2 = -9\n')
    f.write('    omega = -9\n\n')
    
    f.write('    valid = T == 298.15\n\n')
    
    f.write('    return b0, b1, b2, C0, C1, alph1, alph2, omega, valid\n\n')

f.close()