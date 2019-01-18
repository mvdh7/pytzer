import pandas as pd

HMW84 = pd.read_excel('WM13 ref.xlsx', sheet_name='HMW84 T2 cca')

cols = HMW84.columns.values[1:]

f = open('datasets/HMW84_funcs_cca.py','w')

for i in range(len(HMW84.index)):
    
    # theta function
    cats = list(HMW84.index[i])
    cats.sort()
    cats_str = '-'.join(cats)
    cats_fnc = '_'.join(cats)
    
    f.write('def theta_' + cats_fnc + '_HMW84(T):\n')
    
    f.write('# Coefficients from HMW84 Table 2\n')
            
    f.write('    theta = np.full_like(T,' + str(HMW84.theta[i]) \
            + ', dtype=\'float64\')\n')
    f.write('    valid = T == 298.15\n')
    f.write('    return theta, valid\n\n')
    
    for col in cols:
        
        # psi functions
        f.write('def psi_' + cats_fnc + '_' + col + '_HMW84(T):\n')
        
        f.write('# Coefficients from HMW84 Table 2\n')
                
        f.write('    psi = np.full_like(T,' + str(HMW84[col][i]) \
                + ', dtype=\'float64\')\n')
        f.write('    valid = T == 298.15\n')
        f.write('    return psi, valid\n\n')

f.close()