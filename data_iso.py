from autograd import numpy as np
import pytzer as pz
import pandas as pd

isobase = pd.read_excel('datasets/iso.xlsx', sheet_name='Measurements',
                        header=0, skiprows=2, usecols=41)

isopair = ['KCl','NaCl']

def get_isopair(isobase,isopair):
    
    C = ['t','t_unc','t_scale','reps','src','via',isopair[0],isopair[1]]
    
    L = np.logical_and(np.logical_not(np.isnan(isobase[isopair[0]])),
                       np.logical_not(np.isnan(isobase[isopair[1]])))
    
    isobase = isobase.loc[L,C]
    
    
    
    return isobase#, mols_0, ions_0, mols_1, ions_1, T

test = get_isopair(isobase,isopair)
