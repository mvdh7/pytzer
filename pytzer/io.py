from autograd import numpy as np
import pandas as pd

##### FILE I/O ################################################################

def getIons(filename):
  
    # Import input conditions from .csv
    idf = pd.read_csv(filename, float_precision='round_trip')
        
    # Replace missing values with zero
    idf = idf.fillna(0)
    
    # Get temperatures
    T = idf.temp.values
    
    # Get ionic concentrations
    idf_tots = idf[idf.keys()[idf.keys() != 'temp']]
    tots = idf_tots.values
    
    # Get list of ions
    ions = np.array(idf_tots.keys())
       
    return T, tots, ions, idf
