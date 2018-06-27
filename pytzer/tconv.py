from autograd import numpy as np
from . import model

# Convert freezing point depression to water activity
def fpd2aw(fpd):

    # Equation coefficients from S.L. Clegg (pers. comm., 2018)
    lg10aw = \
        - np.float_(4.209099e-03) * fpd    \
        - np.float_(2.151997e-06) * fpd**2 \
        + np.float_(3.233395e-08) * fpd**3 \
        + np.float_(3.445628e-10) * fpd**4 \
        + np.float_(1.758286e-12) * fpd**5 \
        + np.float_(7.649700e-15) * fpd**6 \
        + np.float_(3.117651e-17) * fpd**7 \
        + np.float_(1.228438e-19) * fpd**8 \
        + np.float_(4.745221e-22) * fpd**9
    
    return np.exp(lg10aw * np.log(10))

# Convert freezing point depression to osmotic coefficient
def fpd2osm(mols,fpd):
    return model.aw2osm(mols,fpd2aw(fpd))
