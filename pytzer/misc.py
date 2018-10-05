from autograd import numpy as np
import numpy as xnp

# Extract and vstack a pandas series
def pd2vs(series):
    return np.vstack(series.values)

# Replicates first output (i.e. Lia) of MATLAB's ismember(A,B)
# Inputs A and B should be numpy arrays
def ismember(A,B):
    return xnp.logical_or.reduce([A == Bi for Bi in B])

# Pure water vapour pressure following Buck (1981, 2012)
#  see https://en.wikipedia.org/wiki/Arden_Buck_equation
def vp_H2O(T):
    # input  T  in K
    # output VP in kPa
    return 0.61121 * np.exp((18.678 \
        - (T-273.15)/234.5) * ((T-273.15) / (257.14 + T - 273.15)))
