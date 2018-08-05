from autograd import numpy as np
import numpy as xnp

# Extract and vstack a pandas series
def pd2vs(series):
    return np.vstack(series.values)

# Replicates first output (i.e. Lia) of MATLAB's ismember(A,B)
# Inputs A and B should be numpy arrays
def ismember(A,B):
    return xnp.logical_or.reduce([A == Bi for Bi in B])
