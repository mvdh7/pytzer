from autograd import numpy as np

# Extract and vstack a pandas series
def pd2vs(series):
    return np.vstack(series.values)
