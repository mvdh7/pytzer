from autograd import numpy as np
import numpy as xnp

# --- Extract and vstack a pandas series --------------------------------------
def pd2vs(series):
    return np.vstack(series.values)

# --- Replicates first output (i.e. Lia) of MATLAB's ismember(A,B) ------------
# Inputs A and B should be numpy arrays
def ismember(A,B):
    return xnp.logical_or.reduce([A == Bi for Bi in B])

# --- Pure water vapour pressure following Buck (1981, 2012) ------------------
# See https://en.wikipedia.org/wiki/Arden_Buck_equation
def vp_H2O(T):
    # input  T  in K
    # output VP in kPa
    return 0.61121 * np.exp((18.678 \
        - (T-273.15)/234.5) * ((T-273.15) / (257.14 + T - 273.15)))

# --- Root-mean-square deviation ----------------------------------------------
def rms(A):
    return np.sqrt(np.mean(np.square(A)))

# --- Sn estimator of standard deviation --------------------------------------
# Source: Rousseeuw & Croux (1993), J Am Stat Assoc 88:424, 1273-1283,
#         doi:10.1080/01621459.1993.10476408

def Sn(A):
    
    Ar = A.ravel()
    
    # Eq. 2.1
    Sn0 = 1.1926 * np.median(np.array([np.median(np.abs(Ai \
        - np.concatenate([Ar[:i],Ar[i+1:]]))) for i,Ai in enumerate(Ar)]))
    
    return Sn0

# Test
A = np.vstack([3.,2.,5.,2.,6.,2.])

xSn = Sn(A)

