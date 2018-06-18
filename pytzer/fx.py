import numpy as np

def B(bC,ao,I):     # CRP94 Eq. (AI8)
    return bC[:,0] + bC[:,1] * g(ao[:,1]*np.sqrt(I)) \
                   + bC[:,2] * g(ao[:,2]*np.sqrt(I))
                   
def CT(bC,ao,I):    # CRP94 Eq. (AI11)
    return bC[:,3] + 4 * bC[:,4] * h(ao[:,4]*np.sqrt(I))

##### g AND h FUNCTIONS ######################################################

def g(x):  # CRP94 Eq. (AI13)
    g = np.zeros_like(x)
    L = x != 0
    g[L] = 2 * (1 - (1 + x[L]) * np.exp(-x[L])) / x[L]**2
    return g

def h(x):  # CRP94 Eq. (AI15)
    h = np.zeros_like(x)
    L = x != 0
    h[L] = (6 - (6 + x[L]*(6 + 3*x[L] + x[L]**2)) * np.exp(-x[L])) / x[L]**4
    return h
