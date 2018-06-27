from autograd import numpy as np
from autograd import elementwise_grad as egrad
from scipy.misc import derivative

# Set up functions
def f(x):
    return 3.5*x**2 - 2*x

df  = egrad(f)
d2f = egrad(df)

# Evaluate derivatives with central point vs autograd
x = np.array([1], dtype='float64')

fx = f(x)

dfx_cp = derivative(f,x)
dfx_ag = df(x)

d2fx_cp = derivative(f,x, n=2)
d2fx_ag = d2f(x)

# Simulate uncertainty
xm = np.float_(5)
xu = np.float_(0.001)
xr = np.random.normal(loc=xm, scale=xu, size=int(1e6))

# Directly propagate uncertainties
fxr  = f (xr)
dfxr = df(xr)

sd_xr_dir   = np.std(xr)
sd_fxr_dir  = np.std(fxr)
sd_dfxr_dir = np.std(dfxr)

# Calculate uncertainty propagation
sd_xr_prp   = xu
sd_fxr_prp  = np.abs(df (xm)) * xu
sd_dfxr_prp = np.abs(d2f(xm)) * xu

###############################################################################

# Function of two correlated variables
def g(m,y,z):
    return y*m**2 + z*m + np.sqrt(m)
dg_dm   = egrad(g)
d2g_dm2 = egrad(dg_dm)

m = np.array([5], dtype='float64')

y = np.float_(  6)
z = np.float_(-12)

cv_yz = np.array([[0.01, 0.012],[0.012,0.02]])

m_g     = g(m,y,z)
m_dg_dm = dg_dm(m,y,z)

A = np.hstack((m**2,m))
vr_g = A @ cv_yz @ A.transpose()

dA = np.hstack((2*m,1.))
vr_dg = dA @ cv_yz @ dA.transpose()

# Brute force
bfsize = int(1e5)
yz_bf = np.random.multivariate_normal((y,z),cv_yz, size=bfsize)
cv_bf = np.cov(yz_bf.transpose()) # check it's working

m_bf = np.full(bfsize,m)

g_bf = g(m_bf,yz_bf[:,0],yz_bf[:,1])
dg_bf = dg_dm(m_bf,yz_bf[:,0],yz_bf[:,1])

vr_g_bf  = np.var(g_bf ) # compare vs vr_g
vr_dg_bf = np.var(dg_bf) # compare vs vr_dg
