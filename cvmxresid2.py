import numpy as np
from matplotlib import pyplot as plt

# Set sample size
N = 50
N2 = int(N/2)

# Define systematic and random error magnitudes
S = 1.
R = 0.3

# Determine relevant covariance matrix (single dataset)
cv = np.full((N,N), S**2)
np.fill_diagonal(cv,S**2 + R**2)

# Two datasets with different systematic errors in one covariance matrix
cv2a = np.concatenate((cv[:N2,:N2],np.zeros((N2,N2))))
cv2b = np.concatenate((np.zeros((N2,N2)),cv[:N2,:N2]))
cv2 = np.concatenate((cv2a,cv2b), axis=1)

# Simulate datasets
x = np.linspace(0,10,N)

y1 =  np.random.normal(loc=0, scale=S) \
    + np.random.normal(loc=0, scale=R, size=N)

y2 =  np.random.normal(loc=0, scale=S) \
    + np.random.normal(loc=0, scale=R, size=N)

z1 = np.random.multivariate_normal(np.zeros(N),cv)
z2 = np.random.multivariate_normal(np.zeros(N),cv)

z3 = np.random.multivariate_normal(np.zeros(N),cv2)

# Simulate many datasets and check statistics
zN = np.random.multivariate_normal(np.zeros(N),cv2, size=int(1e4))
zNa_mean = np.mean(zN[:,:N2], axis=1)
zNb_mean = np.mean(zN[:,N2:], axis=1)
zNa_mean_std = np.std(zNa_mean)
zNb_mean_std = np.std(zNb_mean)
zNa_std = np.mean(np.std(zN[:,:N2], axis=1))
zNb_std = np.mean(np.std(zN[:,N2:], axis=1))

# Plot results
_,ax = plt.subplots(1,1)

ax.plot((-0.5,10.5),(0,0), color='k')

ax.scatter(x,y1, color='indigo', alpha=0.7)
ax.scatter(x,y2, color='purple', alpha=0.7)

ax.scatter(x,z1, color='xkcd:grass', alpha=0.7)
ax.scatter(x,z2, color='xkcd:forest', alpha=0.7)

ax.scatter(x,z3, color='xkcd:scarlet', alpha=0.7)

ax.set_xlim((-0.5,10.5))
ax.set_ylim((-3,3))
