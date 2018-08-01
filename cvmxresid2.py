import numpy as np
from matplotlib import pyplot as plt

N = 50

S = 1.
R = 0.3

cv = np.full((N,N), S**2)
np.fill_diagonal(cv,S**2 + R**2)

x = np.linspace(0,10,N)

y1 =  np.random.normal(loc=0, scale=S) \
    + np.random.normal(loc=0, scale=R, size=N)

y2 =  np.random.normal(loc=0, scale=S) \
    + np.random.normal(loc=0, scale=R, size=N)

z1 = np.random.multivariate_normal(np.zeros(N),cv)
z2 = np.random.multivariate_normal(np.zeros(N),cv)

_,ax = plt.subplots(1,1)

ax.plot((-0.5,10.5),(0,0), color='k')

ax.scatter(x,y1, color='indigo', alpha=0.7)
ax.scatter(x,y2, color='purple', alpha=0.7)

ax.scatter(x,z1, color='xkcd:grass', alpha=0.7)
ax.scatter(x,z2, color='xkcd:forest', alpha=0.7)

ax.set_xlim((-0.5,10.5))
ax.set_ylim((-3,3))
