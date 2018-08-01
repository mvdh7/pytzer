import numpy as np
from matplotlib import pyplot as plt

Ureps = int(5e2)

r0 = np.random.normal(loc=1,scale=0.1,size=(Ureps,1))
#r0s = np.copy(r0)
#np.random.shuffle(r0s)
#
#r0x = np.concatenate((r0,r0s),axis=1)
#
#r0cv = np.cov(r0x, rowvar=False)

bsz = Ureps
bmean = np.zeros(bsz)
bcv = np.full((bsz,bsz),0.5)
for i in range(bsz):
    for j in range(bsz):
        if i == j:
            bcv[i,j] = 0.51

b = np.random.multivariate_normal(bmean,bcv, Ureps)

b_means = np.mean(b,axis=1)
b_means_var = np.var(b_means)

b_vars = np.var(b,axis=1)

fig,ax = plt.subplots(1,1)

ax.hist(r0, bins=15, density=True)
#ax.hist(r0s, bins=50, alpha=0.5)

ax.hist(b[0], bins=15, alpha=0.5, density=True)
