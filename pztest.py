import autograd.numpy as np
from scipy.misc import derivative
import pytzer as pz
from pytzer.constants import R, Mw

# Set dict of coefficient functions
cf = pz.cdicts.cf_M88

T,tots,ions,idf = pz.miami.getIons('M88 Table 4.csv')
mols = np.copy(tots)

Gexs = pz.miami.Gex_nRT(mols,ions,T,cf)
acfs = np.exp(pz.miami.fx_ln_acfs(mols,ions,T,cf))

# Test osmotic coefficient - NaCl compares well with Archer (1992)
# M88 Table 4 also works almost perfectly, without yet including unsymm. terms!

# autograd doesn't seem to work due to broadcasting issues for osm derivative
#  hence scipy derivation here for now (which does work great)
osmD = np.full_like(T,np.nan)
for i in range(len(T)):
    osmD[i] = derivative(lambda ww: 
        ww * R*T[i] * pz.miami.Gex_nRT(np.array([mols[i,:]/ww]),ions,T[i],cf),
                         np.array([1.]), dx=1e-8)[0]
osm = 1 - osmD / (R * T * (np.sum(mols,axis=1)))

aw = np.exp(-osm * Mw * np.sum(mols,axis=1))
