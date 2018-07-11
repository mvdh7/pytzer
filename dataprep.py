#%%
import numpy as np
import pytzer as pz
from mvdh import ismember

fpdbase = pz.data.fpd('datasets/')
    
fpdbase = fpdbase[ismember(fpdbase.ele,['NaCl','KCl','CaCl2'])]

eles = fpdbase.ele.unique()

ions,idict = pz.data.ele2ions(eles)

mols = np.zeros((np.shape(fpdbase)[0],len(ions)))

for ele in eles:
    
    C = np.where(ions == idict[ele][0])
    A = np.where(ions == idict[ele][1])
    
    EL = fpdbase.ele == ele
    
    mols[EL,C] = fpdbase.m[EL] * fpdbase.nC[EL]
    mols[EL,A] = fpdbase.m[EL] * fpdbase.nA[EL]

cf = pz.cdicts.GM89
cf.bC['Na-Cl'] = pz.coeffs.bC_Na_Cl_A92ii

fpdbase['osm'] = pz.model.osm(mols,ions,np.vstack(fpdbase.t.values),cf)
