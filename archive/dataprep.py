#%%
import numpy as np
import pytzer as pz

fpdbase,mols,ions = pz.data.fpd('datasets/')
    
subset = np.array(['NaCl','KCl','CaCl2'])

fpdbase,mols,ions = pz.data.subset_ele(fpdbase,mols,ions,
                                       np.array(['NaCl','KCl','CaCl2']))

cf = pz.cdicts.GM89
cf.bC['Na-Cl'] = pz.coeffs.bC_Na_Cl_A92ii

fpdbase['osm'] = pz.model.osm(mols,ions,np.vstack(fpdbase.t.values),cf)
