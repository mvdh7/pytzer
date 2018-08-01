import numpy as np
import pytzer as pz
from scipy.io import savemat

sqtot = np.vstack(np.linspace(0.001,2.5,100))
tot = sqtot**2

mols = np.concatenate((tot,tot), axis=1)
ions = np.array(['Na','Cl'])
T = np.full_like(tot,298.15)
cf = pz.cdicts.MPH

osm = pz.model.osm(mols,ions,T,cf)
aw = pz.model.osm2aw(mols,osm)

savemat('pickles/vplcurve.mat',{'tot' : tot ,
                                'mols': mols,
                                'T'   : T   ,
                                'osm' : osm ,
                                'aw'  : aw  })
