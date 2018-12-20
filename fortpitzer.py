import pytzer as pz
import numpy as np

m = np.vstack(np.linspace(0.5,5,10))
sqm = np.sqrt(m)
#sqm = np.vstack(np.linspace(0.5,np.sqrt(6.25),10))

osm = pz.model.osm(np.concatenate((sqm**2,sqm**2),1),
                   np.array(['Na','Cl']),
                   np.full_like(sqm,298.15),
                   pz.cdicts.MPH)

bCs = pz.cdicts.MPH.bC['Na-Cl'](298.15)
Aosm = pz.cdicts.MPH.dh['Aosm'](298.15)
