from autograd import numpy as np
import pytzer as pz

tot = np.vstack([1.,2.,3.])
ions = np.array(['Na','Cl'])
nC = np.float_(1)
nA = np.float_(1)

cf = pz.cdicts.cdict()
cf.bC['Na-Cl'] = pz.coeffs.bC_Na_Cl_A92ii
cf.dh['Aosm'] = pz.coeffs.Aosm_M88
cf.dh['AH'] = pz.coeffs.AH_MPH

test = pz.tconv.tot2fpd(tot,ions,nC,nA,cf)
