import numpy  as np
import pickle
import pytzer as pz

ele = 'NaCl'

# Load raw datasets
with open('pickles/simpar_fpd_osm25.pkl','rb') as f:
    fpdbase,mols,ions,T,fpderr_sys,fpderr_rdm = pickle.load(f)

# Select electrolyte for analysis
fpdbase,mols,Eions,fpd_T = pz.data.subset_ele(fpdbase,mols,ions,T,[ele])
fpd_T25 = np.full_like(fpd_T,298.15)

# Extract columns from fpdbase
fpd_tot = pz.misc.pd2vs(fpdbase.m)

srcs = {ki:i+1 for i,ki in enumerate(fpderr_rdm[ele].keys())}
fpd_src = np.vstack([srcs[ki] for ki in fpdbase.src])

# Write to file - one per sample
f = open('E:/Dropbox/_UEA_MPH/fort-pitzer/fp_dataprep_fpd_samples_' \
         + ele + '.dat','w')

f.write('%i\n' % np.size(fpd_tot))

for i in range(np.size(fpd_tot)):
    
    f.write('%i %.8f %.2f\n' % (fpd_src[i],fpd_tot[i],fpd_T25[i]))

f.close()

# Write to file - one per dataset
f = open('E:/Dropbox/_UEA_MPH/fort-pitzer/fp_dataprep_fpd_datasets_' \
         + ele + '.dat','w')

f.write('%i\n' % len(srcs))

for src in srcs.keys():
    
#    f.write('%i\n' % srcs[src])
    
    f.write('%i %i %.8f %.8f %.8f\n' % (srcs[src],1,
                                        fpderr_rdm[ele][src][0],
                                        fpderr_rdm[ele][src][1],
                                        fpderr_rdm[ele][src][2]))

f.close()
