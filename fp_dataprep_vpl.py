import numpy  as np
import pickle
import pytzer as pz

ele = 'NaCl'

# Load raw datasets
with open('pickles/simpar_vpl.pkl','rb') as f:
    vplbase,mols,ions,T,vplerr_sys,vplerr_rdm = pickle.load(f)

# Select electrolyte for analysis
vplbase,mols,Eions,vpl_T = pz.data.subset_ele(vplbase,mols,ions,T,[ele])

# Extract columns from vplbase
vpl_tot = pz.misc.pd2vs(vplbase.m)

srcs = {ki:i+1 for i,ki in enumerate(vplerr_rdm[ele].keys())}
vpl_src = np.vstack([srcs[ki] for ki in vplbase.src])

# Write to file - one per sample
f = open('E:/Dropbox/_UEA_MPH/fort-pitzer/datasets/' + ele \
         + '_vpl_samples.dat','w')

f.write('%i\n' % np.size(vpl_tot))

for i in range(np.size(vpl_tot)):
    
    f.write('%i %.8f %.2f\n' % (vpl_src[i],vpl_tot[i],vpl_T[i]))

f.close()

# Write to file - one per dataset
f = open('E:/Dropbox/_UEA_MPH/fort-pitzer/datasets/' + ele \
         + '_vpl_datasets.dat','w')

f.write('%i\n' % len(srcs))

for src in srcs.keys():
    
#    f.write('%i\n' % srcs[src])
    
    f.write('%i %i %.8f %.8f %.8f\n' % (srcs[src],1,
                                        vplerr_rdm[ele][src][0],
                                        vplerr_rdm[ele][src][1],
                                        0))

f.close()

