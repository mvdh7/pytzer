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

vpl_type = np.full_like(vpl_tot,1)

# Write to file
f = open('pickles/fp_dataprep_vpl_v2.dat','w')

f.write('%i\n' % np.size(vpl_tot))

for i in range(np.size(vpl_tot)):
    
    f.write('%i %i %.8f %.2f\n' % (vpl_src[i],vpl_type[i],vpl_tot[i],vpl_T[i]))

f.close()
