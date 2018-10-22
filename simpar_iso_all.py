from autograd import numpy as np
from scipy.io import savemat
import pickle

# Define electrolytes to test
eles = ['NaCl','KCl','CaCl2']

# Load simpar_iso output files
isoerr_sys = {}
isoerr_rdm = {}

for tst in eles:
    for ref in eles:
        if tst != ref:
            
            trtxt = 't' + tst + '_r' + ref
            with open('pickles/simpar_iso_isoerr_'+trtxt+'.pkl','rb') as f:
                isoerr_sys[trtxt],isoerr_rdm[trtxt] = pickle.load(f)
            
            # Create 'all' field for each source
            isoerr_sys[trtxt]['all'] = np.array([isoerr_sys[trtxt][src] \
                for src in isoerr_sys[trtxt].keys()])

# Create overall sys 'all' field
isoerr_sys['all'] = np.concatenate([isoerr_sys[trtxt]['all'] \
                                    for trtxt in isoerr_sys.keys()])

# Save for plotting and simloop
savemat('pickles/simpar_iso_all.mat',{'isoerr_sys':isoerr_sys,
                                      'isoerr_rdm':isoerr_rdm})
    