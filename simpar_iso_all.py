from autograd import numpy as np
from scipy.io import savemat
from scipy.stats import norm
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

# Create overall sys 'all' field - one way only
isoerr_sys['all1w'] = np.array([])
for E,tst in enumerate(eles):
    for ref in eles[E+1:]:
        
        trtxt = 't' + tst + '_r' + ref
        isoerr_sys['all1w'] = np.append(isoerr_sys['all1w'],
                                        isoerr_sys[trtxt]['all'])

# Get summary statistics
isoerr_sys['qsd'] = np.float_(0.85)
isoerr_sys['all_qsd'] = np.diff(np.quantile(isoerr_sys['all'],
    [1-isoerr_sys['qsd'],isoerr_sys['qsd']])) \
    / (2 * norm.ppf(isoerr_sys['qsd']))
    
# Save for plotting and simloop
savemat('pickles/simpar_iso_all.mat',{'isoerr_sys':isoerr_sys,
                                      'isoerr_rdm':isoerr_rdm})
with open('pickles/simpar_iso_all.pkl','wb') as f:
    pickle.dump((isoerr_sys,isoerr_rdm),f)
    