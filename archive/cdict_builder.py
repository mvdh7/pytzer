# Import libraries
from autograd import numpy as np
#import pandas as pd
import pytzer as pz
pd2vs = pz.misc.pd2vs
#from mvdh import ismember

# Load raw datasets
datapath = 'datasets/'
fpdbase,mols,ions = pz.data.fpd(datapath)

# Select electrolytes for analysis
fpdbase,mols,ions = pz.data.subset_ele(fpdbase,mols,ions,
                                       np.array(['NaCl','KCl','CaCl2']))

# Begin cdict
cf = pz.cdicts.cdict()

# Get lists of cations and anions
eles = fpdbase.ele

cf.add_zeros(eles)

#_,cats,anis,_ = pz.data.ele2ions(eles)
#
## Populate cdict with zero functions
#for cat in cats:
#    
#    for ani in anis:
#        cf.bC[cat + '-' + ani] = pz.coeffs.bC_zero
#        
#for C0 in range(len(cats)):
#    for C1 in range(C0+1,len(cats)):
#        
#        cf.theta[cats[C0] + '-' + cats[C1]] = pz.coeffs.theta_zero
#        
#        for ani in anis:
#            
#            cf.psi[cats[C0] + '-' + cats[C1] + '-' + ani] = pz.coeffs.psi_zero
#
#for A0 in range(len(anis)):
#    for A1 in range(A0+1,len(anis)):
#        
#        cf.theta[anis[A0] + '-' + anis[A1]] = pz.coeffs.theta_zero
#        
#        for cat in cats:
#            
#            cf.psi[cat + '-' + anis[A0] + '-' + anis[A1]] = pz.coeffs.psi_zero
