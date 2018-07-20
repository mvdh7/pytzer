import pickle
import pytzer as pz
import numpy as np

with open('pickles/simloop_pytzer_bC_KCl.pkl','rb') as f:
    bC_mean,bC_pool_cv,ele,Ureps = pickle.load(f)

bC = pz.coeffs.bC_K_Cl_A99(np.vstack([298.15]))
