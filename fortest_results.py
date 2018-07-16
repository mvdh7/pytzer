import numpy as np
import pickle

with open('pickles/fortest4.pkl','rb') as f:
    crp94,mH,mHSO4,mSO4,osmST,ln_acfPM = pickle.load(f)

acfPM = np.exp(ln_acfPM)
alpha = mSO4 / (mSO4 + mHSO4)
