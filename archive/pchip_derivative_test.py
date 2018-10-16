import pickle
from scipy.interpolate import pchip
from autograd import numpy as np
from autograd import elementwise_grad as egrad
from autograd.extend import primitive, defvjp

with open('pickles/fortest_CaCl2_10.pkl','rb') as f:
    rc97 = pickle.load(f)[0]
osm_CaCl2_PCHIP = pchip(rc97.tot,rc97.osm)

@primitive
def osm_CaCl2(tot):
    return osm_CaCl2_PCHIP(tot)

def dosm_dtot_CaCl2(tot):
    return osm_CaCl2_PCHIP.derivative()(tot)

def osm_CaCl2_vjp(ans,tot):    
    return lambda g: g * osm_CaCl2_PCHIP.derivative()(tot)
defvjp(osm_CaCl2,osm_CaCl2_vjp)

dosm_dtot_CaCl2_PCHIP = egrad(osm_CaCl2)

tot = np.array([[1.]])
test = osm_CaCl2_PCHIP(tot)
test_d_brute = (osm_CaCl2_PCHIP(tot+0.00001) - osm_CaCl2_PCHIP(tot)) / 0.00001
test_d_exact = dosm_dtot_CaCl2(tot)
test_d_egrad = dosm_dtot_CaCl2_PCHIP(tot)
