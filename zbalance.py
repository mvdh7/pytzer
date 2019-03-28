from autograd import numpy as np
from autograd import elementwise_grad as egrad

def f(x):
    
    return np.sqrt(x)

g = egrad(f)

testx = 4.0

testf = f(testx)
testg = g(testx)
