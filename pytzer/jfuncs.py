import autograd.numpy as np
from autograd import elementwise_grad as egrad


# === Pitzer (1975) Eq. (46) ==================================================

def P75_eq46(x):
    
    def J(x):
    
        # P75 Table III
        C = np.float_([ 4.118 ,
                        7.247 ,
                       -4.408 ,
                        1.837 ,
                       -0.251 ,
                        0.0164])

        Jsum = x * 0
        
        for k in range(6):
            Jsum = Jsum + C[k] * x**-(k+1)
        
        return -x**2 * np.log(x) * np.exp(-10 * x**2) / 6 + 1 / Jsum
    
    Jp = egrad(J)
    
    return J(x), Jp(x)


# === Pitzer (1975) Eq. (47) ==================================================

def P75_eq47(x):
    
    def J(x):
        return x / (4 + 4.581 * x**-0.7237 * np.exp(-0.0120 * x**0.528))
    
    Jp = egrad(J)
    
    return J(x), Jp(x)


# === Harvie's method as described by Pitzer (1991) Ch. 3, pp. 124-125 ========

# This function works in isolation, but autograd doesn't like it!

def Harvie(x):
    
    J  = np.full_like(x,np.nan, dtype='float64')
    Jp = np.full_like(x,np.nan, dtype='float64')
    
    for s in range(len(x)):
    
        if x[s] < 1.:
        
            ak = np.float_([ 1.925154014814667,
                            -0.060076477753119,
                            -0.029779077456514,
                            -0.007299499690937,
                             0.000388260636404,
                             0.000636874599598,
                             0.000036583601823,
                            -0.000045036975204,
                            -0.000004537895710,
                             0.000002937706971,
                             0.000000396566462,
                            -0.000000202099617,
                            -0.000000025267769,
                             0.000000013522610,
                             0.000000001229405,
                            -0.000000000821969,
                            -0.000000000050847,
                             0.000000000046333,
                             0.000000000001943,
                            -0.000000000002563,
                            -0.000000000010991])
        
            z = 4 * x[s]**0.2 - 2
            dz_dx = 4 * x[s]**-0.8 / 5
        
            bk = np.zeros(np.size(ak)+2, dtype='float64')
            dk = np.zeros(np.size(ak)+2, dtype='float64')
            
            for i in reversed(range(21)):
                bk[i] = z*bk[i+1] - bk[i+2] + ak[i]
                dk[i] = bk[i+1] + z*dk[i+1] - dk[i+2]
            
        else:
            
            ak = np.float_([ 0.628023320520852,
                             0.462762985338493,
                             0.150044637187895,
                            -0.028796057604906,
                            -0.036552745910311,
                            -0.001668087945272,
                             0.006519840398744,
                             0.001130378079086,
                            -0.000887171310131,
                            -0.000242107641309,
                             0.000087294451594,
                             0.000034682122751,
                            -0.000004583768938,
                            -0.000003548684306,
                            -0.000000250453880,
                             0.000000216991779,
                             0.000000080779570,
                             0.000000004558555,
                            -0.000000006944757,
                            -0.000000002849257,
                             0.000000000237816])
        
            z = np.float_(40/9) * x[s]**-0.1 - np.float_(22/9)
            dz_dx = -4 * x[s]**-1.1 / 9
        
            bk = np.zeros(np.size(ak)+2, dtype='float64')
            dk = np.zeros(np.size(ak)+2, dtype='float64')
            
            for i in reversed(range(21)):
                bk[i] = z*bk[i+1] - bk[i+2] + ak[i]
                dk[i] = bk[i+1] + z*dk[i+1] - dk[i+2]
        
        J [s] = 0.25 * x[s] - 1 + 0.5 * (bk[0] - bk[2])
        Jp[s] = 0.25 + 0.5 * dz_dx * (dk[0] - dk[2])
        
    return J, Jp
        