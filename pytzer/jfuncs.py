# pytzer: the Pitzer model for chemical speciation
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)

from autograd.numpy import exp, float_, full_like, log, nan, size, zeros, \
                           zeros_like
from autograd import elementwise_grad as egrad
from autograd.extend import primitive, defvjp
from scipy.misc import derivative


# === Pitzer (1975) Eq. (46) ==================================================

def P75_eq46(x):

    def J(x):

        # P75 Table III
        C = float_([ 4.118 ,
                     7.247 ,
                    -4.408 ,
                     1.837 ,
                    -0.251 ,
                     0.0164])

        Jsum = zeros_like(x)

        for k in range(6):
            Jsum = Jsum + C[k] * x**-(k+1)

        return -x**2 * log(x) * exp(-10 * x**2) / 6 + 1 / Jsum

    Jp = egrad(J)

    return J(x), Jp(x)


# === Pitzer (1975) Eq. (47) ==================================================

def P75_eq47(x):

    def J(x):
        return x / (4 + 4.581 * x**-0.7237 * exp(-0.0120 * x**0.528))

    Jp = egrad(J)

    return J(x), Jp(x)


# === Harvie's method as described by Pitzer (1991) Ch. 3, pp. 124-125 ========

# Define the raw function - doesn't work in pytzer (not autograd-able)
# Use Harvie() instead (code comes afterwards)
def _Harvie_raw(x):

    J  = full_like(x,nan, dtype='float64')
    Jp = full_like(x,nan, dtype='float64')

    for s, xs in enumerate(x):

        if xs < 1.:

            # Values from Table B-1, middle column (akI)
            ak = float_([ 1.925154014814667,
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

            z = 4 * xs**(1/5) - 2      # Eq. (B-21)
            dz_dx = 4 * xs**-(4/5) / 5 # Eq. (B-22)

            bk = zeros(size(ak)+2, dtype='float64')
            dk = zeros(size(ak)+2, dtype='float64')

            for i in reversed(range(21)):
                bk[i] = z*bk[i+1] - bk[i+2] + ak[i]   # Eq. (B-23)
                dk[i] = bk[i+1] + z*dk[i+1] - dk[i+2] # Eq. (B-24)

        else:

            # Values from Table B-1, final column (akII)
            ak = float_([ 0.628023320520852,
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

            z = 40/9 * xs**-0.1 - 22/9 # Eq. (B-25)
            dz_dx = -4 * xs**-1.1 / 9  # Eq. (B-26)

            bk = zeros(size(ak)+2, dtype='float64')
            dk = zeros(size(ak)+2, dtype='float64')

            for i in reversed(range(21)):
                bk[i] = z*bk[i+1] - bk[i+2] + ak[i]   # Eq. (B-27)
                dk[i] = bk[i+1] + z*dk[i+1] - dk[i+2] # Eq. (B-28)

        J [s] = 0.25 * xs - 1 + 0.5 * (bk[0] - bk[2]) # Eq. (B-29)
        Jp[s] = 0.25 + 0.5 * dz_dx * (dk[0] - dk[2])  # Eq. (B-30)

    return J, Jp

# Perform code gymnastics so that autograd can differentiate _Harvie_raw
@primitive
def _Harvie_J(x):
    return _Harvie_raw(x)[0]
@primitive
def _Harvie_Jp(x):
    return _Harvie_raw(x)[1]

_Harvie_dx = 1e-9
def _Harvie_J_drv(x):
    return derivative(_Harvie_J,x,  dx=_Harvie_dx) * _Harvie_dx
def _Harvie_Jp_drv(x):
    return derivative(_Harvie_Jp,x, dx=_Harvie_dx) * _Harvie_dx

def _Harvie_J_vjp(ans,x):
    return lambda g: g * _Harvie_J_drv(x)
def _Harvie_Jp_vjp(ans,x):
    return lambda g: g * _Harvie_Jp_drv(x)

defvjp(_Harvie_J ,_Harvie_J_vjp )
defvjp(_Harvie_Jp,_Harvie_Jp_vjp)

# This is the final function to call in pytzer:
def Harvie(x):
    return _Harvie_J(x), _Harvie_Jp(x)
