from autograd.numpy import array, sqrt
from autograd import elementwise_grad as grad

# For pure water!
# Source: http://www.teos-10.org/pubs/IAPWS-2009-Supplementary.pdf
#
# TODO: seawater from http://www.teos-10.org/pubs/IAPWS-08.pdf

def g(tempK, pres):

    # Coefficients of the Gibbs function as defined in Table 2
    Gdict = {
    (0, 0):  0.101342743139674e3,
    (3, 2):  0.499360390819152e3,
    (0, 1):  0.100015695367145e6,
    (3, 3): -0.239545330654412e3,
    (0, 2): -0.254457654203630e4,
    (3, 4):  0.488012518593872e2,
    (0, 3):  0.284517778446287e3,
    (3, 5): -0.166307106208905e1,
    (0, 4): -0.333146754253611e2,
    (4, 0): -0.148185936433658e3,
    (0, 5):  0.420263108803084e1,
    (4, 1):  0.397968445406972e3,
    (0, 6): -0.546428511471039  ,
    (4, 2): -0.301815380621876e3,
    (1, 0):  0.590578347909402e1,
    (4, 3):  0.152196371733841e3,
    (1, 1): -0.270983805184062e3,
    (4, 4): -0.263748377232802e2,
    (1, 2):  0.776153611613101e3,
    (5, 0):  0.580259125842571e2,
    (1, 3): -0.196512550881220e3,
    (5, 1): -0.194618310617595e3,
    (1, 4):  0.289796526294175e2,
    (5, 2):  0.120520654902025e3,
    (1, 5): -0.213290083518327e1,
    (5, 3): -0.552723052340152e2,
    (2, 0): -0.123577859330390e5,
    (5, 4):  0.648190668077221e1,
    (2, 1):  0.145503645404680e4,
    (6, 0): -0.189843846514172e2,
    (2, 2): -0.756558385769359e3,
    (6, 1):  0.635113936641785e2,
    (2, 3):  0.273479662323528e3,
    (6, 2): -0.222897317140459e2,
    (2, 4): -0.555604063817218e2,
    (6, 3):  0.817060541818112e1,
    (2, 5):  0.434420671917197e1,
    (7, 0):  0.305081646487967e1,
    (3, 0):  0.736741204151612e3,
    (7, 1): -0.963108119393062e1,
    (3, 1): -0.672507783145070e3,
    }
    
    # Convert temperature and pressure
    ctau = (tempK - 273.15) / 40
    cpi  = (pres - 101325) / 1e8
    
    # Initialise with zero
    Gsum = 0

    # Increment following Eq. (1)
    for j in range(8):
        for k in range(7):

            if (j, k) in Gdict.keys():
            
                Gsum = Gsum + Gdict[(j, k)] * ctau**j * cpi**k
                
    return Gsum


# Get differentials
gt = grad(g, argnum=0)
gp = grad(g, argnum=1)

gtt = grad(gt, argnum=0)
gtp = grad(gt, argnum=1)
gpp = grad(gp, argnum=1)

# Table 3, Eq. (4): density in kg / m**3
def rho(tempK, pres):
    return 1 / gp(tempK, pres)

# Table 3, Eq. (5): specific entropy in J / (kg * K)
def s(tempK, pres):
    return -gt(tempK, pres)

# Table 3, Eq. (6): specific isobaric heat capacity in J / (kg * K)
def cp(tempK, pres):
    return -tempK * gtt(tempK, pres)

# Table 3, Eq. (7): specific enthalpy in J / kg
def h(tempK, pres):
    return g(tempK, pres) + tempK * s(tempK, pres)

# Table 3, Eq. (8): specific internal energy in J / kg
def u(tempK, pres):
    return g(tempK, pres) + tempK * s(tempK, pres) - \
        pres * gp(tempK, pres)

# Table 3, Eq. (9): specific Helmholtz energy in J / kg
def f(tempK, pres):
    return g(tempK, pres) - pres * gp(tempK, pres)

# Table 3, Eq. (10): thermal expansion coefficient in 1 / K
def alpha(tempK, pres):
    return gtp(tempK, pres) / gp(tempK, pres)

# Table 3, Eq. (11): isentropic temperature-pressure coefficient, adiabatic 
#                    lapse rate in K / Pa
def bs(tempK, pres):
    return -gtp(tempK, pres) / gp(tempK, pres)

# Table 3, Eq. (12): isothermal compressibility in 1 / Pa
def kt(tempK, pres):
    return -gpp(tempK, pres) / gp(tempK, pres)

# Table 3, Eq. (13): isentropic compressibility in 1 / Pa
def ks(tempK, pres):
    return (gtp(tempK, pres)**2 - gtt(tempK, pres) * gpp(tempK, pres)) / \
        (gp(tempK, pres) * gtt(tempK, pres))
        
# Table 3, Eq. (14): speed of sound in m / s
def w(tempK, pres):
    return gp(tempK, pres) * sqrt(gtt(tempK, pres) / \
        (gtp(tempK, pres)**2 - gtt(tempK, pres) * gpp(tempK, pres)))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Evaluate check values from Table 6 - all perfect
tempK = array([273.15, 273.15, 313.15]) # K
pres = array([101325, 1e8, 101325]) # Pa

test00_Gibbs = g(tempK, pres)
test01_gt = gt(tempK, pres)
test02_gp = gp(tempK, pres)
test03_gtt = gtt(tempK, pres)
test04_gtp = gtp(tempK, pres)
test05_gpp = gpp(tempK, pres)
test06_h = h(tempK, pres)
test07_f = f(tempK, pres)
test08_u = u(tempK, pres)
test09_s = s(tempK, pres)
test10_rho = rho(tempK, pres)
test11_cp = cp(tempK, pres)
test12_w = w(tempK, pres)
