import autograd.numpy as np
from autograd import elementwise_grad as egrad
import pandas as pd

##### FILE I/O ################################################################

def getIons(filename):
  
    # Import input conditions from .csv
    df = pd.read_csv(filename)
        
    # Get temperatures
    T = df.temp.values
    
    # Get ionic concentrations
    df_tots = df[df.keys()[df.keys() != 'temp']]
    tots = df_tots.values
    
    # Get list of ions
    ions = df_tots.keys()
       
    return T, tots, ions, df

##### IONIC CHARGES ###########################################################

def getCharges(ions):

    z = {}
    
    z['Na'] = np.float_(+1)
    z['K' ] = np.float_(+1)
    
    z['Cl'] = np.float_(-1)
    
    return np.array([z[ion] for ion in ions])

##### PITZER MODEL SUBFUNCTIONS ###############################################

def g(x):  # CRP94 Eq. (AI13)
    return 2 * (1 - (1 + x) * np.exp(-x)) / x**2

def B(b0,b1,a,I): # CRP94 Eq. (AI8)
    return b0 + b1 * g(a*np.sqrt(I))

def CT(Cphi,z1,z2): # P91 Ch. 3 Eq. (53)
    return Cphi / (2 * np.sqrt(np.abs(z1*z2)))

def fG(T,I): # CRP94 Eq. (AI1)
    return -4 * Aosm_M88(T) * I * np.log(1 + b*np.sqrt(I)) / b
   
##### DEBYE-HUECKEL SLOPE #####################################################

def param_M88(T,a): # M88 Eq. (II-13) = GM89 Eq. (3)
    return a[0] + a[1]*T + a[2]/T + a[3]*np.log(T) + a[4]/(T-263.) \
        + a[5]*T**2 + a[6]/(680.-T) + a[7]/(T-227.)

def Aosm_M88(T):
    
    # Coefficients from M88 Table 1
    a = np.float_([ 3.36901532e-1,
                   -6.32100430e-4,
                    9.14252359e00,
                   -1.35143986e-2,
                    2.26089488e-3,
                    1.92118597e-6,
                    4.52586464e+1,
                    0            ])
    
    return param_M88(T,a)

##### PITZER MODEL COEFFICIENTS ###############################################

b = np.float_(1.2)

def getbC_M88(T,cation,anion):
    
    b0   = param_M88(T,bCa[cation][anion][0])
    b1   = param_M88(T,bCa[cation][anion][1])
    Cphi = param_M88(T,bCa[cation][anion][2])
    
    return b0,b1,Cphi,alp[cation][anion]

# +++ bs and Cs +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

bCa = {}
alp  = {}

# --- Sodium ------------------------------------------------------------------
bCa['Na'] = {}
alp['Na'] = {}

# Na-Cl: M88 Table 1
alp['Na']['Cl'] = np.float_(2)
bCa['Na']['Cl'] = np.float_([[ 1.43783204e+1,
                               5.60767406e-3,
                              -4.22185236e+2,
                              -2.51226677e00,
                               0            ,
                              -2.61718135e-6,
                               4.43854508e00,
                              -1.70502337e00],
                             [-4.83060685e-1,
                               1.40677479e-3,
                               1.19311989e+2,
                               0            ,
                               0            ,
                               0            ,
                               0            ,
                              -4.23433299e00],
                             [-1.00588714e-1,
                              -1.80529413e-5,
                               8.61185543e00,
                               1.24880954e-2,
                               0            ,
                               3.41172108e-8,
                               6.83040995e-2,
                               2.93922611e-1]])

# --- Potassium ---------------------------------------------------------------
bCa['K'] = {}
alp['K'] = {}

# K-Cl: GM89 Table 1
alp['K']['Cl']  = np.float_(2)
bCa['K']['Cl']  = np.float_([[ 2.67375563e+1,
                               1.00721050e-2,
                              -7.58485453e+2,
                              -4.70624175e00,
                               0            ,
                              -3.75994338e-6,
                               0            ,
                               0            ],
                             [-7.41559626e00,
                               0            ,
                               3.22892989e+2,
                               1.16438557e00,
                               0            ,
                               0            ,
                               0            ,
                              -5.94578140e00],
                             [-3.30531334e00,
                              -1.29807848e-3,
                               9.12712100e+1,
                               5.86450181e-1,
                               0            ,
                               4.95713573e-7,
                               0            ,
                               0            ]])

# +++ Thetas ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
def getTh_M88(T,ion0,ion1):
    return param_M88(T,tha[ion0][ion1])

tha = {ion:{} for ion in ['Na','K']}

# --- Cations -----------------------------------------------------------------

# Na-K: GM89 Table 2
tha['Na']['K'] = np.float_([-5.02312111e-2,
                             0            ,
                             1.40213141e+1,
                             0            ,
                             0            ,
                             0            ,
                             0            ,
                             0            ])
tha['K']['Na'] = tha['Na']['K']
    
##### EXCESS GIBBS ENERGY #####################################################
        
def Gex_nRT(T,tots,ions):
    
    zs = getCharges(ions)
    
    # Ionic strength etc.
    I = 0.5 * (np.sum(tots * zs**2, 1))
    Z = np.sum(tots * np.abs(zs), 1)

    # Separate cations and anions
    cats    = tots[:,zs > 0]
    cations = ions[  zs > 0]
    zCs     = zs  [  zs > 0]
    anis    = tots[:,zs < 0]
    anions  = ions[  zs < 0]
    zAs     = zs  [  zs < 0]

    # Debye-Hueckel
    Gex_nRT = fG(T,I)
    
    # c-a interactions
    for C,cation in enumerate(cations):
        for A,anion in enumerate(anions):
            
            b0,b1,Cphi,a = getbC_M88(T,cation,anion)
            
            Gex_nRT = Gex_nRT + cats[:,C] * anis[:,A] \
                * (2*B(b0,b1,a,I) + Z*CT(Cphi,zCs[C],zAs[A]))
    
    # c-c' interactions
    for C0 in range(len(cations)):
        for C1 in range(C0+1,len(cations)):
            
            thC = getTh_M88(T,cations[C0],cations[C1])
            
            Gex_nRT = Gex_nRT + cats[:,C0] * cats[:,C1] \
                * 2 * (thC)# + pz.etheta(t,zC[C0],zC[C1],I))
    
    return Gex_nRT

##### IONIC ACTIVITIES ########################################################

facts = egrad(lambda tots:Gex_nRT(T,tots,ions))

##### TEST AREA ###############################################################

T, tots,ions,df = getIons('ions_in.csv')

Gex = Gex_nRT(T,tots,ions)

ln_acts = facts(tots)
acts = np.exp(ln_acts)

#test, z = Gex_nRT(T,tots,ions)

#def gradtest(v1,v2,v3):
#    return v1**2 + v2 + 3. + v3**3
#v1 = np.array([1.5,2.5])
#v2 = np.array([2.1,2.4])
#v3 = np.array([5.6,1.3])
#
#x = gradtest(v1,v2,v3)
#
#gtest = egrad(gradtest)
#
#dx = gtest(v1,v2,v3)
#
#def gtest2(vrs):
#    return vrs[:,0]**2 + vrs[:,1] + 3. + vrs[:,2]**3
#dgtest2 = egrad(gtest2)
#
#vrs = np.vstack((v1,v2,v3)).transpose()
#
#x2 = gtest2(vrs)
#
#dx2 = dgtest2(vrs)
