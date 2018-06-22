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
    z['Ca'] = np.float_(+2)
    
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

bCdf = {coeff:pd.read_excel('M88.xlsx', sheet_name=coeff) \
        for coeff in ['b0','b1','Cphi']}

bCa = {cation:{} for cation in bCdf['b0'].cation}
alp = {cation:{} for cation in bCdf['b0'].cation}

alist = ['a'+str(n) for n in range(1,9)]

for C,cation in enumerate(bCdf['b0'].cation):
    bCa[cation][bCdf['b0'].anion[C]] = np.array([ \
        bCdf['b0'][alist].loc[C].values,
        bCdf['b1'][alist].loc[C].values,
        bCdf['Cphi'][alist].loc[C].values])
    alp[cation][bCdf['b0'].anion[C]] = np.float_(bCdf['b1'].alpha[C])

# +++ thetas and psis +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
def getTh_M88(T,ion0,ion1):
    return param_M88(T,tha[ion0][ion1])

thdf = pd.read_excel('M88.xlsx', sheet_name='theta')

tha = {ion:{} for ion in pd.concat((thdf.ion0,thdf.ion1)).unique()}
psa = {ion:{} for ion in pd.concat((thdf.ion0,thdf.ion1)).unique()}

for I,ion0 in enumerate(thdf.ion0):
    tha[ion0][thdf.ion1[I]] = thdf[alist].values[0]
    tha[thdf.ion1[I]][ion0] = thdf[alist].values[0]
    psa[ion0][thdf.ion1[I]] = {}
    psa[thdf.ion1[I]][ion0] = {}
    
def getPs_M88(T,ion0,ion1,xion):
    return param_M88(T,psa[ion0][ion1][xion])
    
psdf = pd.read_excel('M88.xlsx', sheet_name='psi')

for I,ion0 in enumerate(psdf.ion0):
    psa[ion0][psdf.ion1[I]][psdf.xion[I]] = psdf[alist].values[0]
    psa[psdf.ion1[I]][ion0][psdf.xion[I]] = psdf[alist].values[0]
    
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
    
    # c-c'-a interactions
            for A in range(len(anions)):
                
                psC = getPs_M88(T,cations[C0],cations[C1],anions[A])
                
                Gex_nRT = Gex_nRT + cats[:,C0] * cats[:,C1] \
                    * anis[:,A] * psC
    
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
