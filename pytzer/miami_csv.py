import autograd.numpy as np
import pandas as pd
from autograd import elementwise_grad as egrad
from scipy.optimize import minimize
from scipy import io
import time

##### FILE I/O ################################################################

def getIons(filename):
  
    # Import input conditions from .csv
    idf = pd.read_csv(filename, float_precision='round_trip')
        
    # Get temperatures
    T = idf.temp.values
    
    # Get ionic concentrations
    idf_tots = idf[idf.keys()[idf.keys() != 'temp']]
    tots = idf_tots.values
    
    # Get list of ions
    ions = idf_tots.keys()
       
    return T, tots, ions, idf

##### PITZER MODEL COEFFICIENTS ###############################################

b = np.float_(1.2)

alist = ['a'+str(n) for n in range(16)]
ilist = ['ion'+str(n) for n in range(1,4)]

def getCoeffbase(filename):
    
    # Import .csv file of coefficient values
    cdf = pd.read_csv(filename, header=1, float_precision='round_trip')
    
    # Convert missing alphas to -9
    cdf['alpha'] = cdf.alpha.fillna(-9)
    
    # Convert missing coefficients to 0
    cdf[alist] = cdf[alist].fillna(0)
    
    # Convert missing ions to ''
    cdf[ilist] = cdf[ilist].fillna('')
    
    # Get ion combination as single sorted string
    cdf['ions'] = ''
    for i in range(cdf.shape[0]):
        ions = list(cdf[ilist].loc[i])
        ions.sort()
        cdf.loc[i,'ions'] = ''.join(ions)
    
    return cdf

def evalCoeffs(T,a): # 0-7 are M88 Eq. (II-13) = GM89 Eq. (3)
                     # 8-13 are extra terms from PP87 for NaOH with P in bar
                     # 14-15 are extras from MP98
    
    Pbar = np.float_(1.01325)
    
    return a[0] \
         + a[1]  * T \
         + a[2]  / T \
         + a[3]  * np.log(T) \
         + a[4]  / (T-263.) \
         + a[5]  * T**2 \
         + a[6]  / (680.-T) \
         + a[7]  / (T-227.) \
         + a[8]  * Pbar \
         + a[9]  * Pbar/T \
         + a[10] * Pbar*T \
         + a[11] * Pbar*T**2 \
         + a[12] / (647.-T) \
         + a[13] * Pbar/(647.-T) \
         + a[14] * (T-298.15) \
         + a[15] * (T-298.15)**2

def getCoeffs(cdf,T):
    
    # Evaluate temperature-sensitive coefficients
    cf = {coeff: {ions:evalCoeffs(T,cdf[alist].loc[i].values) \
                  for i,ions in enumerate(cdf.ions) \
                  if cdf.coefficient[i] == coeff} \
          for coeff in ['Aosm','b0','b1','b2','Cphi','theta','psi']}

    # Get alphas
    cf['a1'] = {ions:cdf.alpha[i] \
                for i,ions in enumerate(cdf.ions) \
                if cdf.coefficient[i] == 'b1'}
    cf['a2'] = {ions:cdf.alpha[i] \
                for i,ions in enumerate(cdf.ions) \
                if cdf.coefficient[i] == 'b2'}
    
    # Unlayer Aosm
    cf['Aosm'] = cf['Aosm']['']
    
    return cf

##### DISSOCIATION CONSTANTS ##################################################

def getDissocbase(filename):
    
    # Import .csv file of coefficient values
    ddf = pd.read_csv(filename, float_precision='round_trip')
    
    # Convert missing coefficients to 0
    ddf = ddf.fillna(0)
    
    return ddf

def evalDissocs(ddf,T,acid): # MP98 Eq. (23)
    
    L = ddf.acid == acid
    lnK = ddf.A[L].values + ddf.B[L].values/T + ddf.C[L].values*np.log(T) \
        + ddf.D[L].values*T
    
    return np.exp(lnK)

##### IONIC CHARGES ###########################################################

def getCharges(ions):

    z = {}
    
    z['H' ] = np.float_(+1)
    z['Na'] = np.float_(+1)
    z['K' ] = np.float_(+1)
    z['Ca'] = np.float_(+2)
    
    z['OH'] = np.float_(-1)
    z['Cl'] = np.float_(-1)
    
    return np.array([z[ion] for ion in ions])

##### DEBYE-HUECKEL SLOPE #####################################################

def fG(Aosm,I): # CRP94 Eq. (AI1)
    
    return -4 * Aosm * I * np.log(1 + b*np.sqrt(I)) / b

##### PITZER MODEL SUBFUNCTIONS ###############################################

def g(x): # CRP94 Eq. (AI13)
    return 2 * (1 - (1 + x) * np.exp(-x)) / x**2

def B(cf,iset,I): # CRP94 Eq. (AI8)
    
    if iset in cf['b0']:
        b0 = cf['b0'][iset]
    else:
        b0 = 0
        print('WARNING: no b0 value for ' + iset \
              + ' found; defaulting to zero')
        
    if iset in cf['b1']:
        b1 = cf['b1'][iset]
        a1 = cf['a1'][iset]
    else:
        b1 = 0
        a1 = -9
        print('WARNING: no b1 value for ' + iset \
              + ' found; defaulting to zero')
        
    if iset in cf['b2']:
        b2 = cf['b2'][iset]
        a2 = cf['a2'][iset]
    else:
        b2 = 0
        a2 = -9
        print('WARNING: no b2 value for ' + iset \
              + ' found; defaulting to zero')

    return b0 + b1 * g(a1*np.sqrt(I)) + b2 * g(a2*np.sqrt(I))

def CT(cf,iset,z1,z2): # P91 Ch. 3 Eq. (53)
    
    if iset in cf['Cphi']:
        Cphi = cf['Cphi'][iset]
    else:
        Cphi = 0
        print('WARNING: no Cphi value for ' + iset \
              + ' found; defaulting to zero')
    
    return Cphi / (2 * np.sqrt(np.abs(z1*z2)))

##### EXCESS GIBBS ENERGY #####################################################
    
def Gex_nRT(mols,ions,cf):
    
    # Ionic strength etc.
    zs = getCharges(ions)
    I = 0.5 * (np.sum(mols * zs**2, 1))
    Z = np.sum(mols * np.abs(zs), 1)
    
    # Separate cations and anions
    CL = zs > 0
    cats    = mols[:,CL]
    cations = ions[  CL]
    zCs     = zs  [  CL]
    AL = zs < 0
    anis    = mols[:,AL]
    anions  = ions[  AL]
    zAs     = zs  [  AL]
    
    # Begin with Debye-Hueckel component
    Gex_nRT = fG(cf['Aosm'],I)
    
    # Add c-a interactions
    for C,cation in enumerate(cations):
        for A,anion in enumerate(anions):

            iset = [cation,anion]
            iset.sort()
            iset= ''.join(iset)
            
            Gex_nRT = Gex_nRT + cats[:,C] * anis[:,A] \
                * (2*B(cf,iset,I) + Z*CT(cf,iset,zCs[C],zAs[A]))
    
    # Add c-c' interactions
    for C0 in range(len(cations)):
        for C1 in range(C0+1,len(cations)):
            
            iset = [cations[C0],cations[C1]]
            iset.sort()
            iset= ''.join(iset)
            
            if iset in cf['theta']:
                theta = cf['theta'][iset]
            else:
                theta = 0
                print('WARNING: no theta value for ' + cations[C0] + '-' \
                      + cations[C1] + ' found; defaulting to zero')
            
            Gex_nRT = Gex_nRT + cats[:,C0] * cats[:,C1] \
                * (2 * theta)# + pz.etheta(t,zC[C0],zC[C1],I))
    
    # Add c-c'-a interactions
            for A in range(len(anions)):
                
                iset = [cations[C0],cations[C1],anions[A]]
                iset.sort()
                iset= ''.join(iset)
            
                if iset in cf['psi']:
                    psi = cf['psi'][iset]
                else:
                    psi = 0
                    print('WARNING: no psi value for ' + cations[C0] + '-' \
                          + cations[C1] + '-' + anions[A] \
                          + ' found; defaulting to zero')
                    
                Gex_nRT = Gex_nRT + cats[:,C0] * cats[:,C1] \
                    * anis[:,A] * psi
                    
    # Add a-a' interactions
    for A0 in range(len(anions)):
        for A1 in range(A0+1,len(anions)):
            
            iset = [anions[A0],anions[A1]]
            iset.sort()
            iset= ''.join(iset)
            
            if iset in cf['theta']:
                theta = cf['theta'][iset]
            else:
                theta = 0
                print('WARNING: no theta value for ' + anions[A0] + '-' \
                      + anions[A1] + ' found; defaulting to zero')
            
            Gex_nRT = Gex_nRT + anis[:,A0] * anis[:,A1] \
                * (2 * theta)# + pz.etheta(t,zC[C0],zC[C1],I))
    
    # Add c-a-a' interactions
            for C in range(len(cations)):
                
                iset = [anions[A0],anions[A1],cations[C]]
                iset.sort()
                iset= ''.join(iset)
            
                if iset in cf['psi']:
                    psi = cf['psi'][iset]
                else:
                    psi = 0
                    print('WARNING: no psi value for ' + anions[A0] + '-' \
                          + anions[A1] + '-' + cations[C] \
                          + ' found; defaulting to zero')
                    
                Gex_nRT = Gex_nRT + anis[:,A0] * anis[:,A1] \
                    * cats[:,C] * psi
    
    return Gex_nRT

# Derive activity coefficient function
ln_acfs = egrad(Gex_nRT)

##### TEST AREA ###############################################################
    
go = time.time()

T,tots,ions,idf = getIons('ions_in.csv')

cdf = getCoeffbase('coefficients.csv')
ddf = getDissocbase('dissociations.csv')

cf = getCoeffs(cdf,T)

mols = np.copy(tots)

Gex = Gex_nRT(mols,ions,cf)

acfs = np.exp(ln_acfs(mols,ions,cf))

def minifun(pH,Kw):
    
    mH = 10.**-pH
#    mOH = Kw / mH
    
    mNa = np.array([6.])
    mCl = np.array([6.])
    
    mOH = mH + mNa - mCl
    
    mols = np.vstack((mH,mOH,mNa,mCl)).transpose()
    ions = np.array(['H','OH','Na','Cl'])
    
    acfs  = ln_acfs(mols,ions,cf)
    
    aH  = np.exp(acfs[:,0])
    aOH = np.exp(acfs[:,1])
    
    DG = np.log(mH*aH * mOH*aOH) - np.log(Kw)
    
#    print(acfs)
    
    return DG, mols, ions, acfs

Kw = evalDissocs(ddf,np.array([298.15]),'H2O')
#DGDG = minifun(np.array([8.,8.,8.,8.,8.,8.,8.,8.,8.]),Kw)


sol = minimize(lambda pH:minifun(pH,Kw)[0]**2,[7.], method='Nelder-Mead')

stop = time.time()
print('Execution time: %.4f seconds' % (stop-go))
print('pH = ' + str(sol['x'][0]))

DG,mols,ions,acfs = minifun(sol['x'][0],Kw)

mH  = mols[:,0]
mOH = mols[:,1]

aH  = acfs[:,0]
aOH = acfs[:,1]

#DGDG = minifun(8.,)

print(mH*mOH * aH*aOH)
print(DG)

xpH = np.linspace(-14,14,141)
xDG = np.zeros_like(xpH)
for X,xval in enumerate(xpH):
    xDG[X] = minifun(xval,Kw)[0]
    
io.savemat('xpHDG.mat',{'pH':xpH, 'DG':xDG})

