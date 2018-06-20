import autograd.numpy as np
import pandas as pd
from autograd import elementwise_grad as egrad
import time

##### FILE I/O ################################################################

def getIons(filename):
  
    # Import input conditions from .csv
    idf = pd.read_csv(filename)
        
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

alist = ['a'+str(n) for n in range(8)]
ilist = ['ion'+str(n) for n in range(1,4)]

def getCoeffbase(filename):
    
    # Import .csv file of coefficient values
    cdf = pd.read_csv(filename, header=1)
    
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

def evalCoeffs(T,a): # M88 Eq. (II-13) = GM89 Eq. (3)
    
    return a[0] + a[1]*T + a[2]/T + a[3]*np.log(T) + a[4]/(T-263.) \
        + a[5]*T**2 + a[6]/(680.-T) + a[7]/(T-227.)

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

##### IONIC CHARGES ###########################################################

def getCharges(ions):

    z = {}
    
    z['Na'] = np.float_(+1)
    z['K' ] = np.float_(+1)
    z['Ca'] = np.float_(+2)
    
    z['Cl'] = np.float_(-1)
    
    return np.array([z[ion] for ion in ions])

##### DEBYE-HUECKEL SLOPE #####################################################

def fG(Aosm,I): # CRP94 Eq. (AI1)
    
    return -4 * Aosm * I * np.log(1 + b*np.sqrt(I)) / b

##### PITZER MODEL SUBFUNCTIONS ###############################################

def g(x): # CRP94 Eq. (AI13)
    return 2 * (1 - (1 + x) * np.exp(-x)) / x**2

def B(cf,iset,I): # CRP94 Eq. (AI8)
    B = cf['b0'][iset] + cf['b1'][iset] * g(cf['a1'][iset]*np.sqrt(I))
    if iset in cf['b2']:
        B = B + cf['b2'][iset] * g(cf['a2'][iset]*np.sqrt(I))
    return B

def CT(cf,iset,z1,z2): # P91 Ch. 3 Eq. (53)
    return cf['Cphi'][iset] / (2 * np.sqrt(np.abs(z1*z2)))

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
    
    return Gex_nRT

# Derive activity coefficient function
ln_acfs = egrad(Gex_nRT)

##### TEST AREA ###############################################################
    
go = time.time()

T,tots,ions,idf = getIons('ions_in.csv')

cdf = getCoeffbase('coefficients.csv')

cf = getCoeffs(cdf,T)

Gex = Gex_nRT(tots,ions,cf)

acfs = np.exp(ln_acfs(tots,ions,cf))

stop = time.time()
print('Execution time: %.4f seconds' % (stop-go))
