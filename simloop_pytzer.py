from autograd import numpy as np
from sys import path#, argv
if 'E:\Dropbox\_UEA_MPH\pitzer-spritzer\python' not in path:
    path.append('E:\\Dropbox\\_UEA_MPH\\pitzer-spritzer\\python')
import pickle, pweb, time
from multiprocessing import Pool
import pytzer as pz

# Get electrolyte to analyse and number of repeats from user input
#ele   =     argv[1]
#Ureps = int(argv[2])
ele = 'NaCl'
Ureps = 100

# Get electrolyte-specific information
Ufcs = {'NaCl' : 'b0b1C0C1',
        'KCl'  : 'b0b1C0'  }

Uaos = {'NaCl' : np.float_([-9, 2,-9,-9, 2.5]),
        'KCl'  : np.float_([-9, 2,-9,-9,-9  ])}

Efc = Ufcs[ele]
Eao = Uaos[ele]

# Load FPD dataset
with open('E:\Dropbox\_UEA_MPH\pitzer-spritzer\python\pickles\simpar_fpd.pkl',
          'rb') as f:
    fpdbase,fpdp,D_bs_fpd,fpd_sys_std = pickle.load(f)

# Preprocess dataset
L = fpdbase.ele == ele

Eargs = [fpdbase.loc[L,var] for var in ['m', 'nu', 'zC', 'zA', 'nC', 'nA',
                                        'osm25_calc', 'src', 'ele', 'fpd',
                                        't25']]
Em,Enu,EzC,EzA,EnC,EnA,Eosm25_calc,Esrc,Eele,Efpd,Et25 = Eargs

# Get pytzer function inputs
EmCmA = np.tile(Em.values,(2,1)).transpose()
zC = np.float_(+1)
zA = np.float_(-1)
T = np.full_like(Em,298.15, dtype='float64')
alph1 = Eao[1]
alph2 = Eao[2]
omega = Eao[4]
nC = np.float_(1)
nA = np.float_(1)

# Define optimisation function
def Eopt(rseed=None):

    # Seed random numbers
    np.random.seed(rseed)

    # Simulate uncertainties
    Uosm = pweb.simpar.fpd(fpdp,Eele,Em,Efpd,Enu,Esrc,D_bs_fpd,fpd_sys_std)

    b0,b1,b2,C0,C1,bCmx,mse \
        = pz.fitting.bC(EmCmA,zC,zA,T,alph1,alph2,omega,nC,nA,Uosm,Efc,'osm')

    return b0,b1,b2,C0,C1

if __name__ == '__main__':

    # Set initial random seed (for reproducibility)
    np.random.seed(295)

    # Generate seeds for random number generator
    rseeds = np.random.randint(0,2**32,size=Ureps,dtype='int64')

    Xtstart = time.time() # begin timer - multiprocessing

    with Pool() as pool:
        bCpool = pool.map(Eopt,rseeds)
        pool.close()
        pool.join()

    # Format pool output
    bCpool = np.array([bCpool[X] for X in range(Ureps)])
    bCpool_cv = np.cov(bCpool,rowvar=False)

    Xtend = time.time() # end timer - multiprocessing

    # Calculate and print processing time
    print('multiprocessing %s: %d reps in %.2f seconds' \
        % (ele,Ureps,(Xtend - Xtstart)))

    # Pickle results
    with open('E:\Dropbox\_UEA_MPH\pytzer\pickles' \
              + '\simloop_pytzer_bC_' + ele + '.pkl','wb') as f:
        pickle.dump((bCpool_cv,ele,Ureps),f)
