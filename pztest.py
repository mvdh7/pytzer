#from copy import deepcopy
import pytzer as pz
#import numpy as np 

# Import test dataset
filename = 'testfiles/pytzerPitzer.csv'

pz.blackbox(filename)

#mols,ions,T = pz.io.getmols(filename)

### Change temperature
##TEMP = np.float_(273.15)
##T[:] = TEMP # just in case
#
## Set up CoefficientDictionary
#cf = deepcopy(pz.cfdicts.MarChemSpec)
#cf.add_zeros(ions) # just in case
#
### Print out coefficients at 298.15 K
##cf.print_coeffs(TEMP,'print_coeffs/' + cf.name + '.txt')
#
## Separate out zero ionic strengths
#zs = pz.props.charges(ions)[0]
#I = np.vstack(0.5 * (np.sum(mols * zs**2, 1)))
#
#Gex_nRT = np.full_like(T   , np.nan)
#osm     = np.full_like(T   , np.nan)
#acfs    = np.full_like(mols, np.nan)
#
#L = (I > 0).ravel()
#
#nargsL  = (mols[ L,:], ions, T[ L], cf)
#nargsLx = (mols[~L,:], ions, T[~L], cf)
#
## Do calculations
#print('Calculating excess Gibbs energies...')
#Gex_nRT[ L] = pz.model.Gex_nRT(*nargsL)
#if np.any(~L):
#    Gex_nRT[~L] = pz.model.Gex_nRT(*nargsLx, Izero=True)
#
#print('Calculating osmotic coefficients...')
#osm[ L] = pz.model.osm(*nargsL)
#if np.any(~L):
#    osm[~L] = pz.model.osm(*nargsLx, Izero=True)
#
#print('Calculating water activity...')
#aw = pz.model.osm2aw(mols,osm)
#
#print('Calculating activity coefficients...')
#acfs[ L,:] = pz.model.acfs(*nargsL)
#if np.any(~L):
#    acfs[~L,:] = pz.model.acfs(*nargsLx, Izero=True)
#
#### Loop it
###for i in range(len(T)):
###    
###    mols_i = np.array([mols[i,:]])
###    T_i    = np.array([T[i]])
###    
###    nargs_i = (mols_i, ions, T_i, cf)
###    
###    Gex_nRT[i]   = pz.model.Gex_nRT(*nargs_i)
###    osm    [i]   = pz.model.osm    (*nargs_i)
###    acfs   [i,:] = pz.model.acfs   (*nargs_i)
##    
### Save results for plotting in MATLAB
##from scipy.io import savemat
##savemat('testfiles/threeway/temptestUpVec.mat',
##        {'mols'   : mols   ,
##         'ions'   : ions   ,
##         'T'      : T      ,
###         'Gex_nRT': Gex_nRT,
##         'acfs'   : acfs   ,
##         'osm'    : osm    ,
###         'aw'     : aw     ,
##         'I'      : I      })
