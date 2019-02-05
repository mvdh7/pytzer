from pandas import read_excel
from numpy import array, concatenate, float_, logical_not, mean, ones_like, \
                  pi, size, sqrt, std, unique

from numpy import abs as np_abs
from numpy import max as np_max
from numpy import sum as np_sum

import pytzer as pz
from scipy.special import factorial

from copy import deepcopy

cf = deepcopy(pz.cfdicts.MarChemSpec25)

cf.ions = concatenate([cf.ions, array(['Li','I','Rb','Cs'])])
cf.add_zeros(cf.ions)

cf.bC['Li-Cl'] = pz.coeffs.bC_Li_Cl_HM83
cf.bC['Cs-Cl'] = pz.coeffs.bC_Cs_Cl_HM83
    
e2i = {'NaOH': [array(['Na','OH']), float_([1,1])],
                  
       'Zn(NO3)2': [array(['Znjj','NO3']), float_([1,2])],
       
       'H2SO4' : [array(['H' ,'SO4']), float_([2,1])],
       'Na2SO4': [array(['Na','SO4']), float_([2,1])],
       'MgSO4' : [array(['Mg','SO4']), float_([1,1])],
       'CuSO4' : [array(['Cu','SO4']), float_([1,1])],
       
       'LiCl' : [array(['Li','Cl']), float_([1,1])],
       'NaCl' : [array(['Na','Cl']), float_([1,1])],
       'MgCl2': [array(['Mg','Cl']), float_([1,2])],
       'KCl'  : [array(['K' ,'Cl']), float_([1,1])],
       'CaCl2': [array(['Ca','Cl']), float_([1,2])],
       'CuCl2': [array(['Cu','Cl']), float_([1,2])],
       'RbCl' : [array(['Rb','Cl']), float_([1,1])],
       'CsCl' : [array(['Cs','Cl']), float_([1,1])],
       'LaCl3': [array(['La','Cl']), float_([1,3])],
       
       'Mg(ClO4)2': [array(['Mg'  ,'ClO4']), float_([1,2])],
       'Zn(ClO4)2': [array(['Znjj','ClO4']), float_([1,2])],
       
       'LiI': [array(['Li','I']), float_([1,1])],
       
       'tris'       : [array(['tris'       ]), float_([1  ])],
       '(trisH)2SO4': [array(['trisH','SO4']), float_([2,1])],
       'trisHCl'    : [array(['trisH','Cl' ]), float_([1,1])],
       
       'glycerol': [array(['glycerol']), float_([1])],
       'sucrose' : [array(['sucrose' ]), float_([1])],
       'urea'    : [array(['urea'    ]), float_([1])]}

filename = '../pytzer-support/datasets/isonew.xlsx'

isonew = read_excel(filename, skiprows=2)

isobase = isonew[['src','temp']]

isodict = {}

for r in range(len(isonew.index)):

    irow = isonew.iloc[r,6:]
    
    isodict[r] = {}
    
    for i, icell in enumerate(irow):
    
        if type(icell) is str:
            
            isodict[r][icell] = {}
            irc = isodict[r][icell]
            
            celes = icell.split('-')
            
            ctots = irow[i-len(celes):i].values.astype('float64')
                        
            irc_ions = concatenate([e2i[cele][0] for cele in celes])
                            
            irc_mols = concatenate([e2i[cele][1] * ctots[c] \
                for c, cele in enumerate(celes)])
            
            irc['ions'] = unique(irc_ions)
            
            irc['mols'] = array([[np_sum(irc_mols[irc_ions == ion]) \
                for ion in irc['ions']]])
            
            irc['T'] = float_([[isonew.temp[r]]])
    
#            irc['osm'] = pz.model.osm(irc['mols'],irc['ions'],irc['T'],cf)
#            
#            irc['aw'] = pz.model.osm2aw(irc['mols'],irc['osm'])
#
## Get all water activities            
#all_aw = [concatenate(
#    [isodict[irow][icell]['aw'] for icell in isodict[irow].keys()]).ravel() \
#     for irow in isodict.keys()]
#
#aw_mean = array([mean(aw) for aw in all_aw])
#aw_sd   = array([std (aw) for aw in all_aw])
#ncups   = array([size(aw) for aw in all_aw])
#
#def std_unbias(stds,nobs):
#
#    def keven(k2):
#        
#        k = k2/2
#        
#        return sqrt(2 / (pi * (2*k - 1))) * 2**(2*k - 2) \
#            * factorial(k - 1)**2 / factorial(2*k - 2)
#    
#    def kodd(k2p1):
#        
#        k = (k2p1 - 1)/2
#        
#        return sqrt(pi / k) * factorial(2*k - 1) \
#            / (2**(2*k - 1) * factorial(k - 1)**2)
#            
#    c4 = ones_like(nobs, dtype='float64')
#    
#    Leven = nobs % 2 == 0
#    Lodd  = logical_not(Leven)
#    
#    c4[Leven] = keven(nobs[Leven])
#    c4[Lodd ] = kodd (nobs[Lodd ])
#    
#    return stds / c4
#            
#aw_sdu = std_unbias(aw_sd,ncups)
#
##%%
#from matplotlib import pyplot as plt
#
#plt.scatter(range(len(aw_sdu)),aw_sdu)
##plt.scatter(aw_mean,aw_sdu)
#
#sdu_max = np_abs(np_max(aw_sdu)) * 1.1
#
#plt.ylim([0,sdu_max])
#
