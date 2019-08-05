from copy import deepcopy
import numpy as np
import pytzer as pz

mols, ions, tempK, pres = pz.io.getmols('testfiles/miami.csv')
zs = pz.properties.charges(ions)[0]
mols[ions == 'OH'][0] = np.array([[np.sum(mols[:, 1]*zs)]])
prmlib = deepcopy(pz.libraries.MIAMI)
prmlib.add_zeros(ions)
#prmlib.bC['H-HSO4'] = pz.parameters.bC_none
acfs = pz.model.acfs(mols, ions, tempK, pres, prmlib=prmlib)
L = 1
for i, ion in enumerate(ions):
    print('{:>6}: {:.4e} {:.4f}'.format(ion, mols[i][L], acfs[i][L]))

#%% Copy in thetas from Pierrot_2018_Interaction_Model.xlsm
PMcats = ['', 'Sr', 'Na', 'K', 'Mg', 'Ca', 'H', 'Li', 'Rb', 'Cs', 'NH4', 'Ba',
    'Mn']
mjCaMax = 5
T = np.pi*100
P = 10.10325
Tcc = np.zeros((13, 13))
Tcc[1, 2] = 0.07
Tcc[1, 3] = 0.07
Tcc[2, 3] = -0.0502312111 + 14.0213141/T
Tcc[2, 4] = 0.07
Tcc[2, 5] = 0.05
Tcc[3, 4] = 0
Tcc[3, 5] = 0.1156
Tcc[4, 5] = 0.007
Tcc[mjCaMax + 1, 1] = 0.0591 + 0.00045 * (T - 298.15)
Tcc[mjCaMax + 1, 2] = 0.03416 - 0.000209 * (T - 273.15)
Tcc[mjCaMax + 1, 3] = 0.005 - 0.0002275 * (T - 298.15)
Tcc[mjCaMax + 1, 4] = 0.062 + 0.0003275 * (T - 298.15)
Tcc[mjCaMax + 1, 5] = 0.0612 + 0.0003275 * (T - 298.15)
Tcc[mjCaMax + 4, 2] = -0.033
Tcc[mjCaMax + 4, 3] = 0
Tcc[mjCaMax + 6, 2] = 0.067
Tcc[mjCaMax + 6, 3] = 0.01
Tcc[mjCaMax + 7, 2] = 0.082
Tcc = (Tcc + np.transpose(Tcc))
Tcc_compare = {}
for c0, cat0 in enumerate(PMcats):
    for c1, cat1 in enumerate(PMcats):
        cc = [cat0, cat1]
        cc.sort()
        ccstr = '{}-{}'.format(*cc)
        if ccstr in prmlib.theta.keys() and ccstr not in Tcc_compare.keys():
            valPM = Tcc[c0, c1]
            valPytz = prmlib.theta[ccstr](T, P)[0]
            Tcc_compare[ccstr] = [valPytz-valPM, valPM, valPytz]
PManis = ['', 'Cl', 'SO4', 'CO3', 'HCO3', 'Br', 'F', 'BOH4', 'HSO4', 'HS',
    'OH']
mjAnMax=7
Taa = np.zeros((13, 13))
Taa[1, 2] = 0.07
Taa[1, 3] = -0.053
Taa[1, 4] = 0.0359
Taa[1, 6] = 0.01
Taa[1, 7] = -0.0323 - 0.000042333*(T - 298.15) - 0.000021926*(T - 298.25)**2
Taa[2, 7] = -0.012
Taa[2, 3] = 0.02
Taa[2, 4] = 0.01
Taa[3, 4] = 0
Taa[mjAnMax + 1, 1] = -0.006
Taa[mjAnMax + 1, 2] = 0
Taa[mjAnMax + 3, 1] = (-0.05 + 0.0003125*(T - 298.15) -
    0.000008362*(T - 298.15)**2)
Taa[mjAnMax + 3, 2] = -0.013
Taa[mjAnMax + 3, 5] = -0.065
Taa = (Taa + np.transpose(Taa))
Taa_compare = {}
for a0, ani0 in enumerate(PManis):
    for a1, ani1 in enumerate(PManis):
        aa = [ani0, ani1]
        aa.sort()
        aastr = '{}-{}'.format(*aa)
        if aastr in prmlib.theta.keys() and aastr not in Taa_compare.keys():
            valPM = Taa[a0, a1]
            valPytz = prmlib.theta[aastr](T, P)[0]
            Taa_compare[aastr] = [valPytz-valPM, valPM, valPytz]
Paac = np.zeros((20, 20, 20))
Paac[1, 2, 2] = -0.009
Paac[1, 2, 3] = -0.212481 + 0.000284698333*T + 37.5619614/T
Paac[1, 2, 4] = -0.004
Paac[1, 2, 5] = -0.018
Paac[1, 3, 2] = 0.016
Paac[1, 4, 2] = -0.0143
Paac[1, 4, 4] = -0.096
Paac[1, 6, 2] = 0.0023
Paac[1, 7, 2] = -0.0132
Paac[1, 7, 4] = -0.235
Paac[1, 7, 5] = -0.8
Paac[2, 3, 2] = -0.005
Paac[2, 3, 3] = -0.009
Paac[2, 4, 2] = -0.005
Paac[2, 4, 4] = -0.161
Paac[3, 4, 2] = 0
Paac[3, 4, 3] = 0
Paac[mjAnMax + 1, 1, 2] = -0.006
Paac[mjAnMax + 1, 2, 2] = 0
Paac[mjAnMax + 1, 2, 3] = -0.0677
Paac[mjAnMax + 3, 1, 2] = -0.006
Paac[mjAnMax + 3, 1, 3] = -0.006
Paac[mjAnMax + 3, 1, 5] = -0.025
Paac[mjAnMax + 3, 2, 2] = -0.009
Paac[mjAnMax + 3, 2, 3] = -0.05
Paac[mjAnMax + 3, 5, 2] = -0.018
Paac[mjAnMax + 3, 5, 3] = -0.014
Paac_compare = {}
for a0, ani0 in enumerate(PManis):
    for a1, ani1 in enumerate(PManis):
        aa = [ani0, ani1]
        aa.sort()
        for c, cat in enumerate(PMcats):
            caastr = '{}-{}-{}'.format(cat, *aa)
            if (caastr in prmlib.psi.keys() and
                    caastr not in Paac_compare.keys()):
                valPM = Paac[a0, a1, c] + Paac[a1, a0, c]
                valPytz = prmlib.psi[caastr](T, P)[0]
                if valPytz-valPM != 0:
                    Paac_compare[caastr] = [valPytz-valPM, valPM, valPytz]
Pcca = np.zeros((20, 20, 20))
Pcca[1, 2, 1] = -0.015
Pcca[1, 3, 1] = -0.015
Pcca[2, 3, 1] = 0.0134211308 - 5.10212917 / T
Pcca[2, 3, 2] = 0.0348115174 - 8.21656777 / T
Pcca[2, 3, 5] = -0.0022
Pcca[2, 4, 1] = 0.0199 - 9.51 / T
Pcca[2, 4, 2] = -0.015
Pcca[2, 5, 1] = -0.003
Pcca[2, 5, 2] = -0.012
Pcca[3, 4, 1] = 0.02586 - 14.27 / T
Pcca[3, 4, 2] = -0.048
Pcca[3, 5, 1] = 0.047627877 - 27.0770507 / T
Pcca[3, 5, 2] = 0
Pcca[4, 5, 1] = -0.012
Pcca[4, 5, 2] = 0.024
Pcca[mjCaMax + 1, 1, 1] = 0.0054 - 0.00021 * (T - 298.15)
Pcca[mjCaMax + 1, 2, 1] = 0.0002
Pcca[mjCaMax + 1, 2, 2] = 0
Pcca[mjCaMax + 1, 2, 5] = -0.012
Pcca[mjCaMax + 1, 3, 1] = -0.011
Pcca[mjCaMax + 1, 3, 2] = 0.197
Pcca[mjCaMax + 1, 3, 5] = -0.021
Pcca[mjCaMax + 1, 4, 1] = 0.001 - 0.0007325 * (T - 298.15)
Pcca[mjCaMax + 1, 4, 5] = -0.005
Pcca[mjCaMax + 1, 5, 1] = 0.0008 - 0.000725 * (T - 298.15)
Pcca[mjCaMax + 4, 2, 1] = -0.003
Pcca[mjCaMax + 4, 3, 1] = -0.0013
Pcca[mjCaMax + 6, 2, 1] = -0.012
Pcca[mjCaMax + 6, 3, 1] = -0.017
Pcca[mjCaMax + 7, 2, 1] = -0.0174
Pcca_compare = {}
for c0, cat0 in enumerate(PMcats):
    for c1, cat1 in enumerate(PMcats):
        cc = [cat0, cat1]
        cc.sort()
        for a, ani in enumerate(PManis):
            ccastr = '{}-{}-{}'.format(*cc, ani)
            if (ccastr in prmlib.psi.keys() and
                    ccastr not in Pcca_compare.keys()):
                valPM = Pcca[c0, c1, a] + Pcca[c1, c0, a]
                valPytz = prmlib.psi[ccastr](T, P)[0]
                if valPytz-valPM != 0:
                    Pcca_compare[ccastr] = [valPytz-valPM, valPM, valPytz]