from autograd import numpy as np
import pytzer as pz

cf = pz.cdicts.M88
cf.dh['AH'] = pz.coeffs.AH_MPH

ions = np.array(['Na','Cl'])

nC = np.float_(1)
nA = np.float_(1)

tot = np.vstack([1.5,1.6,1.7])
mols = np.concatenate((tot,tot), axis=1)

Ts = [            298.15         ,
      np.float_(  298.15        ),
      np.float_([ 298.15       ]),
      np.float_([[298.15      ]]),
      np.float_([ 298.15,298.15,298.15]),
      np.vstack([ 298.15,298.15,298.15])]

ys = [pz.tconv.y(T,T+1) for T in Ts]
zs = [pz.tconv.z(T,T+1) for T in Ts]
Os = [pz.tconv.O(T,T+1) for T in Ts]

Lapps = [pz.model.Lapp(tot,nC,nA,ions,T,cf) for T in Ts]

L1s = [pz.tconv.L1(tot,nC,nA,ions,T,cf) for T in Ts]
L2s = [pz.tconv.L2(tot,nC,nA,ions,T,cf) for T in Ts]

fG = [pz.model.fG(T,tot,cf) for T in Ts]

Gex = [pz.model.Gex_nRT(mols,ions,T,cf) for T in Ts]
dGex_T_dT = [pz.model.dGex_T_dT(mols,ions,T,cf) for T in Ts]

dGex_T_dT_dir = [( pz.model.Gex_nRT(mols,ions,T+1e-8,cf) \
                 - pz.model.Gex_nRT(mols,ions,T,cf))*1e8 for T in Ts]

acfs = [pz.model.acfs(mols,ions,T,cf) for T in Ts]
osms =  pz.model.osm(mols,ions,Ts[5],cf)

ww = np.full_like(Ts[5],1, dtype='float64')
    
osmD = pz.model.osmD(ww,mols,ions,Ts[5],cf)
osmx = 1 - pz.model.osmD(ww,mols,ions,Ts[5],cf) \
    / (pz.constants.R * Ts[5] * np.vstack(np.sum(mols,axis=1)))
