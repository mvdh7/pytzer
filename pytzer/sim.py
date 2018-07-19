from autograd import numpy as np
from autograd import elementwise_grad as egrad
import pandas as pd
from . import data, tconv
    
# For propagating bs & fpd uncertainties
def fpd2osm25(bs,ms,mw,fpd,nC,nA,ions,T0,T1,TR,cf):
    
    tot = bs * ms / (ms + mw)
    
    mols = np.vstack([tot.ravel() * nC,
                      tot.ravel() * nA]).transpose()
    
    osmT0 = tconv.fpd2osm(mols,fpd)
    osm25 = tconv.osm2osm(tot,nC,nA,ions,T0,T1,TR,cf,osmT0)
    
    return osm25

dosm25_dbs  = egrad(fpd2osm25)
dosm25_dfpd = egrad(fpd2osm25, argnum=3)

# Simulate new fpd dataset for one electrolyte
def fpd(tot,bs,fpd,nC,nA,T0,T1,TR,osm,err_cfs_both,fpd_sys_std,ele,src,cf):
    
    mw = np.float_(1)
    ms = tot * mw / (bs - tot)
    
    ions = data.ele2ions(pd.Series([ele]))[0]
    
    osm_fpd = osm + np.random.normal(scale=fpd_sys_std[ele][src],
                                          size=np.shape(tot)) \
                  + err_cfs_both[ele][src][0] \
                  * dosm25_dbs (bs,ms,mw,fpd,nC,nA,ions,T0,T1,TR,cf) \
                  + err_cfs_both[ele][src][1] \
                  * dosm25_dfpd(bs,ms,mw,fpd,nC,nA,ions,T0,T1,TR,cf)
    
    return osm_fpd
