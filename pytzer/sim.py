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
def fpd(ele,tot,srcs,bs,osm,err_cfs_both,fpd_sys_std,
         fpd,nC,nA,ions,T0,T1,TR,cf):
    
    mw = np.float_(1)
    ms = tot * mw / (bs - tot)
    
    dbs  = dosm25_dbs (bs,ms,mw,fpd,nC,nA,ions,T0,T1,TR,cf)
    dfpd = dosm25_dfpd(bs,ms,mw,fpd,nC,nA,ions,T0,T1,TR,cf)
    
    osm_fpd = np.full_like(tot,np.nan)
    
    for src in fpd_sys_std[ele].keys():
    
        SL = srcs == src
        
        osm_fpd[SL] = osm[SL] \
                    + np.random.normal(scale=fpd_sys_std[ele][src],
                                       size=sum(SL)) \
                    + err_cfs_both[ele][src][0] * dbs [SL].ravel() \
                    + err_cfs_both[ele][src][1] * dfpd[SL].ravel()
    
    return osm_fpd
