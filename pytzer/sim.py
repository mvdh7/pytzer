from autograd import numpy as np
from autograd import elementwise_grad as egrad
import pandas as pd
from . import data, model, prop, tconv
    
# Propagate bs & fpd uncertainties
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
def fpd(fpdbase,ele,src,cf,err_cfs_both,fpd_sys_std):
    
    SL = np.logical_and(fpdbase.ele == ele,fpdbase.src == src)
    
    tot = pd2vs(fpdbase.m)
    mw = np.float_(1)
    bs = np.vstack(fpdbase.ele.map(prop.solubility25).values)
    ms = tot * mw / (bs - tot)
    
    ions = data.ele2ions(pd.Series([ele]))[0]
    
    osm_sim = pd2vs(fpdbase.osm_calc[SL]) \
                       + np.random.normal(scale=fpd_sys_std[ele][src],
                                          size=(sum(SL),1)) \
                       + err_cfs_both[ele][src][0] \
                       * dosm25_dbs (bs[SL],ms[SL],mw,
                                     pd2vs(fpdbase.fpd[SL]),
                                     pd2vs(fpdbase.nC[SL]),
                                     pd2vs(fpdbase.nA[SL]),
                                     ions,
                                     pd2vs(fpdbase.t[SL]),
                                     pd2vs(fpdbase.t25[SL]),
                                     pd2vs(fpdbase.t25[SL]),
                                     cf) \
                       + err_cfs_both[ele][src][1] \
                       * dosm25_dfpd(bs[SL],ms[SL],mw,
                                     pd2vs(fpdbase.fpd[SL]),
                                     pd2vs(fpdbase.nC[SL]),
                                     pd2vs(fpdbase.nA[SL]),
                                     ions,
                                     pd2vs(fpdbase.t[SL]),
                                     pd2vs(fpdbase.t25[SL]),
                                     pd2vs(fpdbase.t25[SL]),
                                     cf)
    
    return osm_sim
