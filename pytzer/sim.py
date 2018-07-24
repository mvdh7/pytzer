from autograd import numpy as np

def fpd(tot,fpd_calc,srcs,ele,fpderr_rdm,fpderr_sys):
    
    fpd = fpd_calc
    
    for src in fpderr_rdm[ele].keys():
        
        SL = src == srcs
        
        fpd[SL] = fpd[SL] + np.random.normal(size=1,loc=0,
                            scale=np.abs(fpderr_sys[ele][src][0]) \
                            * np.sqrt(np.pi/2)) \
                          + np.random.normal(size=1,loc=0,
                            scale=np.abs(fpderr_sys[ele][src][1]) \
                            * np.sqrt(np.pi/2)) \
                          * tot[SL] \
                          + np.random.normal(size=sum(SL),loc=0,
                            scale=(fpderr_rdm[ele][src][0] \
                                 + fpderr_rdm[ele][src][1] * tot[SL]) \
                                 * np.sqrt(np.pi/2))
    
    return fpd

# These old methods were for analyses based on uncertainty propagation
#  through to the osmotic coefficient, rather than perturbing the actual FPD
#  measurements directly.
#    
## For propagating bs & fpd uncertainties
#def fpd2osm25(bs,ms,mw,fpd,nC,nA,ions,T0,T1,TR,cf):
#    
#    tot = bs * ms / (ms + mw)
#    
#    mols = np.vstack([tot.ravel() * nC,
#                      tot.ravel() * nA]).transpose()
#    
#    osmT0 = tconv.fpd2osm(mols,fpd)
#    osm25 = tconv.osm2osm(tot,nC,nA,ions,T0,T1,TR,cf,osmT0)
#    
#    return osm25
#
#dosm25_dbs  = egrad(fpd2osm25)
#dosm25_dfpd = egrad(fpd2osm25, argnum=3)
#
## ... and without temperature conversion
#def fpd2osmT0(bs,ms,mw,fpd,nC,nA):
#    
#    tot = bs * ms / (ms + mw)
#    
#    mols = np.vstack([tot.ravel() * nC,
#                      tot.ravel() * nA]).transpose()
#    
#    osmT0 = tconv.fpd2osm(mols,fpd)
#    
#    return osmT0
#
#dosmT0_dbs  = egrad(fpd2osmT0)
#dosmT0_dfpd = egrad(fpd2osmT0, argnum=3)
#
## Simulate new fpd dataset for one electrolyte
## !!!!! CURRENTLY DOESN'T RANDOMLY DEVIATE FOLLOWING SYS ERRORS !!!!!
#def fpd(ele,tot,srcs,bs,osm,err_cfs_both,fpd_sys_std,
#         fpd,nC,nA,ions,T0,T1,TR,cf):
#    
#    mw = np.float_(1)
#    ms = tot * mw / (bs - tot)
#    
#    dbs  = dosm25_dbs (bs,ms,mw,fpd,nC,nA,ions,T0,T1,TR,cf)
#    dfpd = dosm25_dfpd(bs,ms,mw,fpd,nC,nA,ions,T0,T1,TR,cf)
#    
#    osm_fpd = np.full_like(tot,np.nan)
#    
#    for src in fpd_sys_std[ele].keys():
#    
#        SL = srcs == src
#        
#        osm_fpd[SL] = osm[SL] \
#                    + np.random.normal(scale=fpd_sys_std[ele][src],
#                                       size=sum(SL)) \
#                    + err_cfs_both[ele][src][0] * dbs [SL].ravel() \
#                    + err_cfs_both[ele][src][1] * dfpd[SL].ravel()
#    
#    return osm_fpd
#
#def fpdT0(ele,tot,srcs,bs,osm,err_cfs_both,fpd_sys_std,fpd,nC,nA):
#    
#    mw = np.float_(1)
#    ms = tot * mw / (bs - tot)
#    
#    dbs  = dosmT0_dbs (bs,ms,mw,fpd,nC,nA)
#    dfpd = dosmT0_dfpd(bs,ms,mw,fpd,nC,nA)
#    
#    osm_fpd = np.full_like(tot,np.nan)
#    
#    for src in fpd_sys_std[ele].keys():
#    
#        SL = srcs == src
#        
#        osm_fpd[SL] = osm[SL] \
#                    + np.random.normal(scale=fpd_sys_std[ele][src],
#                                       size=sum(SL)) \
#                    + err_cfs_both[ele][src][0] * dbs [SL].ravel() \
#                    + err_cfs_both[ele][src][1] * dfpd[SL].ravel()
#    
#    return osm_fpd
