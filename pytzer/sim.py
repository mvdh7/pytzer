from autograd import numpy as np

##### VAPOUR PRESSURE LOWERING ################################################

def vpl(tot,osm_calc,srcs,ele,vplerr_rdm,vplerr_sys):

    osm = np.copy(osm_calc)

    for S,src in enumerate(list(vplerr_rdm[ele].keys())[:-2]):

        SL = src == srcs

#        # Approach 1
#        osm[SL] = osm[SL] + np.random.normal(size=1,loc=0,
#                            scale=np.abs(vplerr_sys[ele][src][0]) \
#                            * np.sqrt(2/np.pi)) \
#                          + np.random.normal(size=1,loc=0,
#                            scale=np.abs(vplerr_sys[ele][src][1]) \
#                            * np.sqrt(2/np.pi)) \
#                          * tot[SL] \
#                          + np.random.normal(size=sum(SL),loc=0,
#                            scale=(vplerr_rdm[ele][src][0] \
#                                 + vplerr_rdm[ele][src][1] \
#                                 * np.exp(-tot[SL])) \
#                                 * np.sqrt(2/np.pi))

#        # Approach 2
#        osm[SL] = osm[SL] + np.random.normal(size=1,loc=0,
#                            scale=np.abs(vplerr_sys[ele][src][0]) \
#                            * np.sqrt(2/np.pi)) / tot[SL] \
#                          + np.random.normal(size=sum(SL),loc=0,
#                            scale=(vplerr_rdm[ele][src][0] \
#                                 + vplerr_rdm[ele][src][1] \
#                                 * np.exp(-tot[SL])) \
#                                 * np.sqrt(2/np.pi))

        # Approach 3
        all_sys = np.concatenate([vplerr_sys[ele]['all_int'] \
                                  for ele in ('NaCl','KCl','CaCl2')])
        RL = all_sys != 0
        rmsd = np.sqrt(np.mean(all_sys[RL]**2))
        osm[SL] = osm[SL] + np.random.normal(size=1,loc=0,scale=rmsd) / tot[SL] \
                          + np.random.normal(size=sum(SL),loc=0,
                                             scale=(vplerr_rdm[ele][src][0] \
                                                  + vplerr_rdm[ele][src][1] \
                                                  * np.exp(-tot[SL])) \
                                                  * np.sqrt(np.pi/2))

    return osm

##### FREEZING POINT DEPRESSION ###############################################

def fpd_osm25(tot,osm25_calc,srcs,ele,fpderr_rdm,fpderr_sys):
    
    osm25 = np.copy(osm25_calc)
    
    for S,src in enumerate(np.unique(srcs)):

        SL = src == srcs
        
        osm25[SL] = osm25[SL] + np.random.normal(loc=0,
            scale=fpderr_sys['all_rmse']) \
                + np.random.normal(size=sum(SL),loc=0,
                    scale=(fpderr_rdm[ele][src][0] \
                        + fpderr_rdm[ele][src][1] \
                        * np.exp(-tot[SL] * fpderr_rdm[ele][src][2])) \
                        * np.sqrt(2/np.pi))   

    return osm25


def fpd(tot,fpd_calc,srcs,ele,fpderr_rdm,fpderr_sys):

    fpd = np.copy(fpd_calc)

#    # Preparation for Approach 2
#    # Simulate systematic errors for the datasets
#    syserr = np.random.normal(size=len(fpderr_rdm[ele].keys())-2,
#                              scale=0.041, # ADJUSTABLE!
#                              loc=0)
#    # Arrange the simulated errors in ascending order of magnitude
#    simorder = np.argsort(np.abs(syserr))
#    syserr = syserr[simorder]
#
#    # Determine the order of magnitudes of the real systematic errors
#    sysorder = np.argsort(np.abs(fpderr_sys[ele]['all_grad']))

    for S,src in enumerate(list(fpderr_rdm[ele].keys())[:-2]):

        SL = src == srcs

#        # Approach 1: Assume systematic offset in real dataset represents the
#        #             mean of the half-normal distribution for that dataset.
#        #             Probably overestimates uncertainty.
#        fpd[SL] = fpd[SL] + np.random.normal(size=1,loc=0,
#                            scale=np.abs(fpderr_sys[ele][src][0]) \
#                            * np.sqrt(2/np.pi)) \
#                          + np.random.normal(size=1,loc=0,
#                            scale=np.abs(fpderr_sys[ele][src][1]) \
#                            * np.sqrt(2/np.pi)) \
#                          * tot[SL] \
#                          + np.random.normal(size=sum(SL),loc=0,
#                            scale=(fpderr_rdm[ele][src][0] \
#                                 + fpderr_rdm[ele][src][1] * tot[SL]) \
#                                 * np.sqrt(2/np.pi))

#        # Approach 2: Determine systematic offset distribution from all FPD
#        #             datasets for all electrolytes. Simulate new offsets from
#        #             that distribution. Sort the simulated offsets so that
#        #             datasets with smaller systematic offsets in reality are
#        #             simulated with smaller systematic offsets too.
#        fpd[SL] = fpd[SL] + np.random.normal(size=sum(SL),loc=0,
#                                scale=(fpderr_rdm[ele][src][0] \
#                                     + fpderr_rdm[ele][src][1] * tot[SL]) \
#                                     * np.sqrt(np.pi/2)) \
#                          + syserr[np.where(sysorder==S)] * tot[SL]

#        # Approach 3: Determine systematic offset distribution from all FPD
#        #             datasets for all electrolytes. Simulate new offsets from
#        #             that distribution.
#        fpd[SL] = fpd[SL] + np.random.normal(size=1,loc=0,
#                            scale=np.std(fpderr_sys['all_int'])) \
#                          + np.random.normal(size=1,loc=0,
#                            scale=np.std(fpderr_sys['all_grad'])) \
#                          * tot[SL] #\
##                          + np.random.normal(size=sum(SL),loc=0,
##                            scale=(fpderr_rdm[ele][src][0] \
##                                 + fpderr_rdm[ele][src][1] * tot[SL]) \
##                                 * np.sqrt(2/np.pi))

#        # Approach 4: As 3, but with Laplace distributions for systematics
#        fpd[SL] = fpd[SL] + np.random.laplace(size=1,loc=0,
#                            scale=np.sqrt(fpderr_sys['all_int_var']/2)) \
#                          + np.random.laplace(size=1,loc=0,
#                            scale=np.sqrt(fpderr_sys['all_grad_var']/2)) \
#                          * tot[SL] #\
##                          + np.random.normal(size=sum(SL),loc=0,
##                            scale=(fpderr_rdm[ele][src][0] \
##                                 + fpderr_rdm[ele][src][1] * tot[SL]) \
##                                 * np.sqrt(2/np.pi))
        
        # Approach 5: As 3/4, but with multivariate normal distribution for
        #             systematics
        syserrs = np.random.multivariate_normal([0,0],fpderr_sys['all_cov'])
        fpd[SL] = fpd[SL] + syserrs[0] + syserrs[1] * tot[SL] \
                          + np.random.normal(size=sum(SL),loc=0,
                            scale=(fpderr_rdm[ele][src][0] \
                                 + fpderr_rdm[ele][src][1] * tot[SL]) \
                                 * np.sqrt(2/np.pi))   

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
