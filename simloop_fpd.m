fele = 'KCl';
load(['pickles/simloop_fpd_bC_' fele '_10000.mat'])
% load(['pickles/simloop_vpl_bC_' fele '_10.mat'])

bCsim = bCsim([1 2 4]) ;
bCdir = bCdir([1 2 4])';
bCsim_cv = bCsim_cv(:,[1 2 4]);
bCsim_cv = bCsim_cv([1 2 4],:);
bCdir_cv = bCdir_cv(:,[1 2 4]);
bCdir_cv = bCdir_cv([1 2 4],:);

varnames = {'\beta_0' '\beta_1' '\itC\rm_0'};

covmx_viz(1,bCsim,bCsim_cv,varnames)
covmx_viz(2,bCdir,bCdir_cv,varnames)

sqt = sqrt(tot);
plx = sqt;

figure(3); clf

subplot(2,1,1); hold on

    plot(plx,acfMX_sim,'k')
    plot(plx,acfMX_sim+sqrt(UacfMX_sim)*2,'k--')
    plot(plx,acfMX_sim-sqrt(UacfMX_sim)*2,'k--')

    plot(plx,acfMX_dir,'r')
    plot(plx,acfMX_dir+sqrt(UacfMX_dir)*2,'r--')
    plot(plx,acfMX_dir-sqrt(UacfMX_dir)*2,'r--')

    grid on

    xlabel(['[\itm\rm(' fele ') / mol\cdotkg^{' endash '1}]^{1/2}'])
    ylabel('ln \gamma_\pm')
    
subplot(2,1,2); hold on

    plot(plx,acfMX_sim-acfMX_sim,'k')
    plot(plx,sqrt(UacfMX_sim)*2,'k--')
    plot(plx,sqrt(UacfMX_sim)*2,'k--')

    plot(plx,acfMX_dir-acfMX_sim,'r')
    plot(plx,acfMX_dir+sqrt(UacfMX_dir)*2-acfMX_sim,'r--')
    plot(plx,acfMX_dir-sqrt(UacfMX_dir)*2-acfMX_sim,'r--')

    plot(plx,sqrt(UacfMX_dir)*2,'m--')
    
    grid on
    
    xlabel(['[\itm\rm(' fele ') / mol\cdotkg^{' endash '1}]^{1/2}'])
    ylabel('\Delta ln \gamma_\pm')
    