load('pickles/simloop_pytzer_bC_KCl_9600.mat')

bC_mean = bC_mean([1 2 4]) ;
bCo     = bCo    ([1 2 4])';
bC_cv = bC_cv(:,[1 2 4]);
bC_cv = bC_cv([1 2 4],:);

varnames = {'\beta_0' '\beta_1' '\itC\rm_0'};

covmx_viz(1,bC_mean,bC_cv ,varnames)
covmx_viz(2,bCo    ,bCo_cv,varnames)

%%
load('pickles/simloop_res.mat')
sqt = sqrt(tot);
plx = sqt;

figure(3); clf

subplot(2,1,1); hold on

    plot(plx,acf,'k')
    plot(plx,acf+sqrt(Uacf)*2,'k--')
    plot(plx,acf-sqrt(Uacf)*2,'k--')

    plot(plx,acfo,'r')
    plot(plx,acfo+sqrt(Uacfo)*2,'r--')
    plot(plx,acfo-sqrt(Uacfo)*2,'r--')

    grid on

subplot(2,1,2); hold on

    plot(plx,acf-acf,'k')
    plot(plx,sqrt(Uacf)*2,'k--')
    plot(plx,sqrt(Uacf)*2,'k--')

    plot(plx,acfo-acf,'r')
    plot(plx,acfo+sqrt(Uacfo)*2-acf,'r--')
    plot(plx,acfo-sqrt(Uacfo)*2-acf,'r--')

    plot(plx,sqrt(Uacfo)*2,'m--')
    
    grid on
    