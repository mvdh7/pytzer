load('pickles/simloop_pytzer_bC_KCl_960.mat');

bC_mean = bC_mean([1 2 4]) ;
bCo     = bCo    ([1 2 4])';
bC_cv = bC_cv(:,[1 2 4]);
bC_cv = bC_cv([1 2 4],:);

varnames = {'\beta_0' '\beta_1' '\itC\rm_0'};

covmx_viz(1,bC_mean,bC_cv ,varnames)
covmx_viz(2,bCo    ,bCo_cv,varnames)
