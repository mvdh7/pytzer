load('bC_H2SO4_dis.mat')

% Enforce variance-covariance matrix symmetry
bCmx = mean(cat(3,bCmx,transpose(bCmx)),3);

covmx_viz(1,bCs,bCmx,{'\beta_0' '\beta_1' '\itC\rm_0' '\itC\rm_1' ...
    '\beta_0' '\beta_1' '\itC\rm_0' '\itC\rm_1'});
