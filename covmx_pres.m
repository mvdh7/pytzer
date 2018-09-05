load('pickles/simloop_fpd_bC_KCl_10000.mat')

bCsim = bCsim([1 2 4]);
bCsim_cv = bCsim_cv([1 2 4],:);
bCsim_cv = bCsim_cv(:,[1 2 4]);

rng(295)

spl = covmx_viz(1,bCsim,bCsim_cv,{'\beta_0' '\beta_1' '\itC\rm_\phi'}, ...
    'dark');

%%
printsetup(gcf,[30 20])
set(gcf, 'color',48/255 * [1 1 1], 'inverthardcopy','off')

spls = fieldnames(spl);

for S = 1:numel(spls)
    
    spl.(spls{S}).Box = 'off';
    
    spl.(spls{S}).Color = 0 * [1 1 1];
    spl.(spls{S}).XColor = 0.9  * [1 1 1];
    spl.(spls{S}).YColor = 0.9  * [1 1 1];
    
    spl.(spls{S}).Title = [];
    spl.(spls{S}).FontSize = 12;
    
    spl.(spls{S}).XTickLabel = num2str(spl.(spls{S}).XTick','%.2f');
    spl.(spls{S}).YTickLabel = num2str(spl.(spls{S}).YTick','%.2f');
    
end %for S

print('-r300','figures/covmx_pres','-dpng')
