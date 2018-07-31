%%
% Simulated datasets - NaCl
load('pickles/simloop_vpl_test.mat');
vpltest = readtable('pickles/simloop_vpl_test.csv');
%%
fvar = 'osm';

tsrcs = unique(vpltest.src);

U = 1 + U;
if U > 20
    U = 1;
end %if

for U = U%1%:20

figure(3); clf; hold on
printsetup(gcf,[12 9])

% Plot data by source
for S = 1:numel(tsrcs)

    src = tsrcs{S};
    SL = strcmp(vpltest.src,src);
    
    % osm simulations
    scatter(vpltest.m(SL),osm_sim(SL,U)-vpltest.osm_calc(SL), ...
        mksz,fclr.(src),'filled', 'marker',fmrk.(src), ...
        'markeredgecolor',fclr.(src), ...
        'markerfacealpha',0.7, 'markeredgealpha',0.8)

    % osm original dataset
    scatter(vpltest.m(SL),vpltest.osm_meas(SL)-vpltest.osm_calc(SL), ...
        mksz,fclr.(src),'filled', 'marker',fmrk.(src), ...
        'markeredgecolor',fclr.(src), ...
        'markerfacealpha',0.3, 'markeredgealpha',0.4)

    % osm simulation fits
    plot(tot_fitted,osm_fitted(:,U) - osm_fitted_calc,'k');
        
end %for S

% Axis settings
xlim([0 5.5])
setaxes(gca,8)
plot(get(gca,'xlim'),[0 0],'k')
xlabel(['\itm\rm(NaCl) / mol\cdotkg^{' endash '1}'])

ylim(0.0600000001*[-1 1])
set(gca, 'box','on', 'xtick',0:1.1:5.5, 'ytick',-0.1:0.02:0.1)
ylabel('\Delta\phi_{25}')

set(gca, 'yticklabel',num2str(get(gca,'ytick')','%.2f'))

% print('-r300',['figures/simpytz_vpl/MCsim_' num2str(0,'%02.0f')],'-dpng')

end %for U
