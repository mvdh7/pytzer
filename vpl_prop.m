load('pickles/vpl_prop.mat')
load('pickles/simpar_vpl.mat')

figure(1); clf; hold on
printsetup(gcf,[9 6])

xlim([0 6.25])
ylim(0.4*[-1 1])

nl = plot(get(gca,'xlim'),[0 0],'k'); nolegend(nl)

plot(tot,dosm_dT,   'color',[230,159,0]/255, 'linestyle','-')
plot(tot,dosm_dtot, 'color',[0,114,178]/255, 'linestyle',':')
plot(tot,dosm_dvpX, 'color',[0,158,115]/255, 'linestyle','--')

setaxes(gca,8)
set(gca, 'box','on', 'xtick',0:6, 'ytick',-.4:.2:.4)
set(gca, 'yticklabel',num2str(get(gca,'ytick')','%.1f'))

xlabel(['\itm\rm(NaCl) / mol\cdotkg^{' endash '1}'])
ylabel('\partial\phi/\partial\itX\rm')

legend('\itT\rm','\itm\rm','\itp\rm', 'location','ne')

set(gca, 'position',[0.15 0.2 0.75 0.7])

plot(tot,vplerr_sys.sd_Sn./tot,'k')
% plot(tot,dosm_dT * 0.1, 'r--') % best = 0.1 K
% plot(tot,dosm_dtot * -0.01, 'r--') % best = 0.01 to 0.1 mol/kg
plot(tot,dosm_dvpX * -1,'r--') % best = 1 

% print('-r300','figures/vpl_prop','-dpng')
