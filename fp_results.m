load('splines/fp_vpl_NaCl.mat')

ftot = sqrt(tot);

figure(1); clf
printsetup(gcf,[9 12])

subplot(2,1,1); hold on

    patch([ftot; flipud(ftot)], ...
        [osm_sim+sqrt(Uosm_sim); flipud(osm_sim-sqrt(Uosm_sim))], ...
        0.6*[1 1 1], 'edgecolor','none')
    plot(ftot,osm_sim,'k', 'linewidth',1)

    setaxes(gca,8)
    set(gca, 'box','on')
    xlim(ftot([1 end]))
    
subplot(2,1,2); hold on

    plot(ftot,sqrt(Uosm_sim),'k')
    
    setaxes(gca,8)
    set(gca, 'box','on')
    xlim(ftot([1 end]))
    