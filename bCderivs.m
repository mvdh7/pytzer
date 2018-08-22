% Load data
load('pickles/bCderivs.mat');
sqt = sqrt(tot);

% Define colours
b0clr = [166 206 227]/255;
b1clr = [ 31 120 180]/255;

C0clr = [178 223 138]/255;
C1clr = [ 51 160  44]/255;

% Choose x-axis variable
ftot = tot;

% Make figure
figure(1); clf
printsetup(gcf,[9 10])

subplot(2,1,1); hold on

    nl = plot(ftot,osm0-osm0,'k'); nolegend(nl)
    plot(ftot,osm -osm0, 'k')
    
    plot(ftot,osm0 + db0 * bCs(1) - osm0, 'color',b0clr)
    plot(ftot,osm0 + db1 * bCs(2) - osm0, 'color',b1clr)
    plot(ftot,osm0 + dC0 * bCs(3) - osm0, 'color',C0clr)
    plot(ftot,osm0 + dC1 * bCs(4) - osm0, 'color',C1clr)
    
    % Axis settings
    setaxes(gca,8)
    xlim([0 6.25])
    ylim([-0.1 0.6])

    set(gca, 'box','on', 'xtick',0:6, 'ytick',-0.1:0.1:0.6)
    set(gca, 'xticklabel',num2str(get(gca,'xtick')','%.0f'), ...
        'yticklabel',num2str(get(gca,'ytick')','%.1f'))

    % xlabel(['[\itm\rm(NaCl) / mol\cdotkg^{' endash '1}]^{1/2}'])
    xlabel(['\itm\rm(NaCl) / mol\cdotkg^{' endash '1}'])
    ylabel('\Delta\phi')

    % Legend
    legend('D-H','\beta_0','\beta_1','\itC\rm_0','\itC\rm_1', ...
        'location','eastoutside')

    % Position
    set(gca, 'position',[0.12 0.6 0.6 0.375])

subplot(2,1,2); hold on

    plot(ftot,db0 / max(db0), 'color',b0clr)
    plot(ftot,db1 / max(db1), 'color',b1clr)
    plot(ftot,dC0 / max(dC0), 'color',C0clr)
    plot(ftot,dC1 / max(dC1), 'color',C1clr)

    plot(ftot,db1_an / max(db1_an),':' , 'color',b1clr)
    plot(ftot,db1_au / max(db1_au),'--', 'color',b1clr)

    plot(ftot,dC1_on / max(dC1_on),':' , 'color',C1clr)
    plot(ftot,dC1_ou / max(dC1_ou),'--', 'color',C1clr)

    % Axis settings
    setaxes(gca,8)
    xlim([0 6.25])
    ylim([0 1])

    set(gca, 'box','on', 'xtick',0:6, 'ytick',0:0.2:1)
    set(gca, 'xticklabel',num2str(get(gca,'xtick')','%.0f'), ...
        'yticklabel',num2str(get(gca,'ytick')','%.1f'))

    % xlabel(['[\itm\rm(NaCl) / mol\cdotkg^{' endash '1}]^{1/2}'])
    xlabel(['\itm\rm(NaCl) / mol\cdotkg^{' endash '1}'])
    ylabel('d\phi/d\itX\rm / max(d\phi/d\itX\rm)')

    % Legend
    legend('\beta_0','\beta_1','\itC\rm_0','\itC\rm_1', ...
        'location','eastoutside')

    % Position
    set(gca, 'position',[0.12 0.1 0.6 0.375])

% Print
print('-r300','figures/bCderivs','-dpng')
