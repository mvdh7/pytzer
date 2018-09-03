% Select electrolyte
ele = 'NaCl';

% Set limits
switch ele
    case 'NaCl'
        fxl = [0 6.25];
%         fyl = [-0.1 0.6];
        fyl = [0.7 1.3];
    case 'KCl'
        fxl = [0 3.5];
        fyl = [-0.1 0.20000001];
end %switch

% Load data
load(['pickles/bCderivs_' ele '.mat']);
sqt = sqrt(tot);

% Define colours
b0clr = [255 192   0]/255;
b1clr = [146 208  80]/255;

C0clr = [  0 176 240]/255;
% C1clr = [ 51 160  44]/255;

fbgclr = [48 48 48]/255;

% Choose x-axis variable
ftot = tot;

% Make figure
figure(1); clf
printsetup(gcf,[23 10])
set(gcf, 'color',fbgclr, 'inverthardcopy','off')

subplot(1,2,1); hold on

%     patch([ftot; flipud(ftot)],[osm0; flipud(osm0 + db0 * bCs(1))], ...
%         b0clr, 'edgecolor','w')
%     patch([ftot; flipud(ftot)],[osm0 + db0 * bCs(1); 
%         flipud(osm0 + db0 * bCs(1) + db1*bCs(2))], ...
%         b1clr, 'edgecolor','w')
%     patch([ftot; flipud(ftot)],[osm0 + db0 * bCs(1) + db1*bCs(2); 
%         flipud(osm0 + db0 * bCs(1) + db1*bCs(2) + dC0*bCs(3))], ...
%         C0clr, 'edgecolor','w')

    nl = plot(ftot,zeros(size(ftot)), 'color',0.9*[1 1 1]); nolegend(nl)
    plot(ftot,osm0, 'color','w', 'linewidth',1)
    plot(ftot,osm, 'color','w', 'linewidth',1)
    
%     plot(ftot,osm0 + db0 * bCs(1), 'color',b0clr, 'linewidth',1)
%     plot(ftot,osm0 + db1 * bCs(2), 'color',b1clr, 'linewidth',1)
%     plot(ftot,osm0 + dC0 * bCs(3), 'color',C0clr, 'linewidth',1)
    
    
    
%     plot(ftot,dC1 * bCs(4), 'color',C1clr)
    
%     % Check the calculation
%     plot(ftot, ...
%         db0 * bCs(1) + db1 * bCs(2) + dC0 * bCs(3) + dC1 * bCs(4), ...
%         'r:')
    
    % Axis settings
    setaxes(gca,12)
    set(gca, 'color',fbgclr, 'xcolor',0.9*[1 1 1], 'ycolor',0.9*[1 1 1])
    xlim(fxl)
    ylim(fyl)

    set(gca, 'box','on', 'xtick',0:6, 'ytick',-0.1:0.1:2)
    set(gca, 'xticklabel',num2str(get(gca,'xtick')','%.0f'), ...
        'yticklabel',num2str(get(gca,'ytick')','%.1f'))

    % xlabel(['[\itm\rm(' ele ') / mol\cdotkg^{' endash '1}]^{1/2}'])
    xlabel(['\itm\rm(' ele ') / mol\cdotkg^{' endash '1}'])
    ylabel('\phi')
%     text(0,1.08,'(a)', 'units','normalized', 'fontname','arial', ...
%         'color','k', 'fontsize',8)
    
%     % Legend
%     legend('All','\beta_0','\beta_1','\itC\rm_0','\itC\rm_1', ...
%         'location','eastoutside')
    
    % Position
%     set(gca, 'position',[0.12 0.6 0.6 0.35])
    
%     % Messing around
% %     fy = log(ftot+1)*0.1 + sqrt(ftot)*0.01;
% %     fy = (1 - exp(-ftot)) * 0.1 + sqrt(ftot)*0.05 + ftot*0.02;
%     
%     yfitt = fittype('log(m+1)*a + m^2*b + m*c + m^0.5*d + m^0.25*e', ...
%         'independent','m', 'coefficients',{'a' 'b' 'c' 'd' 'e'});
%     yopts = fitoptions(yfitt);
%     yopts.StartPoint = zeros(numel(coeffnames(yfitt)),1);
%     yfit = fit(ftot,osm-osm0,yfitt,yopts);
%     fy = feval(yfit,ftot);
%     
%     fcf = coeffvalues(yfit);
%         
%     plot(ftot,fy,'r--')
%     plot(ftot,log(ftot+1)*fcf(1),'r:')
%     plot(ftot,ftot.^2*fcf(2),'m:')
%     plot(ftot,ftot*fcf(3),'b:')
%     plot(ftot,ftot.^0.5*fcf(4),'c:')
%     plot(ftot,ftot.^0.25*fcf(5),'k:')
%     
%     disp(yfit)

subplot(1,2,2); hold on

    plot(ftot,db0 / max(db0), 'color',b0clr, 'linewidth',1)
    plot(ftot,db1 / max(db1), 'color',b1clr, 'linewidth',1)
    plot(ftot,dC0 / max(dC0), 'color',C0clr, 'linewidth',1)
%     nl = plot(ftot,dC1 / max(dC1), 'color',C1clr); nolegend(nl)

%     plot(ftot,db1_an / max(db1_an),':' , 'color',b1clr)
%     plot(ftot,db1_au / max(db1_au),'--', 'color',b1clr)
% 
%     plot(ftot,dC1_on / max(dC1_on),':' , 'color',C1clr)
%     plot(ftot,dC1_ou / max(dC1_ou),'--', 'color',C1clr)

    % Axis settings
    setaxes(gca,12)
    set(gca, 'color',fbgclr, 'xcolor',0.9*[1 1 1], 'ycolor',0.9*[1 1 1])
    xlim(fxl)
    ylim([0 1])

    set(gca, 'box','on', 'xtick',0:6, 'ytick',0:0.2:1)
    set(gca, 'xticklabel',num2str(get(gca,'xtick')','%.0f'), ...
        'yticklabel',num2str(get(gca,'ytick')','%.1f'))

    % xlabel(['[\itm\rm(' ele ') / mol\cdotkg^{' endash '1}]^{1/2}'])
    xlabel(['\itm\rm(' ele ') / mol\cdotkg^{' endash '1}'])
    ylabel('d\phi/d\itX\rm / max(d\phi/d\itX\rm)')
%     text(0,1.08,'(b)', 'units','normalized', 'fontname','arial', ...
%         'color','k', 'fontsize',8)

%     % Legend
%     legend('+\alpha_1',[endash '\alpha_1'],'+\omega',[endash '\omega'], ...
%         'location','eastoutside')

    % Position
%     set(gca, 'position',[0.12 0.1 0.6 0.35])

% Print
print('-r300',['figures/bCderivs_pres_' ele '_0'],'-dpng')
