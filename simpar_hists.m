% Load fit data
load('pickles/simpar_vpl.mat')
load('pickles/simpar_fpd_osm25.mat')
load('pickles/simpar_iso_all.mat')

isoerr_sys.sd_Sn = isoerr_sys.all_Sn;

syserrs   = {vplerr_sys fpderr_sys isoerr_sys};
sysletter = {'v' 'd' 'q'};
sysxlim   = [0.045 0.04 8e-3];
sysylim   = [4 10 12];
sysxtick  = [0.015 0.01 2e-3];
sysytick  = [1 2 2];
sysbw     = [0.0075 0.003333333333333 0.0005];

clrvpl = [255 140   0]/255;
clrfpd = [  0 191 255]/255;
clriso = [255 105 180]/255;
sysclrs = {clrvpl clrfpd clriso};

%
figure(1); clf
printsetup(gcf,[12 10])

for S = 1:3
    
subplot(2,2,S); hold on
    
    Sx = linspace(-sysxlim(S),sysxlim(S),500);
    Sy = normpdf(Sx,0,syserrs{S}.sd_Sn) * numel(syserrs{S}.all) * sysbw(S);

    xlim(sysxlim(S) * [-1 1])
    ylim([0 sysylim(S)])

    histogram(syserrs{S}.all,-sysxlim(S):sysbw(S):sysxlim(S), ...
        'facecolor',sysclrs{S}, 'edgecolor',sysclrs{S}, ...
        'facealpha',0.3, 'edgealpha',0.6)
    
    plot(get(gca,'xlim'),[0 0],'k')
    
    plot(Sx,Sy, 'color',mean([sysclrs{S};0.5 0.5 0.5],1))%, 'linewidth',1)
    
    setaxes(gca,8)
    set(gca, 'box','on', 'xtick',-sysxlim(S):sysxtick(S):sysxlim(S), ...
        'ytick',0:sysytick(S):50)
    set(gca, 'xticklabel',num2str(get(gca,'xtick')'*1e3,'%.0f'))
    
    xlabel(['\delta_{\it' sysletter{S} '\rm} \times 10^3'])
    ylabel(['\itn_{' sysletter{S} '}\rm'])
    
    text(0,1.1,['(' lcletter(S) ')'], 'units','normalized', ...
        'fontname','arial', 'fontsize',8, 'color','k')

end %for S

print('-r300','figures/simpar_hists','-dpng')
