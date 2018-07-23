%% Load Python outputs
load('pickles/simpytz_fpd.mat');
fpdbase = readtable('pickles/simpytz_fpd.csv');

% Preliminary calculations
fpdsrcs.all.srcs = unique(fpdbase.src);

% % Load Python interpreter
% pyversion C:\Users\yau17reu\anaconda\Anaconda3\envs\spritzer\python.exe

% Plot raw FPD data

% Choose electrolyte to plot
eles = {'NaCl' 'KCl'};

for E = 1:numel(eles)
ele = eles{E};

% Define settings that depend upon electrolyte
switch ele
    case 'NaCl'
        fxl = [0 5.5];
        fyl = 0.25*[-1 1.0000001];
        ele2 = 'KCl';
    case 'KCl'
        fxl = [0 3.5];
        fyl = 0.5*[-1 1];
        ele2 = 'NaCl';
end %switch

% Define marker styles
mrks = repmat({'o' 'v' '^' '<' '>' 'sq' 'd' 'p' 'h'},1,3);
clrs = repmat([228,26,28; 55,126,184; 77,175,74; 152,78,163; 255,127,0; 
    166,86,40; 247,129,191; 153,153,153] / 255,3,1);
for S = 1:numel(fpdsrcs.all.srcs)
    fmrk.(fpdsrcs.all.srcs{S}) = mrks{S};
    fclr.(fpdsrcs.all.srcs{S}) = clrs(S,:);
end %for S
mksz = 10;

% Get logicals etc.
EL = strcmp(fpdbase.ele,ele);
fpdsrcs.(ele).srcs = unique(fpdbase.src(EL));

% Begin figure
figure(1); clf
printsetup(gcf,[14 10])

% (a) Raw FPD measurements' residuals
subplot(2,2,1); hold on

    % Plot data by source
    for S = 1:numel(fpdsrcs.(ele).srcs)

        src = fpdsrcs.(ele).srcs{S};
        SL = EL & strcmp(fpdbase.src,src);

        scatter(fpdbase.m(SL),fpdbase.dfpd(SL), ...
            mksz,fclr.(src),'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0.8)
        
        Sx = minmax(fpdbase.m(SL)');
        Sy = Sx * fpderr_sys.(ele).(src)(2) + fpderr_sys.(ele).(src)(1);
        plot(Sx,Sy, 'color',[fclr.(src) 0.5])

    end %for S

    % Axis settings
    plot(get(gca,'xlim'),[0 0],'k')
    setaxes(gca,8)
    set(gca, 'box','on', 'xtick',0:10, 'ytick',-1:0.1:1)
    
    xlim(fxl)
    ylim(fyl)
    
    xlabel(['\itm\rm(' ele ') / mol\cdotkg^{-1}'])
    ylabel('\Delta(\theta) / K')
    text(0,1.1,'(a)', 'fontname','arial', 'fontsize',8, 'color','k', ...
        'units','normalized')

% (b) After systematic error correction
subplot(2,2,2); hold on
    
    % Plot data by source
    for S = 1:numel(fpdsrcs.(ele).srcs)

        src = fpdsrcs.(ele).srcs{S};
        SL = EL & strcmp(fpdbase.src,fpdsrcs.(ele).srcs{S});

        scatter(fpdbase.m(SL),fpdbase.dfpd_sys(SL), ...
            mksz,fclr.(src),'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0.8)

    end %for S
    
    legend(fpdsrcs.(ele).srcs)
    
    % Axis settings
    nl = plot(get(gca,'xlim'),[0 0],'k'); nolegend(nl)
    setaxes(gca,8)
    set(gca, 'box','on', 'xtick',0:10, 'ytick',-1:0.1:1)
    
    xlim(fxl)
    ylim(fyl)
    
    xlabel(['\itm\rm(' ele ') / mol\cdotkg^{-1}'])
    ylabel('\sigma(\theta) / K')
    text(0,1.1,'(b)', 'fontname','arial', 'fontsize',8, 'color','k', ...
        'units','normalized')
    
% (c) Systematic vs random error components
subplot(2,2,3); hold on
    
    % Plot data by source
    for S = 1:numel(fpdsrcs.(ele).srcs)

        src = fpdsrcs.(ele).srcs{S};
        
        if ismember(src,fpdsrcs.(ele2).srcs)
            plot([sum(fpderr_sys.(ele).(src).^2) ...
                  sum(fpderr_sys.(ele2).(src).^2)], ...
              [sum(fpderr_rdm.(ele).(src).^2) ...
                  sum(fpderr_rdm.(ele2).(src).^2)], ...
                  'color',[fclr.(src) 0.3])
        end %if
        
        scatter(sum(fpderr_sys.(ele).(src).^2), ...
            sum(fpderr_rdm.(ele).(src).^2), ...
            mksz,fclr.(src),'filled', ...
            'marker',fmrk.(src), 'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0.8)

    end %for S
    
    % Add the other electrolyte
    for S = 1:numel(fpdsrcs.(ele2).srcs)

        src = fpdsrcs.(ele2).srcs{S};
        
        scatter(sum(fpderr_sys.(ele2).(src).^2), ...
            sum(fpderr_rdm.(ele2).(src).^2), ...
            mksz,fclr.(fpdsrcs.(ele2).srcs{S}),'filled', ...
            'marker',fmrk.(fpdsrcs.(ele2).srcs{S}), ...
            'markeredgecolor',fclr.(fpdsrcs.(ele2).srcs{S}), ...
            'markerfacealpha',0, 'markeredgealpha',0.8)

    end %for S
    
    % Axis settings
    setaxes(gca,8)
    set(gca, 'box','on', 'xscale','log', 'yscale','log', ...
        'xtick',10.^(-10:0), 'ytick',10.^(-10:0))
    
    xlim([10^-7 10^-1])
    ylim([10^-7 10^-1])
%     plot(get(gca,'xlim'),get(gca,'xlim'),'k')
    
    xlabel('\delta')
    ylabel('\sigma')
    text(0,1.1,'(c)', 'fontname','arial', 'fontsize',8, 'color','k', ...
        'units','normalized')
    
% (d) After systematic error correction
subplot(2,2,4); hold on
    
    % Plot data by source
    for S = 1:numel(fpdsrcs.(ele).srcs)

        src = fpdsrcs.(ele).srcs{S};
        SL = EL & strcmp(fpdbase.src,fpdsrcs.(ele).srcs{S});

        scatter(fpdbase.m(SL),abs(fpdbase.dfpd_sys(SL)), ...
            mksz,fclr.(src),'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0.8)
        
        Sx = linspace(min(fpdbase.m(SL)),max(fpdbase.m(SL)),100);
        Sy = Sx * fpderr_rdm.(ele).(src)(2) + fpderr_rdm.(ele).(src)(1);
        plot(Sx,Sy, 'color',[fclr.(src) 0.5])

    end %for S
    
    % Axis settings
    setaxes(gca,8)
    set(gca, 'box','on', 'xtick',0:10, 'yscale','log', ...
        'ytick',10.^(-10:10))
    
    xlim(fxl)
    ylim([0 fyl(2)])
    
    xlabel(['\itm\rm(' ele ') / mol\cdotkg^{-1}'])
    ylabel('|\sigma(\theta)| / K')
    text(0,1.1,'(d)', 'fontname','arial', 'fontsize',8, 'color','k', ...
        'units','normalized')
    
% Save figure
print('-r300',['figures/simpytz_fpd_' ele],'-dpng')

end %for E
    