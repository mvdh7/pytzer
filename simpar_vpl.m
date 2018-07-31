%% Load Python outputs
vplbase = readtable('pickles/simpar_vpl.csv');
vplsrcs.all.srcs = unique(vplbase.src);

%% Plot raw VPL data

% Choose electrolyte to plot
eles = {'KCl' 'NaCl'};

% Define marker styles
mrks = repmat({'o' 'v' '^' '<' '>' 'sq' 'd' 'p' 'h'},1,3);
clrs = repmat([228,26,28; 55,126,184; 77,175,74; 152,78,163; 255,127,0; 
    166,86,40; 247,129,191; 153,153,153] / 255,3,1);
for S = 1:numel(vplsrcs.all.srcs)
    fmrk.(vplsrcs.all.srcs{S}) = mrks{S};
    fclr.(vplsrcs.all.srcs{S}) = clrs(S,:);
end %for S
mksz = 10;

for E = 1%:numel(eles)
ele = eles{E};

% Get logicals etc.
EL = strcmp(vplbase.ele,ele);
vplsrcs.(ele).srcs = unique(vplbase.src(EL));

% Begin figure
figure(2); clf
printsetup(gcf,[14 10])

% (a) Raw VPL measurements' residuals, only data at 298.15 K
subplot(2,2,1); hold on

    % Plot data by source
    for S = 1:numel(vplsrcs.(ele).srcs)

        src = vplsrcs.(ele).srcs{S};
        SL = EL & strcmp(vplbase.src,src);
        SL = SL & vplbase.t == 298.15;

        scatter(vplbase.m(SL),vplbase.dosm25(SL), ...
            mksz,fclr.(src),'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0.8)
        
    end %for S
    
    plot(get(gca,'xlim'),[0 0],'k')
    setaxes(gca,8)
    set(gca, 'box','on')
    
    xlabel(['\itm\rm(' ele ') / mol\cdotkg^{-1}'])
    ylabel('\Delta(\phi_{25}) / K')
    text(0,1.1,'(a)', 'fontname','arial', 'fontsize',8, 'color','k', ...
        'units','normalized')

% (b) Raw VPL measurements' residuals, only data at 298.15 K
subplot(2,2,2); hold on

    % Plot data by source
    for S = 1:numel(vplsrcs.(ele).srcs)

        src = vplsrcs.(ele).srcs{S};
        SL = EL & strcmp(vplbase.src,src);
        SL = SL & vplbase.t ~= 298.15;

        scatter(vplbase.m(SL),vplbase.dosm25(SL), ...
            mksz,fclr.(src),'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0.8)
        
    end %for S
    
    plot(get(gca,'xlim'),[0 0],'k')
    setaxes(gca,8)
    set(gca, 'box','on')
    
    xlabel(['\itm\rm(' ele ') / mol\cdotkg^{-1}'])
    ylabel('Converted \Delta(\phi_{25}) / K')
    text(0,1.1,'(b)', 'fontname','arial', 'fontsize',8, 'color','k', ...
        'units','normalized')
    
end %for E
