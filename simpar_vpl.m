%% Load Python outputs
load('pickles/simpar_vpl.mat');
vplbase = readtable('pickles/simpar_vpl.csv');
vplsrcs.all.srcs = unique(vplbase.src);
vplc = struct2table(load('pickles/vplcurve.mat'));

% Plot raw VPL data

% Choose electrolyte to plot
eles = {'KCl' 'NaCl' 'CaCl2'};

% Define marker styles
mrks = repmat({'o' 'v' '^' '<' '>' 'sq' 'd' 'p' 'h'},1,3);
msms = repmat([ 1   1   1   1   1   1    1   3   1 ],1,3);
clrs = repmat([228,26,28; 55,126,184; 77,175,74; 152,78,163; 255,127,0; 
    166,86,40; 247,129,191; 153,153,153] / 255,3,1);
for S = 1:numel(vplsrcs.all.srcs)
    fmrk.(vplsrcs.all.srcs{S}) = mrks{S};
    fclr.(vplsrcs.all.srcs{S}) = clrs(S,:);
    fmsm.(vplsrcs.all.srcs{S}) = msms(S);
end %for S
mksz = 10;

for E = 3%:numel(eles)
ele = eles{E};

% Define settings that depend upon electrolyte
switch ele
    case 'KCl'
        fxl = [0 5];
        fxt = 0:5;
        fyl = 0.080000001*[-1 1];
    case 'NaCl'
        fxl = [0 6.5];
        fxt = 0:6;
        fyl = 0.015000000001*[-1 1.0000001];
    case 'CaCl2'
        fxl = [0 7.5];
        fxt = 0:1.5:7.5;
        fyl = 0.2*[-1 1];
end %switch

% Get logicals etc.
EL = strcmp(vplbase.ele,ele);
vplsrcs.(ele).srcs = unique(vplbase.src(EL));

% Begin figure
figure(2); clf
printsetup(gcf,[9 12])
flegs = {};

subplot(2,2,1); hold on

    % Plot data by source
    for S = 1:numel(vplsrcs.(ele).srcs)

        src = vplsrcs.(ele).srcs{S};
        SL = EL & strcmp(vplbase.src,src);
        SL = SL & vplbase.t == 298.15;
               
        scatter(vplbase.m(SL),vplbase.dosm25(SL), ...
            mksz*fmsm.(src),fclr.(src),'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0)
        
        if any(SL)
            Sx = linspace(min(vplbase.m(SL)),max(vplbase.m(SL)),100);
            Sy = vplerr_sys.(ele).(src)(1) ./ Sx ...
                + vplerr_sys.(ele).(src)(2);
            nl = plot(Sx,Sy, 'color',[fclr.(src) 0.5], ...
                'linewidth',0.5); nolegend(nl)
            flegs{end+1} = src;
        end %if
            
    end %for S
    
    xlim(fxl)
    ylim(fyl)
    
    plot(get(gca,'xlim'),[0 0],'k')
    setaxes(gca,8)
    set(gca, 'box','on', 'xtick',fxt, 'ytick',-1:0.005:1)
    set(gca, 'yticklabel',num2str(get(gca,'ytick')'*1e3,'%.0f'))
    
    xlabel(['\itm\rm(' ele ') / mol\cdotkg^{-1}'])
    ylabel('\Delta\phi \times 10^{3}')
    
    text(0,1.09,'(a)', 'units','normalized', 'fontname','arial', ...
        'fontsize',8, 'color','k')
    
    spfig = gca;

subplot(2,2,2); hold on    
    
    setaxes(gca,8)
    set(gca, 'xtick',[], 'ytick',[], 'box','on')
    
    for S = 1:numel(flegs)
        
        src = flegs{S};
        
        scatter(0.6,numel(flegs)-S, mksz*fmsm.(src)*1.5,fclr.(src), ...
            'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0)
        
        text(1.1,numel(flegs)-S,src, 'fontname','arial', 'fontsize',8, ...
            'color','k')
        
    end %for S
    
    xlim([0 5])
    ylim([-0.75 numel(flegs)-0.25])
    
    spleg = gca;

subplot(2,2,3); hold on

    xlim(fxl)
    ylim([0 max(fyl)])

    setaxes(gca,8)
    set(gca, 'box','on', 'xtick',fxt, 'ytick',-1:0.005:1)
    set(gca, 'yticklabel',num2str(get(gca,'ytick')'*1e3,'%.0f'))
    
    xlabel(['\itm\rm(' ele ') / mol\cdotkg^{-1}'])
    ylabel(['|\Delta\phi ' endash ' \delta_{VPL}/\itm\rm| \times 10^{3}'])
    
    text(0,1.09,'(b)', 'units','normalized', 'fontname','arial', ...
        'fontsize',8, 'color','k')
    
    % Plot data by source
    for S = 1:numel(vplsrcs.(ele).srcs)

        src = vplsrcs.(ele).srcs{S};
        SL = EL & strcmp(vplbase.src,src);
        SL = SL & vplbase.t == 298.15;
        
        scatter(vplbase.m(SL),abs(vplbase.dosm25_sys(SL)), ...
            mksz*fmsm.(src),fclr.(src),'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0)
        
        if any(SL)
            Sx = linspace(min(vplbase.m(SL)),max(vplbase.m(SL)),100);
            Sy = vplerr_rdm.(ele).(src)(2) .* exp(-Sx) ...
                + vplerr_rdm.(ele).(src)(1);
            nl = plot(Sx,Sy, 'color',[fclr.(src) 0.5], ...
                'linewidth',0.5); nolegend(nl)
        end %if
            
    end %for S
    
    spfg2 = gca;
    
% Positioning    
spfig.Position = [0.15 0.58 0.6 0.35];
spfg2.Position = [0.15 0.08 0.6 0.35];
spleg.Position = [0.8 0.63 0.18 0.25];

% print('-r300',['figures/simpar_vpl_' ele],'-dpng')

end %for E
