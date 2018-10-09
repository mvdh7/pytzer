%% Load Python outputs
load('pickles/simpar_vpl.mat');
vplbase = readtable('pickles/simpar_vpl.csv');
vplsrcs.all.srcs = unique(vplbase.src);
vplc = struct2table(load('pickles/vplcurve.mat'));

% Plot raw VPL data

% Choose electrolyte to plot
eles = {'KCl' 'NaCl' 'CaCl2'};

% Define marker styles
mrks = repmat({'o' 'v' '^' '<' '>' 'sq' 'h' 'p' 'd'},1,3);
msms = repmat([ 1   1   1   1   1   1    1   3   1 ],1,3);
clrs = repmat([228,26,28; 55,126,184; 77,175,74; 152,78,163; 255,127,0; 
    166,86,40; 247,129,191; 153,153,153; 114 13 14] / 255,3,1);
for S = 1:numel(vplsrcs.all.srcs)
    fmrk.(vplsrcs.all.srcs{S}) = mrks{S};
    fclr.(vplsrcs.all.srcs{S}) = clrs(S,:);
    fmsm.(vplsrcs.all.srcs{S}) = msms(S);
end %for S
mksz = 10;

for E = 3%1:numel(eles)
ele = eles{E};

load(['pickles/simloop_vpl_bC_' ele '_100.mat']);

% Define settings that depend upon electrolyte
eletit = ele;
fxtf = '%.1f';
switch ele
    case 'KCl'
        fxl = [0 5];
        fxt = 0:5;
        fyl = 0.080000001*[-1 1];
        fyti = 0.02;
        fyl2 = [0 0.03];
        fyti2 = 0.01;
    case 'NaCl'
        fxl = [0 6.5];
        fxt = 0:6;
        fyl = 0.02000000001*[-1 1.0000001];
        fyti = 0.005;
        fyl2 = [0 0.012];
        fyti2 = 0.003;
    case 'CaCl2'
        fxl = [0 7.5];
        fxt = 0:1.5:7.5;
        fyl = 0.2*[-1 1];
        fyti = 0.05;
        eletit = 'CaCl_2';
        fxtf = '%.1f';
        fyl2 = [0 0.2];
        fyti2 = 0.05;
end %switch

% Get logicals etc.
EL = strcmp(vplbase.ele,ele);
vplsrcs.(ele).srcs = unique(vplbase.src(EL));

% Begin figure
figure(E); clf
printsetup(gcf,[9 12])
flegs = {};

subplot(2,2,1); hold on

patch([tot; flipud(tot)],[sqrt(Uosm_sim); flipud(-sqrt(Uosm_sim))], ...
    'y', 'edgecolor','none', 'facealpha',0.5)

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
    set(gca, 'box','on', 'xtick',fxt, 'ytick',-1:fyti:1)
    set(gca, 'yticklabel',num2str(get(gca,'ytick')'*1e3,'%.0f'))
    set(gca, 'xticklabel',num2str(get(gca,'xtick')',fxtf))
    
    xlabel(['\itm\rm(' eletit ') / mol\cdotkg^{-1}'])
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
    ylim(fyl2)

    setaxes(gca,8)
    set(gca, 'box','on', 'xtick',fxt, 'ytick',0:fyti2:1)
    set(gca, 'yticklabel',num2str(get(gca,'ytick')'*1e3,'%.0f'))
    set(gca, 'xticklabel',num2str(get(gca,'xtick')',fxtf))
    
    xlabel(['\itm\rm(' eletit ') / mol\cdotkg^{-1}'])
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

print('-r300',['figures/simpar_vpl_' ele],'-dpng')

end %for E

%% Make table
ele = 'KCl';

TVPL = cell(numel(vplerr_rdm.(ele).all_int),8);

TVPL(:,1) = {ele};
TVPL(:,2) = vplsrcs.(ele).srcs;

for S = 1:numel(vplsrcs.(ele).srcs)
    src = vplsrcs.(ele).srcs{S};
    SL = strcmp(vplbase.ele,ele) & strcmp(vplbase.src,src) ...
        & vplbase.t == 298.15;
    TVPL{S,3} = num2str(sum(SL));
    TVPL{S,4} = num2str(min(vplbase.m(SL)),'%.3f');
    TVPL{S,5} = num2str(max(vplbase.m(SL)),'%.3f');
    TVPL{S,6} = num2str(vplerr_sys.(ele).(src)(1)*1e3,'%+.3f');
    TVPL{S,7} = num2str(vplerr_rdm.(ele).(src)(1)*1e3,'%+.3f');
    TVPL{S,8} = num2str(vplerr_rdm.(ele).(src)(2),'%+.3f');
end %for S

%% Histograms
figure(4); clf; hold on
printsetup(gcf,[9 6])

all_sys = [vplerr_sys.CaCl2.all_int ...
           vplerr_sys.KCl.all_int ...
           vplerr_sys.NaCl.all_int]';

all_rdm_int = [vplerr_rdm.CaCl2.all_int ...
               vplerr_rdm.KCl.all_int ...
               vplerr_rdm.NaCl.all_int]';
       
all_rdm_grad = [vplerr_rdm.CaCl2.all_grad ...
               vplerr_rdm.KCl.all_grad ...
               vplerr_rdm.NaCl.all_grad]';
           
L = all_sys ~= 0;           
           
histogram(all_sys(L),-0.044:0.011:0.044, 'facecolor',0.3*[1 1 1])

xlim(0.04*[-1 1])
ylim([0 4])

fx = -0.044:0.0001:0.044;
fy = normpdf(fx,0,sqrt(mean(all_sys(L).^2))) / numel(all_sys(L));

plot(fx,fy,'k', 'linewidth',1)

% scatter(all_rdm_int,abs(all_sys))
% xlim([0 0.01])

clc
disp(['VPL sys RMS: ' num2str(sqrt(mean(all_sys(L).^2)))])

xlabel('\delta_{VPL}')
ylabel('Number of datasets')

setaxes(gca,8)
set(gca, 'box','on', 'ytick',0:5, 'xtick',-0.04:0.01:0.044)

print('-r300','figures/simpar_vpl_hist','-dpng')

%% random components - not really worth showing!
figure(5); clf

L = all_rdm_grad > 0 | all_rdm_int > 0;

scatter(all_rdm_grad(L),all_rdm_int(L))
