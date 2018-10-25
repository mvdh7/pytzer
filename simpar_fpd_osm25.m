%% Load Python outputs
load('pickles/simpar_fpd_osm25.mat');
fpdbase = readtable('pickles/simpar_fpd_osm25.csv');
fpdsrcs.all.srcs = unique(fpdbase.src);

% Plot raw FPD data

% Use common y-axes?
COMMON_Y = 0;

% Do simulations?
DO_SIMS = 0;

if DO_SIMS
    SS = 1:20;
else %if ~DO_SIMS
    SS = 0;
end %if DO_SIMS

% Choose electrolyte to plot
eles = {'NaCl' 'KCl' 'CaCl2'};

% Define marker styles
[fmrk,fclr,fmsm] = simpar_markers(fpdsrcs.all.srcs);
mksz = 10;

for SIM = SS

% Begin figure
figure(1); clf
printsetup(gcf,[18 12])
flegs = {};

for E = 1:numel(eles)
ele = eles{E};

load(['pickles/simloop_fpd_osm25_bC_' ele '_100.mat']);

% Define settings that depend upon electrolyte
eletit = ele;
fxtf = '%.1f';
switch ele
    case 'KCl'
        fxl = [0 5];
        fxt = 0:0.5:5;
%         fxtf = '%.0f';
        fyl = 0.04000000001*[-1 1];
        fyti = 0.01;
%         fyl2 = [0 0.03];
%         fyti2 = 0.01;
    case 'NaCl'
        fxl = [0 6.5];
        fxt = 0:0.5:6;
%         fxtf = '%.0f';
        fyl = 0.0400000001*[-1 1.0000001];
        fyti = 0.01;
%         fyl2 = [0 0.012];
%         fyti2 = 0.003;
    case 'CaCl2'
        fxl = [0 7.5];
        fxt = 0:0.5:7.5;
        fyl = 0.12*[-1 1];
        fyti = 0.03;
        eletit = 'CaCl_2';
%         fxtf = '%.1f';
%         fyl2 = [0 0.2];
%         fyti2 = 0.05;
end %switch

% Override above y-axis settings if requested
if COMMON_Y
    fyl = 0.12*[-1 1];
    fyti = 0.03;
%     fylr
end %if COMMON_Y

if DO_SIMS
    
    fpdbase = readtable(['pickles/Uosm_sim_fpd_osm25_' ele '.csv']);
    Uosm_sim_fpd = load(['pickles/Uosm_sim_fpd_osm25_' ele '.mat']);
    Uosm_sim_fpd = Uosm_sim_fpd.Uosm_sim;
    
    fpdbase.dosm25 = Uosm_sim_fpd(:,SIM) - fpdbase.osm_calc;
    
end %if

% Get logicals etc.
EL = strcmp(fpdbase.ele,ele);
fpdsrcs.(ele).srcs = unique(fpdbase.src(EL));

subplot(2,4,E); hold on

patch(sqrt([tot; flipud(tot)]), ...
    [sqrt(Uosm_sim); flipud(-sqrt(Uosm_sim))], ...
    [1 1 0], 'edgecolor','none', 'facealpha',0.5)
plot(sqrt(tot), sqrt(Uosm_sim),'y')
plot(sqrt(tot),-sqrt(Uosm_sim),'y')

    xlim(sqrt(fxl))
    ylim(fyl)
    
    plot(get(gca,'xlim'),[0 0],'k')

    % Plot data by source
    for S = 1:numel(fpdsrcs.(ele).srcs)

        src = fpdsrcs.(ele).srcs{S};
        SL = EL & strcmp(fpdbase.src,src);
               
        scatter(sqrt(fpdbase.m(SL)),fpdbase.dosm25(SL), ...
            mksz*fmsm.(src),fclr.(src),'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0)
        
        if any(SL)
            Sx = linspace(min(fpdbase.m(SL)),max(fpdbase.m(SL)),100);
            Sy = fpderr_sys.(ele).(src) * ones(size(Sx));
            nl = plot(sqrt(Sx),Sy, 'color',[fclr.(src) 0.5], ...
                'linewidth',0.5); nolegend(nl)
            if ~ismember(src,flegs)
                flegs{end+1} = src;
            end %if
        end %if
            
    end %for S
    
    setaxes(gca,8)
    set(gca, 'box','on', 'xtick',fxt, 'ytick',-1.2:fyti:1)
    set(gca, 'yticklabel',num2str(get(gca,'ytick')'*1e3,'%.0f'))
    set(gca, 'xticklabel',num2str(get(gca,'xtick')',fxtf))
    
    xlabel(['[\itm\rm(' eletit ') / mol\cdotkg^{-1}]^{1/2}'])
    ylabel('\itR_d\rm \times 10^{3}')
    
    text(0,1.09,['(' lcletter(E) ')'], 'units','normalized', ...
        'fontname','arial', 'fontsize',8, 'color','k')
    
    plotbox(gca)
    spfig.(['e' num2str(E)]) = gca;

subplot(2,4,E+4); hold on

    xlim(sqrt(fxl))
    ylim([10^-6 10^0])

    setaxes(gca,8)
    set(gca, 'box','on', 'xtick',fxt)%, 'ytick',0:fyti2:1)
    set(gca, 'yscale','log')
%     set(gca, 'yticklabel',num2str(get(gca,'ytick')'*1e3,'%.0f'))
    set(gca, 'xticklabel',num2str(get(gca,'xtick')',fxtf))
    
    xlabel(['[\itm\rm(' eletit ') / mol\cdotkg^{-1}]^{1/2}'])
    ylabel(['|\itR_d\rm ' endash ...
        ' \it\Delta_{d}\rm(\itm\rm,\it\delta_d\rm)|'])
    
    text(0,1.09,['(' lcletter(E+3) ')'], 'units','normalized', ...
        'fontname','arial', 'fontsize',8, 'color','k')
    
    % Plot data by source
    for S = 1:numel(fpdsrcs.(ele).srcs)

        src = fpdsrcs.(ele).srcs{S};
        SL = EL & strcmp(fpdbase.src,src);
        
        scatter(sqrt(fpdbase.m(SL)),abs(fpdbase.dosm25_sys(SL)), ...
            mksz*fmsm.(src),fclr.(src),'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0)
        
        if any(SL)
            Sx = linspace(sqrt(min(fpdbase.m(SL))), ...
                sqrt(max(fpdbase.m(SL))),500).^2;
            Sy = fpderr_rdm.(ele).(src)(2) ...
                .* exp(-Sx*fpderr_rdm.(ele).(src)(3)) ...
                + fpderr_rdm.(ele).(src)(1);
            nl = plot(sqrt(Sx),Sy, 'color',[fclr.(src) 0.5], ...
                'linewidth',0.5); nolegend(nl)
        end %if
            
    end %for S
    
    plotbox(gca)
    
    spfg2.(['e' num2str(E)]) = gca;
    
end %for E

subplot(1,4,4); hold on    
    
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
    
% Positioning
for E = 1:numel(eles)
    spfig.(['e' num2str(E)]).Position = [0.08+(E-1)*0.28 0.6 0.19 0.34];
    spfg2.(['e' num2str(E)]).Position = [0.08+(E-1)*0.28 0.1 0.19 0.34];
end %for E
spleg.Position = [0.88 0.235 0.1 0.53];

if ~DO_SIMS
    print('-r300',['figures/simpar_fpd_osm25_' num2str(COMMON_Y)],'-dpng')
else
    print('-r300',['figures/simpar_fpd_osm25/sim_' num2str(SIM)],'-dpng')
end %if

end %for SIM

%% Make table
ele = 'CaCl2';

TFPD = cell(numel(fpderr_sys.(ele).all),8);

TFPD(:,1) = {ele};
TFPD(:,2) = fpdsrcs.(ele).srcs;

for S = 1:numel(fpdsrcs.(ele).srcs)
    src = fpdsrcs.(ele).srcs{S};
    SL = strcmp(fpdbase.ele,ele) & strcmp(fpdbase.src,src);
    TFPD{S,3} = num2str(sum(SL));
    TFPD{S,4} = num2str(min(fpdbase.m(SL)),'%.3f');
    TFPD{S,5} = num2str(max(fpdbase.m(SL)),'%.3f');
    TFPD{S,6} = num2str(fpderr_sys.(ele).(src)(1)*1e3,'%+.3f');
    TFPD{S,7} = num2str(fpderr_rdm.(ele).(src)(1)*1e3,'%+.3f');
    TFPD{S,8} = num2str(fpderr_rdm.(ele).(src)(2),'%+.3f');
end %for S

%% histogram
figure(4); clf; hold on
printsetup(gcf,[9 6])

bw = 0.005;

histogram(fpderr_sys.all,-0.03:bw:0.03, 'normalization','count', ...
    'facecolor',0.3*[1 1 1])

all_sysL = fpderr_sys.all;

Sn0 = NaN(numel(all_sysL));
Sn1 = NaN(numel(all_sysL),1);
for C = 1:numel(all_sysL)
    Sn0(:,C) = abs(all_sysL(C) - all_sysL);
    Sn1(C) = median(Sn0(Sn0(:,C) ~= 0,C));
end %for C
sd_Sn = 1.1926 * median(Sn1);

fx = -0.035:0.0001:0.035;
% fy = normpdf(fx,0,fpderr_sys.all_rmse) * numel(fpderr_sys.all) * bw;
fy = normpdf(fx,0,fpderr_sys.sd_Sn) * numel(fpderr_sys.all) * bw;

plot(fx,fy,'k', 'linewidth',1)

xlim(0.035*[-1 1])
ylim([0 15])

setaxes(gca,8)
set(gca, 'box','on')

xlabel('\delta_{FPD}')
ylabel('Number of datasets')

print('-r300','figures/simpar_fpd_osm25_hist','-dpng')
