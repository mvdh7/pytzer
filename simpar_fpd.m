%% Load Python outputs
cd 'E:\Dropbox\_UEA_MPH\pytzer'
load('pickles/simpar_fpd.mat');
fpdbase = readtable('pickles/simpar_fpd.csv');
% fpdbase = readtable('pickles/fpdbase_intermediate.csv');
% fpdbase.dfpd_sys = NaN(height(fpdbase),1);
fpdsrcs.all.srcs = unique(fpdbase.src);

fsim = readtable('pickles/fpdbase_sim.csv');

% Plot raw FPD data

% Choose electrolyte to plot
eles = {'KCl' 'NaCl' 'CaCl2'};

% Define marker styles
mrks = repmat({'o' 'v' '^' '<' '>' 'sq' 'd' 'p' 'h'},1,3);
msms = repmat([ 1   1   1   1   1   1    1   3   1 ],1,3);
clrs = repmat([228,26,28; 55,126,184; 77,175,74; 152,78,163; 255,127,0; 
    166,86,40; 247,129,191; 153,153,153] / 255,3,1);
for S = 1:numel(fpdsrcs.all.srcs)
    fmrk.(fpdsrcs.all.srcs{S}) = mrks{S};
    fclr.(fpdsrcs.all.srcs{S}) = clrs(S,:);
    fmsm.(fpdsrcs.all.srcs{S}) = msms(S);
end %for S
mksz = 10;

for E = 2%1:numel(eles)
ele = eles{E};

% % NB this is for osm25 not FPD!
% load(['pickles/simloop_fpd_bC_' ele '_100.mat']);

% Define settings that depend upon electrolyte
eletit = ele;
switch ele
    case 'KCl'
        fxl = [0 5];
        fxt = 0:5;
        fyl = 0.5*[-1 1];
    case 'NaCl'
        fxl = [0 6.5];
        fxt = 0:6;
        fyl = 0.3*[-1 1.0000001];
    case 'CaCl2'
        fxl = [0 4.5];
        fxt = 0:6;
        fyl = 0.6*[-1 1.0000001];
        eletit = 'CaCl_2';
end %switch

% fpdbase = fsim;
% fpdbase.dfpd = fpdbase.fpd_sim - fpdbase.fpd_calc;
% fpdbase.dfpd_sys = fpdbase.dfpd_sys + fpdbase.fpd - fpdbase.fpd_sim;

% Get logicals etc.
EL = strcmp(fpdbase.ele,ele);
fpdsrcs.(ele).srcs = unique(fpdbase.src(EL));

% Begin figure
figure(E); clf
printsetup(gcf,[9 12])
flegs = {};

subplot(2,2,1); hold on

patch([tot; flipud(tot)],10*[sqrt(Uosm_sim); flipud(-sqrt(Uosm_sim))], ...
    'y', 'edgecolor','none', 'facealpha',0.5)

    % Plot data by source
    for S = 1:numel(fpdsrcs.(ele).srcs)

        src = fpdsrcs.(ele).srcs{S};
        SL = EL & strcmp(fpdbase.src,src);
%         SL = SL & fpdbase.t == 298.15;
        
        scatter(fpdbase.m(SL),fpdbase.dfpd(SL), ...
            mksz*fmsm.(src),fclr.(src),'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0)
        
        if any(SL)
            Sx = linspace(min(fpdbase.m(SL)),max(fpdbase.m(SL)),100);
            Sy = fpderr_sys.(ele).(src)(2) .* Sx ...
                + fpderr_sys.(ele).(src)(1);
            nl = plot(Sx,Sy, 'color',[fclr.(src) 0.5], ...
                'linewidth',0.5); nolegend(nl)
            flegs{end+1} = src;
        end %if
            
    end %for S
    
    xlim(fxl)
    ylim(fyl)
    
    plot(get(gca,'xlim'),[0 0],'k')
    setaxes(gca,8)
    set(gca, 'box','on', 'xtick',fxt, 'ytick',-1:0.1:1)
    set(gca, 'yticklabel',num2str(get(gca,'ytick')','%.1f'))
    
    xlabel(['\itm\rm(' eletit ') / mol\cdotkg^{-1}'])
    ylabel('\Delta\itd\rm / K')
    
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
    ylim([0.999999999e-5 1])

    setaxes(gca,8)
    set(gca, 'box','on', 'xtick',fxt, 'ytick',10.^(-6:0))
%     set(gca, 'yticklabel',num2str(get(gca,'ytick')','%.1f'))
    set(gca, 'YScale','log')
    
    xlabel(['\itm\rm(' eletit ') / mol\cdotkg^{-1}'])
    ylabel(['|\Delta\phi ' endash ' \itm\rm \delta_{FPD}| \times 10^{3}'])
    
    text(0,1.09,'(b)', 'units','normalized', 'fontname','arial', ...
        'fontsize',8, 'color','k')
    
    % Plot data by source
    for S = 1:numel(fpdsrcs.(ele).srcs)

        src = fpdsrcs.(ele).srcs{S};
        SL = EL & strcmp(fpdbase.src,src);
        
        scatter(fpdbase.m(SL),abs(fpdbase.dfpd_sys(SL)), ...
            mksz*fmsm.(src),fclr.(src),'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0)
        
        if any(SL)
            Sx = linspace(min(fpdbase.m(SL)),max(fpdbase.m(SL)),100);
            Sy = fpderr_rdm.(ele).(src)(2) .* Sx ...
                + fpderr_rdm.(ele).(src)(1);
            nl = plot(Sx,Sy, 'color',[fclr.(src) 0.5], ...
                'linewidth',0.5); nolegend(nl)
        end %if
            
    end %for S
    
    spfg2 = gca;
    
% Positioning    
spfig.Position = [0.15 0.58 0.6 0.35];
spfg2.Position = [0.15 0.08 0.6 0.35];
spleg.Position = [0.8 0.63 0.18 0.25];

% print('-r300',['figures/simpar_fpd_' ele],'-dpng')

end %for E

%% Histograms
load('pickles/simpar_fpd.mat');

figure(4); clf
printsetup(gcf,[12 10])

% Systematic - intercept
bw = 0.025;

L = fpderr_sys.all_int ~= 0;
F = L & fpderr_sys.all_int > -0.15;
fx = -0.25:0.0001:0.25;
fy1 = normpdf(fx,0,sqrt(fpderr_sys.all_int_var)) * sum(L) * bw; % Normal
ld = sqrt(fpderr_sys.all_int_var / 2);
fy = exp(-abs(fx)/ld)/(2*ld) * sum(L) * bw; % laplace

subplot(2,2,1); hold on
    
    histogram(fpderr_sys.all_int(L),-0.25:bw:0.25, ...
        'normalization','count', 'facecolor',0.3*[1 1 1])
    plot(fx,fy, 'color','k', 'linestyle','--')
    plot(fx,fy1, 'color','k')
    
    xlim(0.25*[-1 1])
    ylim([0 12])
    
    setaxes(gca,8)
    set(gca, 'box','on', 'xtick',-0.2:0.1:0.2, 'ytick',0:3:25)
    
    xlabel('')
    ylabel('Frequency')
    
% Systematic - gradient
L = fpderr_sys.all_grad ~= 0;

bw = 0.01;

fx = -0.1:0.0001:0.1;
fy1 = normpdf(fx,0,sqrt(fpderr_sys.all_grad_var)) * sum(L) * bw; % Normal
ld = sqrt(fpderr_sys.all_grad_var / 2);
fy = exp(-abs(fx)/ld)/(2*ld) * sum(L) * bw; % laplace

subplot(2,2,2); hold on
    L = fpderr_sys.all_grad ~= 0;
    histogram(fpderr_sys.all_grad(L),-0.08:bw:0.08, ...
        'normalization','count', 'facecolor',0.3*[1 1 1])
    plot(fx,fy, 'color','k', 'linestyle','--')
    plot(fx,fy1, 'color','k')
    
    xlim(0.08*[-1 1])
    ylim([0 6])
    
    setaxes(gca,8)
    set(gca, 'box','on', 'xtick',-0.08:0.04:0.08, 'ytick',0:40)
    
    xlabel('')
    ylabel('Frequency')
    
% Random - intercept
bw = 0.5;

fx = -6:0.01:1;
L = fpderr_rdm.all_int > 0;
fy = normpdf(fx,mean(log10(fpderr_rdm.all_int(L))), ...
    std(log10(fpderr_rdm.all_int(L)))) * sum(L) * bw;

subplot(2,2,3); hold on
    histogram(log10(fpderr_rdm.all_int(L)),-11:bw:1, ...
        'normalization','count', 'facecolor',0.3*[1 1 1])
    plot(fx,fy,'k')
    
    xlim([-6 1])
    ylim([0 6])
    
    setaxes(gca,8)
    set(gca, 'box','on', 'xtick',-11:1, 'ytick',0:9)
    
    xlabel('')
    ylabel('Frequency')
    
% Random - gradient
bw = 1/5;

L = fpderr_rdm.all_grad > 0;
fx = -4.5:0.01:-1.5;
fy = normpdf(fx,nanmean(log10(fpderr_rdm.all_grad(L))), ...
    nanstd(log10(fpderr_rdm.all_grad(L)))) * sum(L) * bw;
ld = sqrt(var(log10(fpderr_rdm.all_grad(L))) / 2);
mu = mean(log10(fpderr_rdm.all_grad(L)));
fy1 = exp(-abs(fx-mu)/ld)/(2*ld) * sum(L) * bw; % laplace

subplot(2,2,4); hold on
    histogram(log10(fpderr_rdm.all_grad(L)),-4.5:bw:-1.5, ...
        'normalization','count', 'facecolor',0.3*[1 1 1])
    plot(fx,fy,'k')
    plot(fx,fy1,'k--')
    
    xlim([-4.5 -1.5])
    ylim([0 5])
    
    setaxes(gca,8)
    set(gca, 'box','on', 'xtick',-11.5:1, 'ytick',0:9)
    
    xlabel('')
    ylabel('Frequency')
    
% sys covariance
figure(5); clf; hold on

L = fpderr_sys.all_grad ~= 0;

scatter(fpderr_sys.all_int(L),fpderr_sys.all_grad(L))

syscov = cov(fpderr_sys.all_int(L),fpderr_sys.all_grad(L));
