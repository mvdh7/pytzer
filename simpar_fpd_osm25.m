%% Load Python outputs
cd 'E:\Dropbox\_UEA_MPH\pytzer'
load('pickles/simpar_fpd_osm25.mat');
% pfpd = struct2table(pfpd.pshape_fpd);
fpdbase = readtable('pickles/simpar_fpd_osm25.csv');
fpdsrcs.all.srcs = unique(fpdbase.src);

% fsim = readtable('pickles/fpdbase_sim_osm25.csv');

% %% Choose electrolyte to plot
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

for E = 1:numel(eles)
ele = eles{E};

load(['pickles/simloop_fpd_osm25_bC_' ele '_100.mat'])

% Define settings that depend upon electrolyte
eletit = ele;
switch ele
    case 'KCl'
        fxl = [0 5];
        fxt = 0:0.5:5;
        fyl = 0.05000000001*[-1 1];
        fyt = -0.12:0.01:0.12;
    case 'NaCl'
        fxl = [0 6.5];
        fxt = 0:0.5:6;
        fyl = 0.05000000001*[-1 1];
        fyt = -0.12:0.01:0.12;
    case 'CaCl2'
        fxl = [0 7.5];
        fxt = 0:0.5:6;
        fyl = 0.1200000001*[-1 1];
        fyt = -0.12:0.03:0.12;
        eletit = 'CaCl_2';
%         fpdbase = fpdbase(fpdbase.m < 3.5 ...
%             | ~strcmp(fpdbase.src,'OBS90'),:);
end %switch

% fpdbase = fsim;
% fpdbase.dosm25 = fpdbase.osm25_sim - fpdbase.osm25_calc;
% fpdbase.dfpd_sys = fpdbase.dfpd_sys + fpdbase.fpd - fpdbase.fpd_sim;

% Get logicals etc.
EL = strcmp(fpdbase.ele,ele);
fpdsrcs.(ele).srcs = unique(fpdbase.src(EL));

% Begin figure
figure(E); clf
printsetup(gcf,[9 12])
flegs = {};

subplot(2,2,1); hold on

patch([tot; flipud(tot)],[sqrt(Uosm_sim); flipud(-sqrt(Uosm_sim))], ...
    'y', 'edgecolor','none', 'facealpha',0.5)

    % Plot data by source
    for S = 1:numel(fpdsrcs.(ele).srcs)

        src = fpdsrcs.(ele).srcs{S};
        SL = EL & strcmp(fpdbase.src,src);
%         SL = SL & fpdbase.t == 298.15;
        
        scatter(sqrt(fpdbase.m(SL)),fpdbase.dosm25(SL), ...
            mksz*fmsm.(src),fclr.(src),'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0)
        
        if any(SL)
            Sx = [min(fpdbase.m(SL)) max(fpdbase.m(SL))];
            Sy = ones(size(Sx)) * fpderr_sys.(ele).(src);
%             Sy = (fpderr_sys.(ele).(src)(2) .* Sx ...
%                 + fpderr_sys.(ele).(src)(1)) ...
%                 .* pshape_fpd.(['dosm25_' ele])(SPL);
            nl = plot(sqrt(Sx),Sy, 'color',[fclr.(src) 0.5], ...
                'linewidth',0.5); nolegend(nl)
            flegs{end+1} = src;
        end %if
            
    end %for S
    
    xlim(sqrt(fxl))
    ylim(fyl)
    
%     plot(sqrt(0.1)*[1 1],fyl,'k')
    
    plot(get(gca,'xlim'),[0 0],'k')
    setaxes(gca,8)
    set(gca, 'box','on', 'xtick',fxt, 'ytick',fyt)
    set(gca, 'xticklabel',num2str(get(gca,'xtick')','%.1f'))
    set(gca, 'yticklabel',num2str(get(gca,'ytick')','%.2f'))
    
    xlabel(['[\itm\rm(' eletit ') / mol\cdotkg^{-1}]^{1/2}'])
    ylabel('\Delta\phi_{25}')
    
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

    xlim(sqrt(fxl))
%     ylim([0.999999999e-5 1])

    setaxes(gca,8)
    set(gca, 'box','on', 'xtick',fxt)%, 'ytick',10.^(-6:0))
%     set(gca, 'yticklabel',num2str(get(gca,'ytick')','%.1f'))
%     set(gca, 'YScale','log')
    
    xlabel(['[\itm\rm(' eletit ') / mol\cdotkg^{-1}]^{1/2}'])
    ylabel(['|\Delta\phi ' endash ' \itm\rm \delta_{FPD}| \times 10^{3}'])
    
    text(0,1.09,'(b)', 'units','normalized', 'fontname','arial', ...
        'fontsize',8, 'color','k')
    
    % Plot data by source
    for S = 1:numel(fpdsrcs.(ele).srcs)

        src = fpdsrcs.(ele).srcs{S};
        SL = EL & strcmp(fpdbase.src,src);
        
        scatter(sqrt(fpdbase.m(SL)),abs(fpdbase.dosm25_sys(SL)), ...
            mksz*fmsm.(src),fclr.(src),'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0)
        
        if any(SL)% && fpderr_rdm.(ele).(src)(2) ~= 0
            Sx = linspace(min(fpdbase.m(SL)),max(fpdbase.m(SL)),1000);
            Sy = fpderr_rdm.(ele).(src)(1) ...
                + fpderr_rdm.(ele).(src)(2) ...
                * exp(-Sx*fpderr_rdm.(ele).(src)(3));
%             Sy = fpderr_rdm.(ele).(src)(1) ...
%                 + fpderr_rdm.(ele).(src)(1) ./ Sx;
            nl = plot(sqrt(Sx),Sy, 'color',[fclr.(src) 0.5], ...
                'linewidth',0.5); nolegend(nl)
        end %if
            
    end %for S
    
%     fx = 0:0.01:6;
%     plot(fx,0.025*exp(-fx*3)+0.002,'k')
    
    set(gca, 'xticklabel',num2str(get(gca,'xtick')','%.1f'))
    set(gca, 'yticklabel',num2str(get(gca,'ytick')'*1e3))
    
    spfg2 = gca;
    
% Positioning    
spfig.Position = [0.15 0.58 0.6 0.35];
spfg2.Position = [0.15 0.1 0.6 0.35];
spleg.Position = [0.8 0.63 0.18 0.25];

print('-r300',['figures/simpar_fpd_osm25_' ele],'-dpng')

end %for E

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
fy = normpdf(fx,0,sd_Sn) * numel(fpderr_sys.all) * bw;

plot(fx,fy,'k', 'linewidth',1)

xlim(0.035*[-1 1])
ylim([0 15])

setaxes(gca,8)
set(gca, 'box','on')

xlabel('\delta_{FPD}')
ylabel('Number of datasets')

print('-r300','figures/simpar_fpd_osm25_hist','-dpng')
