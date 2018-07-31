%% Load Python outputs
load('pickles/simpytz_fpd.mat');
fpdbase = readtable('pickles/simpytz_fpd.csv');

% Preliminary calculations
fpdsrcs.all.srcs = unique(fpdbase.src);

% % Load Python interpreter
% pyversion C:\Users\yau17reu\anaconda\Anaconda3\envs\spritzer\python.exe

%% Plot raw FPD data

% Choose electrolyte to plot
eles = {'CaCl2' 'KCl' 'NaCl'};

for E = 1:numel(eles)
ele = eles{E};

% Define settings that depend upon electrolyte
switch ele
    case 'CaCl2'
        fxl = [0 1.6];
        fxt = 0:0.4:1.6;
        fyl = 0.3*[-1 1];
        ele2 = 'NaCl';
    case 'KCl'
        fxl = [0 3.5];
        fxt = 0:0.7:3.5;
        fyl = 0.5*[-1 1];
        ele2 = 'NaCl';
    case 'NaCl'
        fxl = [0 5.5];
        fxt = 0:1.1:5.5;
        fyl = 0.25*[-1 1.0000001];
        ele2 = 'KCl';
end %switch

% fxl = [0 5.5];
% fxt = 0:1.1:5.5;
% fyl = 0.5*[-1 1];

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
        if strcmp(ele,'CaCl2')
            SL = SL & fpdbase.m <= 1.5;
        end %if

        scatter(fpdbase.m(SL),fpdbase.dfpd(SL), ...
            mksz,fclr.(src),'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0.8)
        
        Sx = minmax(fpdbase.m(SL)');
        Sy = Sx * fpderr_sys.(ele).(src)(2) + fpderr_sys.(ele).(src)(1);
        plot(Sx,Sy, 'color',[fclr.(src) 0.5])

    end %for S

    % Axis settings
    xlim(fxl)
    ylim(fyl)
    
    plot(get(gca,'xlim'),[0 0],'k')
    setaxes(gca,8)
    set(gca, 'box','on', 'xtick',fxt, 'ytick',-1:0.1:1)
    
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
    xlim(fxl)
    ylim(fyl)
    
    nl = plot(get(gca,'xlim'),[0 0],'k'); nolegend(nl)
    setaxes(gca,8)
    set(gca, 'box','on', 'xtick',fxt, 'ytick',-1:0.1:1)
    
    xlabel(['\itm\rm(' ele ') / mol\cdotkg^{-1}'])
    ylabel('\sigma(\theta) / K')
    text(0,1.1,'(b)', 'fontname','arial', 'fontsize',8, 'color','k', ...
        'units','normalized')
    
% (c) Systematic vs random error components
subplot(2,2,3); hold on
    
    % Plot data by source
    for S = 1:numel(fpdsrcs.(ele).srcs)

        src = fpdsrcs.(ele).srcs{S};
        
%         if ismember(src,fpdsrcs.(ele2).srcs)
%             plot([sum(fpderr_sys.(ele).(src).^2) ...
%                   sum(fpderr_sys.(ele2).(src).^2)], ...
%               [sum(fpderr_rdm.(ele).(src).^2) ...
%                   sum(fpderr_rdm.(ele2).(src).^2)], ...
%                   'color',[fclr.(src) 0.3])
%         end %if
        
        % Original
        scatter(sum(fpderr_sys.(ele).(src).^2), ...
            sum(fpderr_rdm.(ele).(src).^2), ...
            mksz,fclr.(src),'filled', ...
            'marker',fmrk.(src), 'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0.8)
        
%         scatter(fpderr_sys.(ele).(src)(1), ...
%             fpderr_sys.(ele).(src)(2), ...
%             mksz,fclr.(src),'filled', ...
%             'marker',fmrk.(src), 'markeredgecolor',fclr.(src), ...
%             'markerfacealpha',0.7, 'markeredgealpha',0.8)

    end %for S
    
%     % Add the other electrolyte
%     for S = 1:numel(fpdsrcs.(ele2).srcs)
% 
%         src = fpdsrcs.(ele2).srcs{S};
%         
%         scatter(sum(fpderr_sys.(ele2).(src).^2), ...
%             sum(fpderr_rdm.(ele2).(src).^2), ...
%             mksz,fclr.(fpdsrcs.(ele2).srcs{S}),'filled', ...
%             'marker',fmrk.(fpdsrcs.(ele2).srcs{S}), ...
%             'markeredgecolor',fclr.(fpdsrcs.(ele2).srcs{S}), ...
%             'markerfacealpha',0, 'markeredgealpha',0.8)
% 
%     end %for S
    
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
        if strcmp(ele,'CaCl2')
            SL = SL & fpdbase.m <= 1.5;
        end %if
        
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
    set(gca, 'box','on', 'xtick',fxt, 'yscale','log', ...
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
    
%%
% Simulated datasets - NaCl
load('pickles/simloop_test.mat');
fpdtest = readtable('pickles/simloop_test.csv');

fvar = 'osm25';

tsrcs = unique(fpdtest.src);

U = 1 + U;
if U > 20
    U = 1;
end %if

for U = U%1%:20

figure(3); clf; hold on
printsetup(gcf,[12 9])

% Plot data by source
for S = 1:numel(tsrcs)

    src = tsrcs{S};
    SL = strcmp(fpdtest.src,src);

    switch fvar
        case 'osm25'
    
        % osm25 simulations
        scatter(fpdtest.m(SL),osm25_sim(SL,U)-fpdtest.osm25_calc(SL), ...
            mksz,fclr.(src),'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0.8)

        % osm25 original dataset
        scatter(fpdtest.m(SL),fpdtest.osm25(SL)-fpdtest.osm25_calc(SL), ...
            mksz,fclr.(src),'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.3, 'markeredgealpha',0.4)
    
        % osm25 simulation fits
        plot(tot_fitted,osm25_fitted(:,U) - osm25_fitted_calc,'k');

        case 'fpd'
    
        % FPD simulations
        scatter(fpdtest.m(SL),fpd_sim(SL,U)-fpdtest.fpd_calc(SL), ...
            mksz,fclr.(src),'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0.8)

        % FPD original dataset
        scatter(fpdtest.m(SL),fpdtest.fpd(SL)-fpdtest.fpd_calc(SL), ...
            mksz,fclr.(src),'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.2, 'markeredgealpha',0.3)
    
    end %switch
        
end %for S


% Axis settings
xlim([0 5.5])
setaxes(gca,8)
plot(get(gca,'xlim'),[0 0],'k')
xlabel(['\itm\rm(NaCl) / mol\cdotkg^{' endash '1}'])

switch fvar
    case 'osm25'
        ylim(0.0600000001*[-1 1])
        set(gca, 'box','on', 'xtick',0:1.1:5.5, 'ytick',-0.1:0.02:0.1)
        ylabel('\Delta\phi_{25}')
    case 'fpd'
        ylim(0.5*[-1 1])
        set(gca, 'box','on', 'xtick',0:1.1:5.5, 'ytick',-0.5:0.25:0.5)
        ylabel('\Delta\theta / K')
end %switch

set(gca, 'yticklabel',num2str(get(gca,'ytick')','%.2f'))

% print('-r300',['figures/simpytz_fpd/MCsim_' num2str(0,'%02.0f')],'-dpng')

end %for U

%%
ele = 'NaCl';

figure(2); clf

subplot(2,4,1)

    histogram([fpderr_sys.NaCl.all_int ...
               fpderr_sys.KCl.all_int ...
               fpderr_sys.CaCl2.all_int], -0.25:0.05:0.25)

subplot(2,4,2); hold on

    spdata = [fpderr_sys.NaCl.all_grad ...
               fpderr_sys.KCl.all_grad ...
               fpderr_sys.CaCl2.all_grad];

    histogram(spdata, -0.25:0.05:0.25)
           
    fx = -0.2:0.001:0.2;
    plot(fx,normpdf(fx,mean(spdata),std(spdata)))
    plot(fx,normpdf(fx,0,rms(spdata)))
    title(rms(spdata))
    
subplot(2,4,3)

    scatter(([fpderr_sys.NaCl.all_int ...
               fpderr_sys.KCl.all_int ...
               fpderr_sys.CaCl2.all_int]), ...
            ([fpderr_sys.NaCl.all_grad ...
               fpderr_sys.KCl.all_grad ...
               fpderr_sys.CaCl2.all_grad]))
           
subplot(2,4,4)

    histogram(log(sqrt([fpderr_sys.NaCl.all_int ...
               fpderr_sys.KCl.all_int ...
               fpderr_sys.CaCl2.all_int].^2 ...
               + [fpderr_sys.NaCl.all_grad ...
               fpderr_sys.KCl.all_grad ...
               fpderr_sys.CaCl2.all_grad].^2)),10)
           
subplot(2,4,5)

    histogram([fpderr_rdm.NaCl.all_int ...
               fpderr_rdm.KCl.all_int ...
               fpderr_rdm.CaCl2.all_int], 0:0.01:0.1)
    
subplot(2,4,6)

    histogram([fpderr_rdm.NaCl.all_grad ...
               fpderr_rdm.KCl.all_grad ...
               fpderr_rdm.CaCl2.all_grad], 0:0.01:0.1)
           
subplot(2,4,7)

    scatter([fpderr_rdm.NaCl.all_int ...
               fpderr_rdm.KCl.all_int ...
               fpderr_rdm.CaCl2.all_int], ...
            [fpderr_rdm.NaCl.all_grad ...
               fpderr_rdm.KCl.all_grad ...
               fpderr_rdm.CaCl2.all_grad])
           
subplot(2,4,8)

    histogram(log(sqrt([fpderr_rdm.NaCl.all_int ...
               fpderr_rdm.KCl.all_int ...
               fpderr_rdm.CaCl2.all_int].^2 ...
               + [fpderr_rdm.NaCl.all_grad ...
               fpderr_rdm.KCl.all_grad ...
               fpderr_rdm.CaCl2.all_grad].^2)),5)
