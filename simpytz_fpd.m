%% Load Python outputs
load('pickles/simpytz_fpdT0.mat');
fpdbase = readtable('pickles/simpytz_fpdT0.csv');

% Preliminary calculations
fpdbase.dfpd = fpdbase.fpd - fpdbase.fpd_calc;
fpdsrcs.all.srcs = unique(fpdbase.src);

% % Load Python interpreter
% pyversion C:\Users\yau17reu\anaconda\Anaconda3\envs\spritzer\python.exe

%% Plot raw FPD data

% Choose electrolyte to plot
ele = 'NaCl';

% Define marker styles
mrks = repmat({'o' 'v' '^' '<' '>' 'sq' 'd' 'p' 'h'},1,3);
clrs = repmat([228,26,28; 55,126,184; 77,175,74; 152,78,163; 255,127,0; 
    166,86,40; 247,129,191; 153,153,153] / 255,3,1);
for S = 1:numel(fpdsrcs.all.srcs)
    fmrk.(fpdsrcs.all.srcs{S}) = mrks{S};
    fclr.(fpdsrcs.all.srcs{S}) = clrs(S,:);
end %for S

% Get logicals etc.
EL = strcmp(fpdbase.ele,ele);
fpdsrcs.(ele).srcs = unique(fpdbase.src(EL));

% Begin figure
figure(1); clf
printsetup(gcf,[9 12])

% (a) Raw FPD measurements' residuals
subplot(2,2,1); hold on

    % Plot data by source
    for S = 1:numel(fpdsrcs.(ele).srcs)

        SL = EL & strcmp(fpdbase.src,fpdsrcs.(ele).srcs{S});

        scatter(fpdbase.m(SL),fpdbase.dfpd(SL),20, ...
            fclr.(fpdsrcs.(ele).srcs{S}),'filled', ...
            'marker',fmrk.(fpdsrcs.(ele).srcs{S}), ...
            'markeredgecolor',fclr.(fpdsrcs.(ele).srcs{S}), ...
            'markerfacealpha',0.7, 'markeredgealpha',0.8)

    end %for S

    % Axis settings
    plot(get(gca,'xlim'),[0 0],'k')
    setaxes(gca,8)
    set(gca, 'box','on')
    
    xlabel(['\itm\rm(' ele ') / mol\cdotkg^{-1}'])
    ylabel('\Delta(\theta) / K')
    text(0,1.1,'(a)', 'fontname','arial', 'fontsize',8, 'color','k', ...
        'units','normalized')

% (b) After systematic error correction
subplot(2,2,2); hold on
    
    % Plot data by source
    for S = 1:numel(fpdsrcs.(ele).srcs)

        SL = EL & strcmp(fpdbase.src,fpdsrcs.(ele).srcs{S});

        scatter(fpdbase.m(SL),fpdbase.dfpd(SL),20, ...
            fclr.(fpdsrcs.(ele).srcs{S}),'filled', ...
            'marker',fmrk.(fpdsrcs.(ele).srcs{S}), ...
            'markeredgecolor',fclr.(fpdsrcs.(ele).srcs{S}), ...
            'markerfacealpha',0.7, 'markeredgealpha',0.8)

    end %for S
    
    plot(get(gca,'xlim'),[0 0],'k')
    setaxes(gca,8)
    set(gca, 'box','on')
    
    xlabel(['\itm\rm(' ele ') / mol\cdotkg^{-1}'])
    ylabel('\sigma(\theta) / K')
    text(0,1.1,'(b)', 'fontname','arial', 'fontsize',8, 'color','k', ...
        'units','normalized')
    