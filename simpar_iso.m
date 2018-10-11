% Select test and reference electrolytes
tst = 'KCl';
tsttit = 'KCl';
ref = 'NaCl';

% Load data from simpar_iso.py
froot = 'pickles/simpar_iso_';
isobase = readtable([froot 'isobase_t' tst '_r' ref '.csv']);
load([froot 'pshape_t' tst '_r' ref '.mat'])
load([froot 'isoerr_t' tst '_r' ref '.mat'])

% Define marker styles
srcs = unique(isobase.src);
mrks = repmat({'o' 'v' '^' '<' '>' 'sq' 'h' 'p' 'd'},1,3);
msms = repmat([ 1   1   1   1   1   1    1   3   1 ],1,3);
clrs = repmat([228,26,28; 55,126,184; 77,175,74; 152,78,163; 255,127,0; 
    166,86,40; 247,129,191; 153,153,153; 114 13 14] / 255,3,1);
for S = 1:numel(srcs)
    fmrk.(srcs{S}) = mrks{S};
    fclr.(srcs{S}) = clrs(S,:);
    fmsm.(srcs{S}) = msms(S);
end %for S
mksz = 10;

figure(1); clf
printsetup(gcf,[9 12])

subplot(2,1,1); hold on

    xlim([0 5])
    ylim(0.012*[-1 1])

    for S = 1:numel(srcs)

        SL = strcmp(isobase.src,srcs{S});

        scatter(isobase.(tst)(SL),isobase.(['dosm_' tst])(SL), ...
            mksz*fmsm.(srcs{S}),fclr.(srcs{S}),'filled', ...
            'marker',fmrk.(srcs{S}))

        TL = tot >= min(isobase.(tst)(SL)) & tot <= max(isobase.(tst)(SL));
        plot(tot(TL),isoerr_sys.(srcs{S}) * (1 + 1./(tot(TL) + 0.03)), ...
            'color',fclr.(srcs{S}), 'linewidth',1)

    end %for S
    
    plot(get(gca,'xlim'),[0 0],'k')
    
    setaxes(gca,8)
    set(gca, 'box','on')
    
    xlabel(['\itm\rm(' tsttit ') / mol\cdotkg^{' endash '1}'])

    % %% Components
    % plot(tot,dosm_dtot *5e-3, 'linewidth',2)
    % plot(tot,dosm_dtotR*1e-2, 'linewidth',2)
    % plot(tot,(dosm_dtot + dosm_dtotR)*5e-3, 'linewidth',2)
    % plot(tot,4e-4*(1 + 1./(tot+0.03)),'k', 'linewidth',2)
    
    spfig = gca;

subplot(2,1,2); hold on

    for S = 1:numel(srcs)

        SL = strcmp(isobase.src,srcs{S});

        scatter(isobase.(tst)(SL), ...
            abs(isobase.(['dosm_' tst '_sys'])(SL)), ...
            mksz*fmsm.(srcs{S}),fclr.(srcs{S}),'filled', ...
            'marker',fmrk.(srcs{S}))

        TL = tot >= min(isobase.(tst)(SL)) & tot <= max(isobase.(tst)(SL));
        nl = plot(tot(TL),isoerr_rdm.(srcs{S})(1) ...
            + isoerr_rdm.(srcs{S})(2) ./ (tot(TL) + 0.03), ...
            'color',fclr.(srcs{S}), 'linewidth',1); nolegend(nl)
        
    end %for S

    legend(srcs)
    
    setaxes(gca,8)
    set(gca, 'box','on')
    
    xlabel(['\itm\rm(' tsttit ') / mol\cdotkg^{' endash '1}'])
    
    spfg2 = gca;

spfig.Position = [0.15 0.58 0.6 0.35];
spfg2.Position = [0.15 0.08 0.6 0.35];
% spleg.Position = [0.8 0.63 0.18 0.25];
