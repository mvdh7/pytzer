% Load data from simpar_iso.py
isobase = readtable('pickles/simpar_iso_isobase_tKCl_rNaCl.csv');
load('pickles/simpar_iso_pshape_tKCl_rNaCl.mat')
isoerr_sys = load('pickles/simpar_iso_isoerr_sys_tKCl_rNaCl.mat');

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

subplot(2,1,1); hold on

grid on

xlim([0 5])
ylim(0.012*[-1 1])


for S = 1:numel(srcs)

    SL = strcmp(isobase.src,srcs{S});
    
%     scatter(isobase.NaCl(SL), ...
%         isobase.aw_ref_NaCl(SL) - isobase.aw_ref_KCl(SL))
    
    scatter(isobase.KCl(SL),isobase.dosm_KCl(SL),mksz*fmsm.(srcs{S}), ...
        fclr.(srcs{S}),'filled', 'marker',fmrk.(srcs{S}))
    
    TL = tot >= min(isobase.KCl(SL)) & tot <= max(isobase.KCl(SL));
    plot(tot(TL),isoerr_sys.(srcs{S}) * (1 + 1./(tot(TL) + 0.03)), ...
        'color',fclr.(srcs{S}), 'linewidth',1)
    
%     scatter(isobase.NaCl(SL), ...
%         isobase.osm_meas_NaCl(SL) - isobase.osm_ref_NaCl(SL))

end %for S

% %% Components
% plot(tot,dosm_dtot *5e-3, 'linewidth',2)
% plot(tot,dosm_dtotR*1e-2, 'linewidth',2)
% plot(tot,(dosm_dtot + dosm_dtotR)*5e-3, 'linewidth',2)
% plot(tot,4e-4*(1 + 1./(tot+0.03)),'k', 'linewidth',2)

subplot(2,1,2); hold on

for S = 1:numel(srcs)

    SL = strcmp(isobase.src,srcs{S});
    
    scatter(isobase.KCl(SL),abs(isobase.dosm_KCl_sys(SL)))
    
end %for S
