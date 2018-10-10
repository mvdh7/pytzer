isobase = readtable('pickles/data_iso.csv');
srcs = unique(isobase.src);
load('pickles/data_iso.mat');

figure(1); clf; hold on

for S = 1:numel(srcs)

    SL = strcmp(isobase.src,srcs{S});
    
%     scatter(isobase.NaCl(SL), ...
%         isobase.aw_ref_NaCl(SL) - isobase.aw_ref_KCl(SL))
    
    scatter(isobase.KCl(SL), ...
        isobase.osm_meas_KCl(SL) - isobase.osm_ref_KCl(SL))
    
%     scatter(isobase.NaCl(SL), ...
%         isobase.osm_meas_NaCl(SL) - isobase.osm_ref_NaCl(SL))
    
end %for S

grid on

xlim([0 5])
ylim(0.015*[-1 1])

plot(tot,dosm_dtot *5e-3)
plot(tot,dosm_dtotR*1e-2)
plot(tot,(dosm_dtot + dosm_dtotR)*5e-3)
