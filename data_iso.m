isobase = readtable('pickles/data_iso.csv');
srcs = unique(isobase.src);

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

