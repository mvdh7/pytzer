%% Load Python outputs
fpdbase = readtable('pickles/simpytz_fpd.csv');

EL = strcmp(fpdbase.ele,'MgCl2');

figure(2); clf

subplot(2,1,1); hold on
    scatter(fpdbase.m(EL),fpdbase.fpd(EL))
    plot(fpdbase.m(EL),fpdbase.fpd_calc(EL))
    grid on

subplot(2,1,2); hold on
    scatter(fpdbase.m(EL),fpdbase.fpd(EL) - fpdbase.fpd_calc(EL))
    grid on
    