load('pickles/bCderivs.mat');
sqt = sqrt(tot);

ftot = sqt;

figure(1); clf

subplot(2,1,1); hold on

    plot(ftot,osm)

subplot(2,1,2); hold on

    plot(ftot,db0 / max(db0))
    plot(ftot,db1 / max(db1))
    plot(ftot,dC0 / max(dC0))
    plot(ftot,dC1 / max(dC1))
