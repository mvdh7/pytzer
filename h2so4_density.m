mol = [0; 1; 2; 3; 4; 5; 6; 7;
       0; 1; 2; 3; 4; 5; 6; 7;
       0; 1; 2; 3; 4; 5; 6; 7];
dens = [0.9998395; 1.067264; 1.129999; 1.189254; 1.245142; 1.298313;
        1.350050; 1.401148;
        0.9970449; 1.058814; 1.118045; 1.175312; 1.230379; 1.283444;
        1.335313; 1.386454;
        0.9880363; 1.047443; 1.105118; 1.161359; 1.215849; 1.268863;
        1.320940; 1.372211];
tmp = [273.15; 273.15; 273.15; 273.15; 273.15; 273.15; 273.15; 273.15;
       298.15; 298.15; 298.15; 298.15; 298.15; 298.15; 298.15; 298.15;
       323.15; 323.15; 323.15; 323.15; 323.15; 323.15; 323.15; 323.15];

ftype = 'purequadratic';

fit25 = regstats(dens,[mol tmp],ftype,{'beta' 'yhat' 'adjrsquare'});

fx = (0:0.01:7)';

figure(1); clf

subplot(2,1,1); hold on
    scatter(mol,dens,50,tmp,'filled');
    plot(fx,x2fx([fx 273.15*ones(size(fx))],ftype)*fit25.beta,'k');
    plot(fx,x2fx([fx 298.15*ones(size(fx))],ftype)*fit25.beta,'k');
    plot(fx,x2fx([fx 323.15*ones(size(fx))],ftype)*fit25.beta,'k');
    grid on

subplot(2,1,2); hold on
    scatter(mol,x2fx([mol tmp],ftype)*fit25.beta - dens,50,tmp,'filled');
    plot(get(gca,'xlim'),[0 0],'k');
    grid on
