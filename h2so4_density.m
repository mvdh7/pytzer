% Training data obtained on 2018-07-03 from
%  http://www.aim.env.uea.ac.uk/aim/density/density_electrolyte.php

% mol(arity) in mol/L
mol = [0; 1; 2; 3; 4; 5; 6; 7;  % 273.15 K
       0; 1; 2; 3; 4; 5; 6; 7;  % 298.15 K
       0; 1; 2; 3; 4; 5; 6; 7]; % 323.15 K
   
% dens(ity) in g/cm3
dens = [0.9998395; 1.067264; 1.129999; 1.189254; 1.245142; 1.298313;
        1.350050 ; 1.401148;  % 273.15 K
        0.9970449; 1.058814; 1.118045; 1.175312; 1.230379; 1.283444;
        1.335313 ; 1.386454;  % 298.15 K
        0.9880363; 1.047443; 1.105118; 1.161359; 1.215849; 1.268863;
        1.320940 ; 1.372211]; % 323.15 K
    
% temp(erature) in K
temp = [273.15; 273.15; 273.15; 273.15; 273.15; 273.15; 273.15; 273.15;
        298.15; 298.15; 298.15; 298.15; 298.15; 298.15; 298.15; 298.15;
        323.15; 323.15; 323.15; 323.15; 323.15; 323.15; 323.15; 323.15];

% Do fitting
ftype = 'purequadratic';
fitFULL = regstats(dens,[mol temp],ftype,{'beta' 'yhat' 'adjrsquare'});

% Round fit results for use in Excel etc.
fitRND = fitFULL;
fitRND.beta = round(fitFULL.beta,5,'significant');

% Plot results
fx = (0:0.01:7)';

figure(1); clf

subplot(2,1,1); hold on
    scatter(mol,dens,50,temp,'filled');
    plot(fx,x2fx([fx 273.15*ones(size(fx))],ftype)*fitRND.beta,'k');
    plot(fx,x2fx([fx 298.15*ones(size(fx))],ftype)*fitRND.beta,'k');
    plot(fx,x2fx([fx 323.15*ones(size(fx))],ftype)*fitRND.beta,'k');
    grid on

subplot(2,1,2); hold on
    scatter(mol,x2fx([mol temp],ftype)*fitRND.beta - dens, ...
        50,temp,'filled');
    plot(get(gca,'xlim'),[0 0],'k');
    grid on
