% Load fortest output
load('pickles/fortest_CaCl2.mat')

% Move into table
clear rc97
rc97.temp   = 298.15*ones(size(tot));
rc97.tot    = tot;
rc97.osm    = osm;
rc97.aw     = NaN(size(tot));
rc97.acfPM  = exp(ln_acfPM);
rc97.dissoc = alpha;
rc97 = struct2table(rc97);

% Compare with real values
t11 = readtable('datasets/RC97 Table 11.xlsx');
