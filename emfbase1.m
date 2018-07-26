%% Load EMF dataset
filepath = 'datasets/emf.xlsx';
iopts = detectImportOptions(filepath, 'numheaderlines',2); 
emfbase = readtable(filepath,iopts, 'sheet','EMF data', ...
    'readvariablenames',true);
emfbase = emfbase(:,1:5);

clear filepath iopts

% Get temperatures
temps = unique(emfbase.t);

%% Plot
figure(1); clf; hold on

for T = 1:numel(temps)

    TL = emfbase.t == temps(T);
    
    plot(emfbase.m(TL),emfbase.emf(TL))
    
end %for T
    
grid on
