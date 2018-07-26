%% Load EMF dataset
% filepath = 'datasets/emf.xlsx';
% iopts = detectImportOptions(filepath, 'numheaderlines',2); 
% emfbase = readtable(filepath,iopts, 'sheet','EMF data', ...
%     'readvariablenames',true);
% emfbase = emfbase(:,1:5);
% 
% clear filepath iopts
fid = fopen('datasets/emfbase1.csv');
head = textscan(fid,repmat('%s',1,17),1, 'delimiter',',');
data = textscan(fid,'%f%s%s%f%f%f%f%f%f%f%f%f%f%f%f%f%f', ...
    'headerlines',1, 'delimiter',',');
fclose(fid);

clear emfbase
for V = 1:numel(head)
    if ~strcmp(head{V}{1},'')
        emfbase.(head{V}{1}) = data{V};
    end %if
end %for
emfbase = struct2table(emfbase);

clear fid head data ans

% Get temperatures
temps = unique(emfbase.t);

tclrs = parula(numel(temps)+1);

%% Plot
figure(1); clf; hold on

for T = 1:numel(temps)

    TL = emfbase.t == temps(T);
    
    plot(emfbase.m(TL),emfbase.emf(TL), 'color',tclrs(T,:))
    
end %for T
    
grid on

%%
figure(2); clf; hold on
for T = 1:numel(temps)

    TL = emfbase.t == temps(T);
    
    plot(emfbase.m(TL),emfbase.ln_acf_racf(TL) ...
        - emfbase.ln_acf_racf_calc(TL), 'marker','o', ...
        'color',tclrs(T,:), 'linewidth',1)
    
end %for T

xlabel('\itm\rm(NaCl) / mol\cdotkg^{-1}')
ylabel('\Delta ln (\gamma / \gamma_{0.1})')
