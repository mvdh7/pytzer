%% Import results
filepath = 'testfiles/';
filestem = 'pytzerPitzer';

SIGFIGS = 12;

py = readtable([filepath filestem '_py.csv']);
jl = readtable([filepath filestem '_jl.csv']);

% Get list of ions
gions = py.Properties.VariableNames;
gions = gions(~cellfun(@isempty,regexp(gions,'^g')));

ispl = regexp(gions,'^g','split');
ions = cell(size(gions));
for I = 1:numel(ions)
    ions{I} = ispl{I}{2};
end %for I
clear ispl I

py.mols = table2array(py(:,ions));
jl.mols = table2array(jl(:,ions));

py.acfs = round(table2array(py(:,gions)),SIGFIGS);
jl.acfs = round(table2array(jl(:,gions)),SIGFIGS);

% Calculate differences
fdiff = abs(jl.acfs - py.acfs); % pytzer vs Pitzer.jl
% fdosm = jl.osm - py.osm;

% Plot results
figure(3); clf
printsetup(gcf,[8 8])

subplot(1,4,1:3)

    % Everything
    u = imagesc(-1.5*ones(size(fdiff)));

    hold on

    % Values disagree
    v = imagesc(-0.5*ones(size(fdiff)));
    v.AlphaData = fdiff > 0;

    % Molality > 0
    x = imagesc(0.5*ones(size(fdiff)));
    x.AlphaData = py.mols > 0;

    % Values agree, molality > 0
    y = imagesc(1.5*ones(size(fdiff)));
    y.AlphaData = py.mols > 0 & fdiff == 0;

    colormap([0.8 0.9 1;
              1 0.6 0.6;
              0.9 0.1 0.1;
              0.6 0.8 1])

    % colorbar

    setaxes(gca,8)
    set(gca, 'XTick',1:12)%, 'YTick',[1 100:100:600])
    set(gca, 'XTickLabel',ions)

    xtickangle(45)

    ylabel('Row number')

% subplot(1,4,4); hold on
% 
%     xlim([0.5 numel(pz.osm)+0.5])
%     ylim(0.0000100001 * [-1 1])
% 
%      bar(1:numel(fp.osm),fdosm, ...
%         'facecolor','r', 'edgecolor','r')
%     plot(get(gca,'xlim'),[0 0],'k')
% 
%     setaxes(gca,8)
%     set(gca, ...'XTick',[1 100:100:600], ...'YTick',-0.15:0.05:0.15, ...
%         'box','on', 'XAxisLocation','top')
% %     set(gca, 'YTickLabel',num2str(get(gca,'ytick')','%.2f'))
%     
%     xlabel('Row number')
%     ylabel('\Delta osm')
%     
% %     text(0,1.1,['(b) ' fvar ': pytzer ' endash ' FastPitz'], ...
% %         'fontname','arial', 'fontsize',9, 'units','normalized')
%     
%     view(90,90)
    
% Get activity discrepancies

badacfix = find(any(abs(fdiff) > 0,2));

badamols  = py.mols  (badacfix,:);
bad_diffs = fdiff(badacfix,:);

clc
for R = 1:size(badamols,1)
    RL = badamols(R,:) > 0;
    EL = abs(bad_diffs(R,:)) > 0;
    disp([num2str(badacfix(R)) ' / ions are: ' strjoin(ions(RL)) ...
        ' / errors in: ' strjoin(ions(EL)) ' / errors: ' ...
        num2str(bad_diffs(R,EL))] )
end %for R
