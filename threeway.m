% Load pytzer results
pz = load('testfiles/threeway/threeway.mat');
pz.ions = cellstr(pz.ions)';

pz.acfs = round(pz.acfs,6,'significant');
pz.osm = round(pz.osm,6,'significant');

% Load Simon's results
fid = fopen('testfiles/threeway/FastPitz.Rs1');
fp_data = textscan(fid,repmat('%f',1,29), 'headerlines',162);
fclose(fid);

fp.T = fp_data{1} + 298.15;
fp.P = fp_data{2};

fp.aw  = fp_data{3};
fp.osm = fp_data{4};

fp.I = fp_data{5};

fp.acfs = cat(2,fp_data{6:17});
fp.mols = cat(2,fp_data{18:29});

clear ans fid fp_data

% % Load David's results
% fid = fopen('testfiles/threeway/GIVAKT.csv');
% xl_data = textscan(fid,repmat('%f',1,28), 'headerlines',1);
% fclose(fid);
% 
% xl.aw  = xl_data{1};
% xl.osm = xl_data{3};
% 
% xl.acfs = cat(2,xl_data{4:15});
% xl.mols = cat(2,xl_data{16:27});

%% Load David & Simon v2
% num = xlsread('testfiles\threeway\Comparison.xlsx','FastPitzout');
% fp.osm  = num(:,1);
% fp.aw   = num(:,3);
% fp.acfs = num(:,4:15);
% fp.mols = num(:,16:27);

num = xlsread('testfiles/threeway/Comparison.xlsx','GIVAKT');
xl.osm  = num(:,1);
xl.aw   = num(:,3);
xl.acfs = num(:,4:15);
xl.mols = num(:,16:27);

clear num

%% =================================================== (1) Compare mols ===

xl.acfs(fp.mols == 0) = fp.acfs(fp.mols == 0);

fvar = 'acfs';

figure(1); clf
printsetup(gcf,[10 12])

subplot(2,1,1)
    imagesc(xl.(fvar) - fp.(fvar))
    
    fcb = colorbar;
    fcb.TickDirection = 'out';
    fcb.Label.String = ['\Delta ' fvar ' / mol/kg'];
    fcb.Label.Color = 'K';
    fcb.Label.FontName = 'arial';
    fcb.Label.FontSize = 8;
    fcb.Color = 'k';
    fcb.FontName = 'arial';
    fcb.FontSize = 8;

    switch fvar
        case 'mols'
            caxis(1.5*[-1 1])
            fcb.Ticks = -1.5:0.5:1.5;
            fcb.TickLabels = num2str(fcb.Ticks','%.1f');
        case 'acfs'
            caxis(1e-2*[-1 1])
            fcb.Ticks = -0.01:0.005:0.01;
            fcb.TickLabels = num2str(fcb.Ticks','%.3f');
    end %switch
    
    setaxes(gca,8)
    set(gca, 'XTick',0:12, 'YTick',[1 100:100:600])
    set(gca, 'XTickLabel',pz.ions)
    
    xtickangle(45)
    
    ylabel('Row number')
    
    text(0,1.1,['(a) ' fvar ': GIVAKT ' endash ' FastPitz'], ...
        'fontname','arial', 'fontsize',9, 'units','normalized')

subplot(2,1,2)
    imagesc(pz.(fvar) - fp.(fvar))

    fcb = colorbar;
    fcb.TickDirection = 'out';
    fcb.Label.String = ['\Delta ' fvar ' / mol/kg'];
    fcb.Label.Color = 'K';
    fcb.Label.FontName = 'arial';
    fcb.Label.FontSize = 8;
    fcb.Color = 'k';
    fcb.FontName = 'arial';
    fcb.FontSize = 8;
    
    switch fvar
        case 'mols'
            caxis(1.5*[-1 1])
            fcb.Ticks = -1.5:0.5:1.5;
            fcb.TickLabels = num2str(fcb.Ticks','%.1f');
        case 'acfs'
            caxis(1e-2*[-1 1])
            fcb.Ticks = -0.01:0.005:0.01;
            fcb.TickLabels = num2str(fcb.Ticks','%.3f');
    end %switch 
        
    setaxes(gca,8)
    set(gca, 'XTick',0:12, 'YTick',[1 100:100:600])
    set(gca, 'XTickLabel',pz.ions)
    
    xtickangle(45)
    
    ylabel('Row number')
    
    text(0,1.1,['(b) ' fvar ': pytzer ' endash ' FastPitz'], ...
        'fontname','arial', 'fontsize',9, 'units','normalized')

colormap(cbrew_ryb(256))
    
print('-r300',['testfiles/threeway/' fvar],'-dpng')

%%
fvar = 'osm';

figure(2); clf
printsetup(gcf,[10 12])

subplot(2,1,1); hold on

    xlim([1 numel(xl.(fvar))])
    ylim(0.15 * [-1 1])

    bar(1:numel(xl.(fvar)),xl.(fvar) - fp.(fvar), ...
        'facecolor','r', 'edgecolor','r')
    plot(get(gca,'xlim'),[0 0],'k')

    setaxes(gca,8)
    set(gca, 'XTick',[1 100:100:600], 'YTick',-0.15:0.05:0.15, 'box','on')
    set(gca, 'YTickLabel',num2str(get(gca,'ytick')','%.2f'))
    
    xlabel('Row number')
    ylabel(['\Delta ' fvar])
    
    text(0,1.1,['(a) ' fvar ': GIVAKT ' endash ' FastPitz'], ...
        'fontname','arial', 'fontsize',9, 'units','normalized')
    
subplot(2,1,2); hold on

    xlim([1 numel(xl.(fvar))])
    ylim(0.1500001 * [-1 1])

     bar(1:numel(fp.(fvar)),pz.(fvar) - fp.(fvar), ...
        'facecolor','r', 'edgecolor','r')
    plot(get(gca,'xlim'),[0 0],'k')

    setaxes(gca,8)
    set(gca, 'XTick',[1 100:100:600], 'YTick',-0.15:0.05:0.15, 'box','on')
    set(gca, 'YTickLabel',num2str(get(gca,'ytick')','%.2f'))
    
    xlabel('Row number')
    ylabel(['\Delta ' fvar])
    
    text(0,1.1,['(b) ' fvar ': pytzer ' endash ' FastPitz'], ...
        'fontname','arial', 'fontsize',9, 'units','normalized')
    
print('-r300',['testfiles/threeway/' fvar],'-dpng')
    
%% Get osmotic coefficient discrepancies (pytzer vs FastPitz)

badosm = find(abs(pz.osm - fp.osm) > 1e-4);

badmols = pz.mols(badosm,:);

clc
for R = 1:size(badmols,1)
    RL = badmols(R,:) > 0;
    disp([num2str(badosm(R)) ' ' strjoin(pz.ions(RL))] )
end %for R

%% Get activity discrepancies (pytzer vs FastPitz)

acfs_diff = pz.acfs - fp.acfs;

badacfix = find(any(abs(acfs_diff) > 0,2));

badamols  = pz.mols  (badacfix,:);
bad_diffs = acfs_diff(badacfix,:);

clc
for R = 1:size(badamols,1)
    RL = badamols(R,:) > 0;
    EL = abs(bad_diffs(R,:)) > 0;
    disp([num2str(badacfix(R)) ' / ions are: ' strjoin(pz.ions(RL)) ...
        ' / errors in: ' strjoin(pz.ions(EL)) ' / errors: ' ...
        num2str(bad_diffs(R,EL))] )
end %for R

%%
figure(3); clf
printsetup(gcf,[8 8])

fdiff = abs(xl.acfs - fp.acfs);

x = imagesc(0.5*ones(size(fp.mols)));
x.AlphaData = fp.mols > 0;

hold on

y = imagesc(1.5*ones(size(fp.mols)));
y.AlphaData = fdiff > 0;

colormap([0.1 0.6 1;
          1 0.4 0.4])

setaxes(gca,8)
set(gca, 'XTick',0:12, 'YTick',[1 100:100:600])
set(gca, 'XTickLabel',pz.ions)

xtickangle(45)

ylabel('Row number')
      