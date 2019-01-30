% Load pytzer results
pz = load('testfiles/threeway/threeway.mat');
pz.ions = cellstr(pz.ions)';

% Load Simon's results
fid = fopen('testfiles/threeway/FastPitz.Rs1');
fp_data = textscan(fid,repmat('%f',1,29), 'headerlines',162);
fclose(fid);

fp.T = fp_data{1} + 298.15;
fp.P = fp_data{2};

fp.aw  = fp_data{3};
fp.osm = fp_data{4};

fp.acfs = cat(2,fp_data{6:17});
fp.mols = cat(2,fp_data{18:29});

clear ans fid fp_data

% Load David's results
fid = fopen('testfiles/threeway/GIVAKT.csv');
xl_data = textscan(fid,repmat('%f',1,28), 'headerlines',1);
fclose(fid);

xl.aw  = xl_data{1};
xl.osm = xl_data{3};

xl.acfs = cat(2,xl_data{4:15});
xl.mols = cat(2,xl_data{16:27});

%% Compare mols
figure(1); clf
printsetup(gcf,[10 12])

subplot(2,1,2)
    imagesc(pz.mols - fp.mols)

    fcb = colorbar;
    fcb.TickDirection = 'out';
    fcb.Label.String = '\Delta molality / mol/kg';
    fcb.Ticks = -1.5:0.5:1.5;
    fcb.TickLabels = num2str(fcb.Ticks','%.1f');
    fcb.Label.Color = 'K';
    fcb.Label.FontName = 'arial';
    fcb.Label.FontSize = 8;
    fcb.Color = 'k';
    fcb.FontName = 'arial';
    fcb.FontSize = 8;
    
    setaxes(gca,8)
    set(gca, 'XTick',0:12, 'YTick',[1 100:100:600])
    set(gca, 'XTickLabel',pz.ions)
    
    xtickangle(45)
    
    ylabel('Row number')
    
    text(0,1.1,['(b) Molality: pytzer ' endash ' FastPitz'], ...
        'fontname','arial', 'fontsize',9, 'units','normalized')
    
subplot(2,1,1)
    imagesc(xl.mols - fp.mols)
    
    fcb = colorbar;
    fcb.TickDirection = 'out';
    fcb.Label.String = '\Delta molality / mol/kg';
    fcb.Ticks = -1.5:0.5:1.5;
    fcb.TickLabels = num2str(fcb.Ticks','%.1f');
    fcb.Label.Color = 'K';
    fcb.Label.FontName = 'arial';
    fcb.Label.FontSize = 8;
    fcb.Color = 'k';
    fcb.FontName = 'arial';
    fcb.FontSize = 8;
    
    setaxes(gca,8)
    set(gca, 'XTick',0:12, 'YTick',[1 100:100:600])
    set(gca, 'XTickLabel',pz.ions)
    
    xtickangle(45)
    
    ylabel('Row number')
    
    text(0,1.1,['(a) Molality: GIVAKT ' endash ' FastPitz'], ...
        'fontname','arial', 'fontsize',9, 'units','normalized')

colormap(cbrew_ryb(256))
    
print('-r300','testfiles/threeway/molality','-dpng')
