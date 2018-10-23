%% Load databases
vplbase = readtable('pickles/simpar_vpl.csv');
fpdbase = readtable('pickles/simpar_fpd_osm25.csv');
isobase = readtable('pickles/simpar_iso_isobase_tKCl_rNaCl.csv');

%% 
eles = {'NaCl' 'KCl' 'CaCl2'};

errtype = 'sim'; % sim or dir

for E = 2%1:numel(eles)
ele = eles{E};

% ----- ELECTROLYTE SPECIFICS ---------------------------------------------
eletit = ele;
switch ele
    case 'NaCl'
        fxl = 6.26;
        fxt = 0:6;
        fyl = 0.04;
        fyt = -.04:.01:.04;
    case 'KCl'
        fxl = 4.84;
        fxt = 0:6;
        fyl = 0.05;
        fyt = -.05:.01:.05;
    case 'CaCl2'
        fxl = 8;
        fxt = 0:10;
        fyl = 0.2;
        fyt = -.2:.05:.2;
        eletit = 'CaCl_2';
end %switch

clrvpl = [255 140 0]/255;
clrfpd = [0 191 255]/255;
clriso = [255 105 180]/255;

VL = strcmp(vplbase.ele,ele);
FL = strcmp(fpdbase.ele,ele);

vplu = load(['pickles/simloop_vpl_bC_' ele '_1000.mat']);
fpdu = load(['pickles/simloop_fpd_osm25_bC_' ele '_100.mat']);
isou = load('pickles/simloop_iso_bC_tKCl_rNaCl_10.mat');

varf = (1./fpdu.(['Uosm_' errtype]) + 1./vplu.(['Uosm_' errtype])) ...
    .* (fpdu.(['Uosm_' errtype]) .* vplu.(['Uosm_' errtype]) ...
    ./ (fpdu.(['Uosm_' errtype]) + vplu.(['Uosm_' errtype]))).^2;

% ===== BEGIN FIGURE ======================================================
figure(8); clf; hold on
printsetup(gcf,[9 6])

% ----- PATCHES -----------------------------------------------------------
patch([vplu.tot; flipud(vplu.tot)], ...
    [sqrt(vplu.(['Uosm_' errtype])); 
    flipud(-sqrt(vplu.(['Uosm_' errtype])))], ...
    clrvpl, 'edgecolor','none', 'facealpha',0.2)
patch([fpdu.tot; flipud(fpdu.tot)], ...
    [sqrt(fpdu.(['Uosm_' errtype])); 
    flipud(-sqrt(fpdu.(['Uosm_' errtype])))], ...
    clrfpd, 'edgecolor','none', 'facealpha',0.2)
patch([isou.tot; flipud(isou.tot)], ...
    [sqrt(isou.(['Uosm_' errtype])); 
    flipud(-sqrt(isou.(['Uosm_' errtype])))], ...
    clriso, 'edgecolor','none', 'facealpha',0.3)

% ----- PATCH OUTLINES ----------------------------------------------------
x = plot(vplu.tot, sqrt(vplu.(['Uosm_' errtype])), 'color',clrvpl);
x.Color = [x.Color 0.6];
x = plot(vplu.tot,-sqrt(vplu.(['Uosm_' errtype])), 'color',clrvpl);
x.Color = [x.Color 0.6];

x = plot(fpdu.tot, sqrt(fpdu.(['Uosm_' errtype])), 'color',clrfpd);
x.Color = [x.Color 0.6];
x = plot(fpdu.tot,-sqrt(fpdu.(['Uosm_' errtype])), 'color',clrfpd);
x.Color = [x.Color 0.6];

x = plot(isou.tot, sqrt(isou.(['Uosm_' errtype])), 'color',clriso);
x.Color = [x.Color 0.6];
x = plot(isou.tot,-sqrt(isou.(['Uosm_' errtype])), 'color',clriso);
x.Color = [x.Color 0.6];

plot(vplu.tot, sqrt(varf),'k:')
plot(vplu.tot,-sqrt(varf),'k:')

% ----- SCATTER DATA ------------------------------------------------------
mksz = 5;
mkfa = 0.7;
scatter(vplbase.m(VL),vplbase.dosm25(VL),mksz*0.8,clrvpl,'filled', ...
    'markerfacealpha',mkfa, 'marker','o')
scatter(fpdbase.m(FL),fpdbase.dosm25(FL),mksz,clrfpd,'filled', ...
    'markerfacealpha',mkfa, 'marker','v')
if strcmp(ele,'KCl')
scatter(isobase.(ele),isobase.(['dosm_' ele]),mksz,clriso,'filled', ...
    'markerfacealpha',mkfa, 'marker','sq')
end %if

% ----- AXIS SETTINGS -----------------------------------------------------
xlim([0 fxl])
ylim(fyl*[-1 1])

% plot(get(gca,'xlim'),[0 0],'k')

setaxes(gca,8)
set(gca, 'box','on', 'xtick',fxt, 'ytick',fyt)
set(gca, 'yticklabel',num2str(fyt','%.2f'))

xlabel(['\itm\rm(' eletit ') / mol\cdotkg^{' endash '1}'])
ylabel('\Delta\phi_{25}')

set(gca, 'position',[0.2 0.2 0.7 0.7])

% ----- SAVE TO FILE ------------------------------------------------------
print('-r300',['figures/simloops_' ele '_' errtype],'-dpng')

end %for E
