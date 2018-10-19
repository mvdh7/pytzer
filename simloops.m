vplbase = readtable('pickles/simpar_vpl.csv');
fpdbase = readtable('pickles/simpar_fpd.csv');

%%
eles = {'NaCl' 'KCl' 'CaCl2'};

for E = 1:numel(eles)
ele = eles{E};

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

clrvpl = [1 0.5 0];
clrfpd = [0 0.8 1];

VL = strcmp(vplbase.ele,ele);
FL = strcmp(fpdbase.ele,ele);

vplu = load(['pickles/simloop_vpl_bC_' ele '_1000.mat']);
fpdu = load(['pickles/simloop_fpd_bC_' ele '_1000.mat']);

varf = (1./fpdu.Uosm_sim + 1./vplu.Uosm_sim) ...
    .* (fpdu.Uosm_sim .* vplu.Uosm_sim ...
    ./ (fpdu.Uosm_sim + vplu.Uosm_sim)).^2;

figure(1); clf; hold on
printsetup(gcf,[9 6])

patch([vplu.tot; flipud(vplu.tot)], ...
    [sqrt(vplu.Uosm_sim); flipud(-sqrt(vplu.Uosm_sim))], ...
    clrvpl, 'edgecolor','none', 'facealpha',0.3)
patch([fpdu.tot; flipud(fpdu.tot)], ...
    [sqrt(fpdu.Uosm_sim); flipud(-sqrt(fpdu.Uosm_sim))], ...
    clrfpd, 'edgecolor','none', 'facealpha',0.3)

x = plot(vplu.tot, sqrt(vplu.Uosm_sim), 'color',clrvpl);
x.Color = [x.Color 0.6];
x = plot(vplu.tot,-sqrt(vplu.Uosm_sim), 'color',clrvpl);
x.Color = [x.Color 0.6];

x = plot(fpdu.tot, sqrt(fpdu.Uosm_sim), 'color',clrfpd);
x.Color = [x.Color 0.6];
x = plot(fpdu.tot,-sqrt(fpdu.Uosm_sim), 'color',clrfpd);
x.Color = [x.Color 0.6];

plot(vplu.tot, sqrt(varf),'k:')
plot(vplu.tot,-sqrt(varf),'k:')

mksz = 5;
mkfa = 0.7;
scatter(vplbase.m(VL),vplbase.dosm25(VL),mksz*0.8,clrvpl,'filled', ...
    'markerfacealpha',mkfa, 'marker','o')
scatter(fpdbase.m(FL),fpdbase.dosm25(FL),mksz,clrfpd,'filled', ...
    'markerfacealpha',mkfa, 'marker','v')

xlim([0 fxl])
ylim(fyl*[-1 1])

plot(get(gca,'xlim'),[0 0],'k')

setaxes(gca,8)
set(gca, 'box','on', 'xtick',fxt, 'ytick',fyt)
set(gca, 'yticklabel',num2str(fyt','%.2f'))

xlabel(['\itm\rm(' eletit ') / mol\cdotkg^{' endash '1}'])
ylabel('\Delta\phi_{25}')

set(gca, 'position',[0.2 0.2 0.7 0.7])

print('-r300',['figures/simloops_' ele],'-dpng')

end %for E
