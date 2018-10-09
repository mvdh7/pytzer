%%
ele = 'KCl';

% Simulated datasets
load(['pickles/simloop_vpl_bC_' ele '_100.mat']);
load(['pickles/simloop_vpl_test_' ele '.mat']);
vpltest = readtable(['pickles/simloop_vpl_test_' ele '.csv']);
%
fvar = 'osm';
mksz = 30;

tsrcs = unique(vpltest.src);

% U = 1 + U;
% if U > 20
%     U = 1;
% end %if

Ufile = [];

figtype = 'final';
switch figtype
    case 'ani'
        Ulist = 1:20;
        Usims = true;
        Udata = false;
        Ulines = false;
        Upatch = false;
    case 'final'
        Ulist = 1;
        Usims = false;
        Udata = true;
        Ulines = false;
        Upatch = true;
        Ufile = 99;
    case 'data'
        Ulist = 1;
        Usims = false;
        Udata = true;
        Ulines = true;
        Upatch = false;
        Ufile = 0;
end %switch
        
for U = Ulist

if isempty(Ufile)
    Ufile = U;
end %if
   
figure(3); clf; hold on
printsetup(gcf,[10 7.5])
set(gcf, 'color',[48 48 48]/255, 'inverthardcopy','off')

if Upatch
patch([tot; flipud(tot)],[sqrt(Uosm_sim); flipud(-sqrt(Uosm_sim))], ...
    'y', 'edgecolor','none', 'facealpha',0.5)
patch([tot; flipud(tot)],[sqrt(Uosm_dir); flipud(-sqrt(Uosm_dir))], ...
    'y', 'edgecolor','none', 'facealpha',0.5)

% plot(tot, sqrt(Uosm_sim)*2,'y:')
% plot(tot,-sqrt(Uosm_sim)*2,'y:')
end %if Upatch

% Plot data by source
for S = 1:numel(tsrcs)

    src = tsrcs{S};
    SL = strcmp(vpltest.src,src) & vpltest.t == 298.15;
    
    if Usims
    % osm simulations
    scatter(vpltest.m(SL),osm_sim(SL,U)-vpltest.osm_calc(SL), ...
        mksz,fclr.(src),'filled', 'marker',fmrk.(src), ...
        'markeredgecolor',fclr.(src), ...
        'markerfacealpha',0.8, 'markeredgealpha',1)
    end %if Usims
    
    if Udata
    % osm original dataset
    scatter(vpltest.m(SL),vpltest.osm_meas(SL)-vpltest.osm_calc(SL), ...
        mksz,fclr.(src),'filled', 'marker',fmrk.(src), ...
        'markeredgecolor',fclr.(src), ...
        'markerfacealpha',0.8, 'markeredgealpha',0.1)
    end %if Udata

    if Ulines
    if any(SL)
        Sx = linspace(min(vpltest.m(SL)),max(vpltest.m(SL)),100);
        Sy = vplerr_sys.(ele).(src)(1) ./ Sx ...
            + vplerr_sys.(ele).(src)(2);
        plot(Sx,Sy, 'color',[fclr.(src) 0.8], 'linewidth',1)
    end %if
    end %if Ulines
    
    if Usims
%     osm simulation fits
    plot(tot_fitted,osm_fitted(:,U) - osm_fitted_calc,'y', 'linewidth',2)
    end %if Usims
        
end %for S

% Axis settings
xlim([0 6.5])
setaxes(gca,15)
set(gca, 'color',0.5*[65 65 65]/255, 'xcolor','w', 'ycolor','w', ...
    'linewidth',1)
plot(get(gca,'xlim'),[0 0],'w', 'linewidth',1)
xlabel(['\itm\rm(NaCl) / mol\cdotkg^{' endash '1}'])

ylim(0.0500000001*[-1 1])
set(gca, 'box','off', 'xtick',0:6, 'ytick',-0.1:0.01:0.1)
ylabel('\Delta\phi')

set(gca, 'yticklabel',num2str(get(gca,'ytick')','%.2f'))

print('-r150',['figures/simpytz_vpl/MCsim_KCl' ...
    num2str(Ufile,'%02.0f')],'-dpng')
Ufile = [];

end %for U
