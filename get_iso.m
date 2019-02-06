ele = 'NaCl';

clear t
t = load(['testfiles/isonew_' ele '.mat']);
t.elemix = cellstr(t.elemix);
t = struct2table(t);

L = true(height(t),1) & t.T <= 323.15;

elemixs = unique(t.elemix);

% Rank best to worst
ele_delaw = NaN(size(elemixs));
for E = 1:numel(elemixs)
    EL = L & strcmp(t.elemix,elemixs{E});
    ele_delaw(E) = nanstd_Sn(t.delaw(EL));
end %for E
[ele_delaw,eleix] = sort(ele_delaw);
elemixs = elemixs(eleix);

% Get subplot arrangement
spsf = sqrt(numel(elemixs) / goldratio);
spsf(2) = spsf * goldratio;
sps = floor(spsf);
if prod(sps) < numel(elemixs)
    sps = [ceil(spsf(1)) floor(spsf(2))];
end %if
if prod(sps) < numel(elemixs)
    sps = ceil(spsf);
end %if

fylimmin = 1e-3;

figure(1); clf
printsetup(gcf,sps*2)

for E = 1:numel(elemixs)

subplot(sps(1),sps(2),E); hold on

    EL = L & strcmp(t.elemix,elemixs{E});

    xlim([0 max(t.testtot)])
    plot(get(gca,'xlim'),[0 0],'k')
    
    if any(EL)
    
        scatter(t.testtot(EL),t.delaw(EL)*1e3,10,t.T(EL),'filled')

        
        ylim(max(max(abs(t.delaw(EL))),fylimmin)*[-1 1]*1.1e3)
        

    %     if max(abs(t.delaw(EL))) > fylimmin
            plot(get(gca,'xlim'), fylimmin*[1 1]*1e3, 'color',[1 0.5 0.5])
            plot(get(gca,'xlim'),-fylimmin*[1 1]*1e3, 'color',[1 0.5 0.5])
    %     end %if

        caxis([273.15 323.15])
        
    end %if
        
    xlabel(['\itm\rm(' ele ')'])
    ylabel('\Delta\ita_w\rm \times 10^{3}')

    title(elemixs{E})

    setaxes(gca,8)
    set(gca, 'box','on')

end %for E
