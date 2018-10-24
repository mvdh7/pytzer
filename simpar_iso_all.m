load('pickles/simpar_iso_all.mat')

figure(1); clf; hold on
printsetup(gcf,[9 6])

bw = 1;
histogram(isoerr_sys.all,(-8:bw:8)*1e-3, 'normalization','count', ...
    'facecolor',0.3*[1 1 1])

MAD = median(abs(isoerr_sys.all - median(isoerr_sys.all)));
sd_MAD = MAD * 1.4826;

% Sn0 = NaN(numel(isoerr_sys.all));
% Sn1 = NaN(numel(isoerr_sys.all),1);
% for C = 1:numel(isoerr_sys.all)
%     Sn0(:,C) = abs(isoerr_sys.all(C) - isoerr_sys.all);
%     Sn1(C) = median(Sn0(Sn0(:,C) ~= 0,C));
% end %for C
% sd_Sn = 1.1926 * median(Sn1);

mu = 0;
fx = (-9:0.01:9)*1e-3;
fy = normpdf(fx,mu,isoerr_sys.all_Sn) ...isoerr_sys.all_qsd) ...
    * bw*1e-3 * numel(isoerr_sys.all);

% ld = isoerr_sys.all_laplace;
% fy1 = exp(-abs(fx-mu)/ld)/(2*ld) * numel(isoerr_sys.all) ...
%     * bw*1e-3; % laplace

plot(fx,fy,'k', 'linewidth',1)
% plot(fx,fy1,'k--', 'linewidth',1)

xlim(9e-3*[-1 1])
ylim([0 16])

setaxes(gca,8)
set(gca, 'box','on', 'xtick',(-8:2:8)*1e-3, 'ytick',0:4:16)
set(gca, 'xticklabel',num2str(get(gca,'xtick')'*1e3))

xlabel('\delta_{IE} \times 10^3')
ylabel('Number of datasets')

print('-r300','figures/simpar_iso_all','-dpng')
